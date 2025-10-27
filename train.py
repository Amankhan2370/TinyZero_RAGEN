#!/usr/bin/env python3
"""
TinyZero with A*PO - Main Training Script
Advanced implementation with distributed training support.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from pathlib import Path

from models.transformer import TransformerLM
from models.fsdp_wrapper import setup_fsdp_model, FSDPConfig
from environments.countdown import CountdownEnvironment
from environments.multiplication import MultiplicationEnvironment
from algorithms.astarpo import AStarPO, AStarPOConfig
from algorithms.trajectory import TrajectoryCollector
from utils.logging import setup_logging, WandbLogger
from utils.distributed import setup_distributed, cleanup_distributed, all_reduce_dict, is_main_process

class TinyZeroTrainer:
    """Main trainer class for TinyZero with A*PO."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = config_path
        self.load_config()
        
        # Setup distributed training
        self.local_rank = setup_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        # Initialize components
        self.setup_models()
        self.setup_environments()
        self.setup_algorithm()
        self.setup_optimizer()
        
        # Training state
        self.iteration = 0
        self.best_reward = -float('inf')
        self.trajectory_collector = TrajectoryCollector()
        
        # Setup logging (only on main process)
        if is_main_process():
            self.logger, self.wandb_logger = setup_logging(
                log_dir=self.config['experiment']['log_dir'],
                experiment_name=self.config['experiment']['name'],
                use_wandb=self.config['logging']['use_wandb'],
                wandb_config=self.config
            )
            self.logger.log_experiment_config(self.config)
        else:
            self.logger = None
            self.wandb_logger = None
            
    def load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def setup_models(self):
        """Initialize policy and reference models."""
        model_config = self.config['model']
        
        # Policy model (trainable)
        self.policy_model = TransformerLM(
            vocab_size=model_config['vocab_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            mlp_ratio=model_config['mlp_ratio'],
            dropout=model_config['dropout']
        )
        
        # Reference model (frozen)
        self.reference_model = TransformerLM(
            vocab_size=model_config['vocab_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            mlp_ratio=model_config['mlp_ratio'],
            dropout=model_config['dropout']
        )
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
            
        # Wrap with FSDP if enabled
        if self.config['distributed']['use_fsdp']:
            fsdp_config = FSDPConfig(self.config['distributed'])
            self.policy_model = setup_fsdp_model(self.policy_model, fsdp_config)
            self.reference_model = setup_fsdp_model(self.reference_model, fsdp_config)
        else:
            self.policy_model = self.policy_model.to(self.device)
            self.reference_model = self.reference_model.to(self.device)
            
    def setup_environments(self):
        """Initialize training environments."""
        env_config = self.config['environments']
        
        self.countdown_env = CountdownEnvironment(env_config['countdown'])
        self.multiplication_env = MultiplicationEnvironment(env_config['multiplication'])
        
        self.environments = {
            'countdown': self.countdown_env,
            'multiplication': self.multiplication_env
        }
        
    def setup_algorithm(self):
        """Initialize A*PO algorithm."""
        astarpo_config = AStarPOConfig(**self.config['astarpo'])
        self.astarpo = AStarPO(astarpo_config, self.device)
        
    def setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        training_config = self.config['training']
        
        # Filter out parameters that don't require gradients
        params = [p for p in self.policy_model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            params,
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=training_config['total_iterations']
        )
        
    def collect_trajectories(self) -> list:
        """Collect trajectories using current policy."""
        self.trajectory_collector.clear_completed()
        
        num_trajectories = self.config['astarpo']['num_trajectories']
        trajectories_per_env = num_trajectories // len(self.environments)
        
        for env_name, env in self.environments.items():
            for episode in range(trajectories_per_env):
                trajectory_id = self.trajectory_collector.start_trajectory(
                    env_name, env.reset()
                )
                
                state = env.reset()
                done = False
                step_count = 0
                
                while not done and step_count < self.config['training']['max_trajectory_length']:
                    # Generate action using policy
                    action, log_prob = self.generate_action(state)
                    
                    # Execute action in environment
                    next_state, reward, done, info = env.step(action)
                    
                    # Create trajectory step
                    from algorithms.trajectory import TrajectoryStep
                    step = TrajectoryStep(
                        state=state,
                        action=action,
                        reward=reward,
                        log_prob=log_prob,
                        done=done,
                        info=info
                    )
                    
                    # Add to trajectory
                    self.trajectory_collector.add_step(trajectory_id, step)
                    
                    state = next_state
                    step_count += 1
                    
        return self.trajectory_collector.get_completed_trajectories()
    
    def generate_action(self, state: str):
        """Generate action using current policy."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            
            # Tokenize state
            inputs = tokenizer(state, return_tensors="pt").to(self.device)
            prompt_length = inputs['input_ids'].shape[1]
            
            # Generate completion
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_length=prompt_length + 50,
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Extract generated tokens
            generated_tokens = outputs.sequences[0, prompt_length:]
            action = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Compute log probability
            log_probs = []
            for i, step_scores in enumerate(outputs.scores):
                if i >= len(generated_tokens):
                    break
                step_log_probs = F.log_softmax(step_scores[0], dim=-1)
                token_log_prob = step_log_probs[generated_tokens[i]]
                log_probs.append(token_log_prob)
            
            total_log_prob = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0)
            
            return action, total_log_prob
            
        except Exception as e:
            # Fallback action
            return "1 + 1", torch.tensor(0.0)
    
    def compute_reward(self, prompt: str, completion: str) -> float:
        """Compute reward for prompt-completion pair."""
        # Simple reward function - extend based on environment type
        if "countdown" in prompt.lower():
            return self.evaluate_countdown_completion(prompt, completion)
        elif "multiplication" in prompt.lower():
            return self.evaluate_multiplication_completion(prompt, completion)
        else:
            return 0.0
    
    def evaluate_countdown_completion(self, prompt: str, completion: str) -> float:
        """Evaluate countdown completion reward."""
        try:
            import re
            # Extract target from prompt
            target_match = re.search(r'Target: (\d+)', prompt)
            if not target_match:
                return 0.0
                
            target = int(target_match.group(1))
            
            # Extract final number from completion
            numbers = re.findall(r'\d+', completion)
            if not numbers:
                return 0.0
                
            final_value = int(numbers[-1])
            distance = abs(final_value - target)
            
            if distance == 0:
                return 1.0
            elif distance <= 10:
                return 0.5
            else:
                return max(0.0, 1.0 - distance / 100)
                
        except:
            return 0.0
    
    def evaluate_multiplication_completion(self, prompt: str, completion: str) -> float:
        """Evaluate multiplication completion reward."""
        try:
            import re
            # Extract operands from prompt
            numbers = re.findall(r'(\d+) × (\d+)', prompt)
            if not numbers:
                return 0.0
                
            a, b = map(int, numbers[0])
            correct_answer = a * b
            
            # Extract answer from completion
            answer_numbers = re.findall(r'\d+', completion)
            if not answer_numbers:
                return 0.0
                
            provided_answer = int(answer_numbers[-1])
            
            if provided_answer == correct_answer:
                return 1.0
            else:
                return 0.0
                
        except:
            return 0.0
    
    def train_iteration(self):
        """Perform one training iteration."""
        self.iteration += 1
        
        # Collect trajectories
        trajectories = self.collect_trajectories()
        
        # Perform A*PO update
        update_stats = self.astarpo.update_policy(
            self.policy_model,
            self.reference_model,
            self.optimizer,
            trajectories,
            self.compute_reward
        )
        
        # Update learning rate
        self.scheduler.step()
        
        # Compute additional statistics
        trajectory_stats = self.compute_trajectory_statistics(trajectories)
        
        # Combine statistics
        combined_stats = {
            'iteration': self.iteration,
            'learning_rate': self.scheduler.get_last_lr()[0],
            **update_stats,
            **trajectory_stats
        }
        
        # All-reduce statistics across processes
        combined_stats = all_reduce_dict(combined_stats)
        
        # Log statistics (only on main process)
        if is_main_process():
            self.log_training_stats(combined_stats)
            
            # Save checkpoint if needed
            if self.iteration % self.config['experiment']['save_interval'] == 0:
                self.save_checkpoint()
                
            # Evaluate if needed
            if self.iteration % self.config['experiment']['eval_interval'] == 0:
                eval_stats = self.evaluate_model()
                combined_stats.update(eval_stats)
        
        return combined_stats
    
    def compute_trajectory_statistics(self, trajectories: list) -> dict:
        """Compute statistics from collected trajectories."""
        if not trajectories:
            return {}
            
        total_rewards = [t.total_reward for t in trajectories]
        trajectory_lengths = [len(t.steps) for t in trajectories]
        completion_rates = [1.0 if t.terminal else 0.0 for t in trajectories]
        
        return {
            'avg_trajectory_reward': sum(total_rewards) / len(total_rewards),
            'max_trajectory_reward': max(total_rewards),
            'avg_trajectory_length': sum(trajectory_lengths) / len(trajectory_lengths),
            'completion_rate': sum(completion_rates) / len(completion_rates),
        }
    
    def log_training_stats(self, stats: dict):
        """Log training statistics."""
        if self.logger:
            self.logger.log_metrics(stats, self.iteration)
            
        if self.wandb_logger:
            self.wandb_logger.log_metrics(stats, self.iteration)
    
    def evaluate_model(self) -> dict:
        """Evaluate current model performance."""
        # Simple evaluation - extend with proper evaluation from evaluate.py
        eval_stats = {
            'eval_countdown_score': 0.0,
            'eval_multiplication_score': 0.0,
        }
        
        if self.logger:
            self.logger.logger.info(f"Evaluation at iteration {self.iteration}: {eval_stats}")
            
        return eval_stats
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        if not is_main_process():
            return
            
        checkpoint_dir = Path(self.config['experiment']['checkpoint_dir'])
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"
        
        checkpoint = {
            'iteration': self.iteration,
            'policy_model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_reward': self.best_reward,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if self.logger:
            self.logger.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        total_iterations = self.config['training']['total_iterations']
        
        if is_main_process():
            self.logger.logger.info(f"Starting training for {total_iterations} iterations")
        
        try:
            for iteration in range(total_iterations):
                stats = self.train_iteration()
                
                # Early stopping based on performance
                if stats.get('avg_trajectory_reward', 0) > self.best_reward:
                    self.best_reward = stats['avg_trajectory_reward']
                    
        except KeyboardInterrupt:
            if is_main_process():
                self.logger.logger.info("Training interrupted by user")
        except Exception as e:
            if is_main_process():
                self.logger.logger.error(f"Training failed: {e}")
            raise
        finally:
            # Save final checkpoint
            if is_main_process():
                self.save_checkpoint()
                
            # Cleanup
            cleanup_distributed()
            
            if self.wandb_logger:
                self.wandb_logger.finish()
                
            if is_main_process():
                self.logger.logger.info("Training completed")

def main():
    """Main entry point."""
    trainer = TinyZeroTrainer()
    trainer.train()

if __name__ == "__main__":
    main()