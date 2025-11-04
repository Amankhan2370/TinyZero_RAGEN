import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .trajectory import Trajectory, TrajectoryBatch

@dataclass
class AStarPOConfig:
    """Configuration for A*PO algorithm."""
    beta: float = 0.1  # KL regularization coefficient
    num_trajectories: int = 100  # Number of trajectories per iteration
    num_completions: int = 20  # Completions per prompt for V* estimation
    advantage_clip: float = 1.0  # Clip advantages to this value
    gamma: float = 1.0  # Discount factor
    value_estimation_samples: int = 100  # Samples for Monte Carlo V* estimation
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

class AStarPO:
    """
    A*PO (State-Thinking-Actions-Reward Policy Optimization)
    Advanced implementation with proper numerical stability and optimization.
    """
    
    def __init__(self, config: AStarPOConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
    def compute_optimal_value(self, prompts: List[str], reference_model: nn.Module, 
                            reward_fn: callable) -> torch.Tensor:
        """
        Compute V*(x) = β log E_{y∼π_ref(·|x)}[exp(r(x,y)/β)]
        Using Monte Carlo estimation with importance sampling.
        """
        v_star_values = []
        
        for prompt in prompts:
            with torch.no_grad():
                # Sample multiple completions from reference model
                completions, log_probs = self._sample_completions_batch(
                    reference_model, prompt, self.config.num_completions
                )
                
                # Compute rewards for each completion
                rewards = torch.tensor([
                    reward_fn(prompt, completion) for completion in completions
                ], device=self.device, dtype=torch.float32)
                
                # Compute V* using numerically stable log-sum-exp
                v_star = self._compute_v_star_stable(rewards, log_probs)
                v_star_values.append(v_star)
                
        return torch.stack(v_star_values)
    
    def _compute_v_star_stable(self, rewards: torch.Tensor, 
                             log_probs: torch.Tensor) -> torch.Tensor:
        """Compute V* with numerical stability."""
        # Normalize rewards for numerical stability
        max_reward = rewards.max()
        scaled_rewards = (rewards - max_reward) / self.config.beta
        
        # Compute log expectation using log-sum-exp
        log_expectation = torch.logsumexp(scaled_rewards + log_probs, dim=0)
        log_expectation -= torch.logsumexp(log_probs, dim=0)  # Importance weighting
        
        v_star = self.config.beta * log_expectation + max_reward
        return v_star
    
    def compute_advantages(self, rewards: torch.Tensor, 
                          v_star: torch.Tensor) -> torch.Tensor:
        """Compute advantages A*(x,y) = r(x,y) - V*(x) with clipping."""
        advantages = rewards - v_star.unsqueeze(-1)
        
        if self.config.advantage_clip > 0:
            advantages = torch.clamp(
                advantages, 
                -self.config.advantage_clip, 
                self.config.advantage_clip
            )
            
        return advantages
    
    def compute_policy_loss(self, policy_log_probs: torch.Tensor,
                          reference_log_probs: torch.Tensor,
                          advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute A*PO policy loss:
        L(θ) = E[(β log(π_θ/π_ref) - A*)^2] + λ * entropy_regularization
        """
        log_ratio = policy_log_probs - reference_log_probs
        scaled_log_ratio = self.config.beta * log_ratio
        
        # Main regression loss
        regression_loss = F.mse_loss(scaled_log_ratio, advantages)
        
        # Add entropy regularization for exploration
        entropy = -torch.exp(policy_log_probs) * policy_log_probs
        entropy_loss = -entropy.mean() * 0.01  # Small entropy bonus
        
        total_loss = regression_loss + entropy_loss
        return total_loss
    
    def update_policy(self, policy_model: nn.Module, reference_model: nn.Module,
                     optimizer: torch.optim.Optimizer, trajectories: List[Trajectory],
                     reward_fn: callable) -> Dict[str, float]:
        """
        Perform A*PO policy update step.
        
        Args:
            policy_model: Current policy being optimized
            reference_model: Frozen reference policy
            optimizer: Policy optimizer
            trajectories: Collected trajectories
            reward_fn: Reward function for evaluating completions
            
        Returns:
            Dictionary of training statistics
        """
        policy_model.train()
        reference_model.eval()
        
        if not trajectories:
            return {'policy_loss': 0.0, 'avg_advantage': 0.0, 'avg_reward': 0.0}
        
        # Extract prompts from trajectories
        prompts = [t.initial_state for t in trajectories]
        
        # Compute V* for all prompts
        with torch.no_grad():
            v_star = self.compute_optimal_value(prompts, reference_model, reward_fn)
        
        total_loss = 0.0
        total_advantage = 0.0
        total_reward = 0.0
        num_updates = 0
        
        optimizer.zero_grad()
        
        for i, (trajectory, v_star_prompt) in enumerate(zip(trajectories, v_star)):
            if not trajectory.steps:
                continue
                
            # Convert trajectory to tensors
            traj_tensors = trajectory.to_tensor_dict(self.device)
            rewards = traj_tensors['rewards']
            policy_log_probs = traj_tensors['log_probs']
            
            # Sample reference completions for this prompt
            with torch.no_grad():
                _, reference_log_probs = self._sample_completions_batch(
                    reference_model, trajectory.initial_state, 1
                )
                reference_log_prob = reference_log_probs.mean()
            
            # Compute advantages
            advantages = self.compute_advantages(rewards, v_star_prompt)
            
            # Compute policy loss
            loss = self.compute_policy_loss(
                policy_log_probs.mean(), 
                reference_log_prob, 
                advantages.mean()
            )
            
            # Accumulate gradients
            loss.backward()
            
            # Accumulate statistics
            total_loss += loss.item()
            total_advantage += advantages.mean().item()
            total_reward += rewards.mean().item()
            num_updates += 1
        
        # Clip gradients and update policy
        if num_updates > 0:
            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), 
                self.config.max_grad_norm
            )
            optimizer.step()
        
        # Compute average statistics
        stats = {
            'policy_loss': total_loss / max(num_updates, 1),
            'avg_advantage': total_advantage / max(num_updates, 1),
            'avg_reward': total_reward / max(num_updates, 1),
            'num_trajectories': len(trajectories),
            'num_updates': num_updates
        }
        
        return stats
    
    def _sample_completions_batch(self, model: nn.Module, prompt: str, 
                                num_samples: int) -> Tuple[List[str], torch.Tensor]:
        """
        Sample multiple completions from model efficiently.
        
        Returns:
            Tuple of (completions, log_probs)
        """
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_length = inputs['input_ids'].shape[1]
            
            completions = []
            all_log_probs = []
            
            # Batch sampling for efficiency
            batch_size = min(num_samples, 8)  # Adjust based on memory
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
                
                # Expand inputs for batch processing
                batch_inputs = {
                    k: v.repeat(current_batch_size, 1) for k, v in inputs.items()
                }
                
                with torch.no_grad():
                    outputs = model.generate(
                        **batch_inputs,
                        max_length=prompt_length + 50,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                        num_return_sequences=current_batch_size
                    )
                
                # Process each sequence in batch
                for seq_idx in range(current_batch_size):
                    # Extract completion tokens
                    completion_tokens = outputs.sequences[seq_idx, prompt_length:]
                    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                    completions.append(completion)
                    
                    # Compute log probability
                    seq_log_probs = []
                    for step_idx, step_scores in enumerate(outputs.scores):
                        if step_idx >= len(completion_tokens):
                            break
                        step_log_probs = F.log_softmax(step_scores[seq_idx], dim=-1)
                        token_log_prob = step_log_probs[completion_tokens[step_idx]]
                        seq_log_probs.append(token_log_prob)
                    
                    if seq_log_probs:
                        total_log_prob = torch.stack(seq_log_probs).sum()
                        all_log_probs.append(total_log_prob)
                    else:
                        all_log_probs.append(torch.tensor(0.0, device=self.device))
            
            log_probs_tensor = torch.stack(all_log_probs)
            return completions, log_probs_tensor
            
        except Exception as e:
            # Fallback to sequential sampling
            import warnings
            warnings.warn(f"Batch sampling failed: {e}. Falling back to sequential.")
            return self._sample_completions_sequential(model, prompt, num_samples)
    
    def _sample_completions_sequential(self, model: nn.Module, prompt: str,
                                     num_samples: int) -> Tuple[List[str], torch.Tensor]:
        """Sequential sampling fallback."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs['input_ids'].shape[1]
        
        completions = []
        log_probs = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=prompt_length + 50,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            completion_tokens = outputs.sequences[0, prompt_length:]
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            completions.append(completion)
            
            # Compute log probability
            seq_log_probs = []
            for i, step_scores in enumerate(outputs.scores):
                if i >= len(completion_tokens):
                    break
                step_log_probs = F.log_softmax(step_scores[0], dim=-1)
                token_log_prob = step_log_probs[completion_tokens[i]]
                seq_log_probs.append(token_log_prob)
            
            total_log_prob = torch.stack(seq_log_probs).sum() if seq_log_probs else torch.tensor(0.0)
            log_probs.append(total_log_prob)
        
        return completions, torch.stack(log_probs)