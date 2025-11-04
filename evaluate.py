#!/usr/bin/env python3
"""
TinyZero with A*PO - Evaluation Script
Comprehensive evaluation of trained models.
"""

import torch
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

from models.transformer import TransformerLM
from environments.countdown import CountdownEnvironment
from environments.multiplication import MultiplicationEnvironment

class TinyZeroEvaluator:
    """Comprehensive evaluator for TinyZero models."""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.load_config()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.setup_environments()
        
    def load_config(self):
        """Load configuration."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def load_model(self):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        
        # Recreate model architecture
        model_config = self.config['model']
        self.model = TransformerLM(
            vocab_size=model_config['vocab_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            mlp_ratio=model_config['mlp_ratio'],
            dropout=model_config['dropout']
        )
        
        # Load trained weights
        if 'policy_model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['policy_model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {self.checkpoint_path}")
        
    def setup_environments(self):
        """Initialize evaluation environments."""
        env_config = self.config['environments']
        
        self.countdown_env = CountdownEnvironment(env_config['countdown'])
        self.multiplication_env = MultiplicationEnvironment(env_config['multiplication'])
        
    def evaluate_countdown(self, num_episodes: int = 100) -> Dict[str, float]:
        """Comprehensive evaluation on countdown task."""
        successes = 0
        total_reward = 0.0
        steps_per_episode = []
        final_distances = []
        
        for episode in range(num_episodes):
            state = self.countdown_env.reset()
            done = False
            steps = 0
            
            while not done and steps < 20:  # Increased step limit for evaluation
                action = self.generate_action(state)
                next_state, reward, done, info = self.countdown_env.step(action)
                
                state = next_state
                steps += 1
                total_reward += reward
                
                if done:
                    break
            
            # Check if successful
            if info.get('distance', float('inf')) == 0:
                successes += 1
                
            steps_per_episode.append(steps)
            final_distances.append(info.get('distance', float('inf')))
            
        success_rate = successes / num_episodes
        avg_reward = total_reward / num_episodes
        avg_steps = sum(steps_per_episode) / len(steps_per_episode)
        avg_distance = sum(final_distances) / len(final_distances)
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'avg_final_distance': avg_distance,
            'num_episodes': num_episodes
        }
        
    def evaluate_multiplication(self, num_problems: int = 100) -> Dict[str, float]:
        """Comprehensive evaluation on multiplication task."""
        correct = 0
        reasoning_quality_scores = []
        
        for problem in range(num_problems):
            state = self.multiplication_env.reset()
            action = self.generate_action(state)
            _, reward, _, info = self.multiplication_env.step(action)
            
            if info.get('is_correct', False):
                correct += 1
                
            # Score reasoning quality
            reasoning_quality = self.score_reasoning_quality(action)
            reasoning_quality_scores.append(reasoning_quality)
            
        accuracy = correct / num_problems
        avg_reasoning_quality = sum(reasoning_quality_scores) / len(reasoning_quality_scores)
        
        return {
            'accuracy': accuracy,
            'avg_reasoning_quality': avg_reasoning_quality,
            'correct': correct,
            'total': num_problems
        }
        
    def score_reasoning_quality(self, action: str) -> float:
        """Score the quality of reasoning in the action."""
        import re
        
        # Check for step-by-step reasoning
        has_steps = len(re.findall(r'\d+\.|\n', action)) > 1
        has_calculation = any(op in action for op in ['+', '-', '*', '/', '×', '÷'])
        has_explanation = any(word in action.lower() for word in 
                            ['because', 'therefore', 'thus', 'since', 'so'])
        
        score = 0.0
        if has_steps:
            score += 0.4
        if has_calculation:
            score += 0.4
        if has_explanation:
            score += 0.2
            
        return min(score, 1.0)
        
    def generate_action(self, state: str) -> str:
        """Generate action using trained model."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            
            # Tokenize state
            inputs = tokenizer(state, return_tensors="pt").to(self.device)
            prompt_length = inputs['input_ids'].shape[1]
            
            # Generate completion
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=prompt_length + 100,  # Longer for evaluation
                    do_sample=False,  # Deterministic for evaluation
                    temperature=0.7,
                    return_dict_in_generate=True
                )
            
            # Extract generated tokens
            generated_tokens = outputs.sequences[0, prompt_length:]
            action = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return action
            
        except Exception as e:
            print(f"Action generation failed: {e}")
            return "1 + 1"  # Fallback action
            
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on both tasks."""
        print("Starting comprehensive evaluation...")
        
        # Evaluate countdown
        print("\nEvaluating Countdown Task...")
        countdown_results = self.evaluate_countdown(
            self.config['evaluation']['countdown_tests']
        )
        
        # Evaluate multiplication  
        print("Evaluating Multiplication Task...")
        multiplication_results = self.evaluate_multiplication(
            self.config['evaluation']['multiplication_tests']
        )
        
        # Combine results
        results = {
            'countdown': countdown_results,
            'multiplication': multiplication_results,
            'overall_score': (
                countdown_results['success_rate'] + 
                multiplication_results['accuracy']
            ) / 2
        }
        
        return results
        
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format."""
        print("\n" + "="*60)
        print("TINYZERO A*PO - EVALUATION RESULTS")
        print("="*60)
        
        # Countdown results
        countdown = results['countdown']
        print(f"\nCOUNTDOWN TASK (n={countdown['num_episodes']}):")
        print(f"  Success Rate: {countdown['success_rate']:.1%}")
        print(f"  Average Reward: {countdown['avg_reward']:.3f}")
        print(f"  Average Steps: {countdown['avg_steps']:.1f}")
        print(f"  Average Final Distance: {countdown['avg_final_distance']:.1f}")
        
        # Multiplication results
        multiplication = results['multiplication']
        print(f"\nMULTIPLICATION TASK (n={multiplication['total']}):")
        print(f"  Accuracy: {multiplication['accuracy']:.1%}")
        print(f"  Reasoning Quality: {multiplication['avg_reasoning_quality']:.3f}")
        print(f"  Correct: {multiplication['correct']}/{multiplication['total']}")
        
        # Overall score
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Combined Score: {results['overall_score']:.3f}")
        print("="*60)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate TinyZero model")
    parser.add_argument("--config", default="configs/default.yaml", 
                       help="Path to config file")
    parser.add_argument("--checkpoint", required=True,
                       help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = TinyZeroEvaluator(args.config, args.checkpoint)
    results = evaluator.run_comprehensive_evaluation()
    evaluator.print_results(results)
    
    # Save results to file
    output_file = Path("evaluation_results.json")
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()