import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any

from .astarpo import AStarPO, AStarPOConfig
from algorithms.trajectory import Trajectory


class RAGENConfig(AStarPOConfig):
    """Configuration for RAGEN integrated with A*PO."""
    num_candidates: int = 16
    kl_coef: float = 0.1


class RAGEN:
    """
    Simple RAGEN implementation that uses the repository's A*PO utilities
    to compute V* and advantages, and applies the A*PO regression-style
    policy update on generated candidate completions.
    This class exposes the same update_policy interface used by the trainer.
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        # Accept either dict or AStarPOConfig-like mapping
        if isinstance(config, AStarPOConfig):
            self.config = config
        else:
            # Build AStarPOConfig then extend
            self.config = AStarPOConfig(
                **config) if isinstance(config, dict) else AStarPOConfig()
        self.device = device
        # Use internal AStarPO helper for V* computation and loss utilities
        self.astarpo = AStarPO(self.config, device)

    def update_policy(self, policy_model, reference_model, optimizer, trajectories: List[Trajectory], reward_fn: callable) -> Dict[str, float]:
        """
        RAGEN update step: for each trajectory prompt sample K candidate completions
        from the current policy, compute rewards, compute V* using reference model
        (via A*PO's compute_optimal_value), compute advantages and apply the
        regression loss similar to A*PO.
        """
        policy_model.train()
        reference_model.eval()

        if not trajectories:
            return {'policy_loss': 0.0, 'avg_advantage': 0.0, 'avg_reward': 0.0}

        prompts = [t.initial_state for t in trajectories]

        # Compute V* for prompts using the reference model
        with torch.no_grad():
            v_star = self.astarpo.compute_optimal_value(
                prompts, reference_model, reward_fn)

        total_loss = 0.0
        total_adv = 0.0
        total_reward = 0.0
        n = 0

        optimizer.zero_grad()

        for traj, v_star_prompt in zip(trajectories, v_star):
            prompt = traj.initial_state
            
            # Sample candidates from policy (WITH gradients)
            completions_policy, policy_log_probs = self._sample_with_gradients(
                policy_model, prompt, self.config.num_completions)
            
            # Sample reference log-probs (WITHOUT gradients)
            with torch.no_grad():
                _, ref_log_probs = self.astarpo._sample_completions_batch(
                    reference_model, prompt, self.config.num_completions)

            # Compute rewards for each completion
            rewards = torch.tensor([reward_fn(
                prompt, c) for c in completions_policy], device=self.device, dtype=torch.float32)

            # Compute advantages as in A*PO
            with torch.no_grad():
                advantages = self.astarpo.compute_advantages(
                    rewards, v_star_prompt)

            # Policy log probs should have gradients, ref shouldn't
            log_ratio = policy_log_probs - ref_log_probs.detach()
            scaled_log_ratio = self.config.beta * log_ratio

            # Ensure shapes align
            if scaled_log_ratio.shape != advantages.shape:
                advantages = advantages.view_as(scaled_log_ratio)

            # Detach advantages to avoid backprop through them
            loss = F.mse_loss(scaled_log_ratio, advantages.detach())

            # small entropy bonus (encourage exploration)
            entropy = -(policy_log_probs.exp() * policy_log_probs).mean()
            loss = loss - 0.01 * entropy

            loss.backward()

            total_loss += loss.item()
            total_adv += advantages.mean().item()
            total_reward += rewards.mean().item()
            n += 1

        if n > 0:
            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), self.config.max_grad_norm)
            optimizer.step()

        stats = {
            'policy_loss': total_loss / max(n, 1),
            'avg_advantage': total_adv / max(n, 1),
            'avg_reward': total_reward / max(n, 1),
            'num_trajectories': len(trajectories),
            'num_updates': n
        }

        return stats
    
    def _sample_with_gradients(self, model, prompt, num_samples):
        """Sample completions from model while maintaining gradients."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs['input_ids'].shape[1]
        
        completions = []
        log_probs_list = []
        
        for _ in range(num_samples):
            # Generate with the model (keeping gradients)
            outputs = model.generate(
                **inputs,
                max_length=prompt_length + 50,
                do_sample=True,
                temperature=0.8,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Extract completion
            completion_tokens = outputs.sequences[0, prompt_length:]
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            completions.append(completion)
            
            # Compute log probability WITH gradients for policy model
            # We need to recompute forward pass to maintain gradients
            completion_ids = completion_tokens.unsqueeze(0)
            full_ids = torch.cat([inputs['input_ids'], completion_ids], dim=1)
            
            # Forward pass through model to get logits
            model_outputs = model(full_ids)
            logits = model_outputs['logits']
            
            # Compute log probs for the generated tokens
            log_probs = F.log_softmax(logits[:, prompt_length-1:-1, :], dim=-1)
            
            # Gather log probs of actual tokens
            token_log_probs = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
            total_log_prob = token_log_probs.sum()
            
            log_probs_list.append(total_log_prob)
        
        # Stack log probs (these should have gradients)
        log_probs_tensor = torch.stack(log_probs_list)
        
        return completions, log_probs_tensor