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
            # Sample candidates from policy
            completions, policy_log_probs = self.astarpo._sample_completions_batch(
                policy_model, prompt, self.config.num_completions)

            # Sample reference log-probs for same prompt (importance baseline)
            with torch.no_grad():
                _, ref_log_probs = self.astarpo._sample_completions_batch(
                    reference_model, prompt, self.config.num_completions)

            # Compute rewards for each completion
            rewards = torch.tensor([reward_fn(
                prompt, c) for c in completions], device=self.device, dtype=torch.float32)

            # Compute advantages as in A*PO
            advantages = self.astarpo.compute_advantages(
                rewards, v_star_prompt)

            # Policy and reference log probs are tensors shaped (num_completions,)
            # Convert to scalars per-sample and compute loss
            # Use vectorised MSE between beta * log_ratio and advantages
            log_ratio = policy_log_probs - ref_log_probs
            scaled_log_ratio = self.config.beta * log_ratio

            # Ensure shapes align
            if scaled_log_ratio.shape != advantages.shape:
                advantages = advantages.view_as(scaled_log_ratio)

            loss = F.mse_loss(scaled_log_ratio, advantages)

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
