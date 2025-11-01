import gym
from typing import Dict, Any, Tuple
from .base import BaseEnvironment


class FrozenLakeEnvironment(BaseEnvironment):
    """Simple wrapper for OpenAI Gym FrozenLake to integrate with TinyZero.

    The wrapper exposes textual prompts that the LM can respond to. The LM's
    generated action is expected to contain a tokenized integer (0..3). This is
    a lightweight bridge for demo and reproduction experiments.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        env_name = config.get('env_name', 'FrozenLake-v1')
        # Use non-slippery variant by default for reproducibility
        self.env = gym.make(
            env_name, is_slippery=config.get('is_slippery', False))

    def reset(self) -> str:
        obs = self.env.reset()
        # Gym returns tuple (obs, info) in newer gym versions; handle both
        try:
            if isinstance(obs, tuple):
                obs, _ = obs
        except Exception:
            pass

        self._step_count = 0
        self._total_reward = 0.0
        return self._build_prompt(obs)

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        # Try to parse an integer action from the model output
        act = 0
        try:
            # Extract last number in action string
            import re
            numbers = re.findall(r"\d+", action)
            if numbers:
                act = int(numbers[-1])
            else:
                act = int(action.strip())
        except Exception:
            # fallback to noop (0)
            act = 0

        res = self.env.step(act)
        # Handle gym API differences
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
        else:
            obs, reward, done, info = res

        self._update_stats(reward)

        return self._build_prompt(obs), float(reward), bool(done), info

    def is_terminal(self) -> bool:
        # Gym tracks terminal state internally; reuse base method
        return False

    def get_valid_actions(self):
        # Discrete actions 0..3 for FrozenLake
        return ["0", "1", "2", "3"]

    def _build_prompt(self, obs) -> str:
        prompt = f"FrozenLake observation: {obs}\n"
        prompt += "Choose an action (0=Left,1=Down,2=Right,3=Up). Reply with the action index."
        return prompt
