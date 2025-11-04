from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import numpy as np

class BaseEnvironment(ABC):
    """
    Abstract base class for reinforcement learning environments.
    Defines the interface that all environments must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._step_count = 0
        self._total_reward = 0.0
        self.reset()
        
    @abstractmethod
    def reset(self) -> str:
        """
        Reset the environment to initial state.
        
        Returns:
            str: Initial state description/prompt
        """
        pass
        
    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action string to execute
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass
        
    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        pass
        
    @abstractmethod
    def get_valid_actions(self) -> List[str]:
        """Get list of valid actions in current state."""
        pass
        
    def validate_action(self, action: str) -> bool:
        """Validate if action is legal in current state."""
        valid_actions = self.get_valid_actions()
        return len(valid_actions) == 0 or action in valid_actions
        
    def get_state_representation(self) -> str:
        """Get string representation of current state."""
        return str(self)
        
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for current episode."""
        return {
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "is_terminal": self.is_terminal(),
        }
        
    def _update_stats(self, reward: float):
        """Update internal statistics."""
        self._step_count += 1
        self._total_reward += reward
        
    def render(self) -> str:
        """Render current state as string for display."""
        return self.get_state_representation()
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(steps={self._step_count}, reward={self._total_reward:.2f})"