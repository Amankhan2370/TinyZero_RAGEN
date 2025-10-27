import random
import re
import operator
from typing import Dict, Any, Tuple, List
from .base import BaseEnvironment

class CountdownEnvironment(BaseEnvironment):
    """
    Countdown numbers game environment.
    Goal: Use arithmetic operations to reach a target number.
    """
    
    OPERATIONS = {
        '+': operator.add,
        '-': operator.sub, 
        '*': operator.mul,
        '/': operator.truediv
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_range = config.get('target_range', [100, 1000])
        self.number_range = config.get('number_range', [1, 100])
        self.max_numbers = config.get('max_numbers', 6)
        self.max_steps = config.get('max_steps', 10)
        self.allow_negatives = config.get('allow_negatives', False)
        self.allow_fractions = config.get('allow_fractions', False)
        
        self.target = None
        self.numbers = None
        self.used_indices = None
        self.current_value = None
        self.expression = None
        
    def reset(self) -> str:
        """Reset environment with new target and numbers."""
        self.target = random.randint(*self.target_range)
        self.numbers = [random.randint(*self.number_range) for _ in range(self.max_numbers)]
        self.used_indices = set()
        self.current_value = None
        self.expression = None
        self._step_count = 0
        self._total_reward = 0.0
        
        return self._build_prompt()
        
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute arithmetic operation action."""
        self._step_count += 1
        
        try:
            # Parse and validate action
            num1, num2, op = self._parse_action(action)
            if not self._validate_operation(num1, num2, op):
                return self._build_prompt(), -0.5, False, {"error": "Invalid operation"}
                
            # Perform calculation
            result = self._calculate(num1, num2, op)
            if result is None:
                return self._build_prompt(), -0.5, False, {"error": "Calculation failed"}
                
            # Update state
            self.current_value = result
            if self.expression:
                self.expression = f"({self.expression}) {op} {num2}"
            else:
                self.expression = f"{num1} {op} {num2}"
                
            # Compute reward and check termination
            reward = self._compute_reward()
            done = self._check_termination()
            
            info = {
                "result": result,
                "expression": self.expression,
                "target": self.target,
                "distance": abs(result - self.target),
                "numbers_remaining": self._get_available_numbers(),
            }
            
            self._total_reward += reward
            return self._build_prompt(), reward, done, info
            
        except Exception as e:
            return self._build_prompt(), -0.5, False, {"error": str(e)}
            
    def _parse_action(self, action: str) -> Tuple[int, int, str]:
        """Parse action string into numbers and operation."""
        # Match patterns like "25 + 37" or "numbers[0] + numbers[2]"
        pattern = r'(\d+|numbers\[\d+\])\s*([+\-*/])\s*(\d+|numbers\[\d+\])'
        match = re.match(pattern, action.strip())
        if not match:
            raise ValueError("Invalid action format")
            
        num1_str, op, num2_str = match.groups()
        
        # Parse numbers (could be direct values or array indices)
        num1 = self._parse_number(num1_str)
        num2 = self._parse_number(num2_str)
        
        return num1, num2, op
        
    def _parse_number(self, num_str: str) -> int:
        """Parse number from string, handling array indices."""
        if num_str.startswith('numbers['):
            idx = int(re.search(r'\[(\d+)\]', num_str).group(1))
            if idx < 0 or idx >= len(self.numbers):
                raise ValueError(f"Invalid number index: {idx}")
            if idx in self.used_indices:
                raise ValueError(f"Number at index {idx} already used")
            self.used_indices.add(idx)
            return self.numbers[idx]
        else:
            num = int(num_str)
            if num not in self.numbers:
                raise ValueError(f"Number {num} not available")
            idx = self.numbers.index(num)
            if idx in self.used_indices:
                raise ValueError(f"Number {num} already used")
            self.used_indices.add(idx)
            return num
            
    def _validate_operation(self, num1: int, num2: int, op: str) -> bool:
        """Validate if operation is allowed."""
        if op == '-' and num1 <= num2 and not self.allow_negatives:
            return False
        if op == '/' and (num2 == 0 or num1 % num2 != 0) and not self.allow_fractions:
            return False
        return True
        
    def _calculate(self, num1: int, num2: int, op: str) -> float:
        """Perform arithmetic calculation."""
        try:
            result = self.OPERATIONS[op](num1, num2)
            # Ensure integer result unless fractions allowed
            if not self.allow_fractions and not result.is_integer():
                return None
            return result
        except (ZeroDivisionError, ValueError):
            return None
            
    def _compute_reward(self) -> float:
        """Compute reward based on distance to target."""
        if self.current_value is None:
            return -0.1
            
        distance = abs(self.current_value - self.target)
        
        if distance == 0:
            return 1.0  # Perfect solution
        elif distance <= 5:
            return 0.8  # Very close
        elif distance <= 20:
            return 0.3  # Close
        elif distance <= 50:
            return 0.1  # Somewhat close
        else:
            return -0.01 * distance  # Penalize large distances
            
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        return (self.current_value == self.target or
                self._step_count >= self.max_steps or
                len(self.used_indices) == self.max_numbers or
                len(self._get_available_numbers()) < 2)
                
    def _get_available_numbers(self) -> List[int]:
        """Get list of available numbers."""
        return [n for i, n in enumerate(self.numbers) if i not in self.used_indices]
        
    def _build_prompt(self) -> str:
        """Build state prompt for language model."""
        prompt = f"Countdown Game - Target: {self.target}\n"
        prompt += f"Available numbers: {self._get_available_numbers()}\n"
        
        if self.current_value is not None:
            prompt += f"Current value: {self.current_value} (expression: {self.expression})\n"
            
        prompt += "Enter your next operation (e.g., '25 + 37' or 'numbers[0] * numbers[2]'):"
        return prompt
        
    def is_terminal(self) -> bool:
        return self._check_termination()
        
    def get_valid_actions(self) -> List[str]:
        """Generate valid actions based on available numbers."""
        available = self._get_available_numbers()
        if len(available) < 2:
            return []
            
        actions = []
        for i, num1 in enumerate(available):
            for j, num2 in enumerate(available):
                if i == j:
                    continue
                for op in ['+', '-', '*', '/']:
                    if self._validate_operation(num1, num2, op):
                        actions.append(f"{num1} {op} {num2}")
        return actions
        
    def render(self) -> str:
        """Render current state for display."""
        return self._build_prompt()