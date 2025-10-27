import random
from typing import Dict, Any, Tuple
from .base import BaseEnvironment

class MultiplicationEnvironment(BaseEnvironment):
    """Multiplication task environment"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.operand_range = config['operand_range']
        self.max_steps = config['max_steps']
        self.reset()
    
    def reset(self) -> str:
        """Reset with new multiplication problem"""
        self.a = random.randint(*self.operand_range)
        self.b = random.randint(*self.operand_range)
        self.correct_answer = self.a * self.b
        self.steps_taken = 0
        self.final_answer = None
        
        return self._get_state_prompt()
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Process answer"""
        self.steps_taken += 1
        
        try:
            # Extract numeric answer
            answer = self._extract_answer(action)
            self.final_answer = answer
            
            # Compute reward
            reward = 1.0 if answer == self.correct_answer else 0.0
            done = True  # Single-step task
            
            info = {
                "correct_answer": self.correct_answer,
                "provided_answer": answer,
                "is_correct": reward == 1.0
            }
            
            return self._get_state_prompt(), reward, done, info
            
        except Exception as e:
            return self._get_state_prompt(), 0.0, True, {"error": str(e)}
    
    def _extract_answer(self, action: str) -> int:
        """Extract numeric answer from model response"""
        import re
        
        # Look for numbers in the response
        numbers = re.findall(r'\d+', action)
        if numbers:
            return int(numbers[-1])  # Take the last number as answer
        
        # Try to evaluate simple expressions
        try:
            # Remove any non-math characters and evaluate
            clean_action = re.sub(r'[^\d+\-*/().]', '', action)
            if clean_action:
                return eval(clean_action)
        except:
            pass
        
        raise ValueError("Could not extract valid answer")
    
    def _get_state_prompt(self) -> str:
        """Generate multiplication problem prompt"""
        prompt = f"Multiplication Problem:\nWhat is {self.a} × {self.b}?\n"
        prompt += "Show your reasoning step by step, then provide the final answer."
        return prompt
    
    def is_terminal(self, state: str) -> bool:
        return self.steps_taken >= 1
    
    def get_reward(self, state: str, action: str) -> float:
        answer = self._extract_answer(action)
        return 1.0 if answer == self.correct_answer else 0.0