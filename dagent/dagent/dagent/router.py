import math
from typing import Dict, List


class UCBRouter:
    def __init__(self, exploration_c: float = 1.41):
        self.counts: Dict[str, int] = {}
        self.rewards: Dict[str, float] = {}
        self.exploration_c = exploration_c
    
    def select(self, strategies: List[str]) -> str:
        total = sum(self.counts.get(s, 1) for s in strategies)
        best, best_score = None, -float('inf')
        
        for s in strategies:
            c = self.counts.get(s, 1)
            r = self.rewards.get(s, 0)
            avg = r / c
            explore = math.sqrt(math.log(total + 1) / c)
            score = avg + self.exploration_c * explore
            
            if score > best_score:
                best, best_score = s, score
        
        return best
    
    def update(self, strategy: str, reward: float):
        self.counts[strategy] = self.counts.get(strategy, 1) + 1
        self.rewards[strategy] = self.rewards.get(strategy, 0) + reward
