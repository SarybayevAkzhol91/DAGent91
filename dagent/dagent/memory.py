import time
from typing import Any, Dict, List


class Memory:
    def __init__(self, max_size: int = 1000):
        self.data = []
        self.max_size = max_size
    
    def add(self, task: str, step: Any, result: Any, strategy: str):
        self.data.append({
            "task": task,
            "tool": step.tool,
            "success": result.success,
            "confidence": result.confidence,
            "strategy": strategy,
            "ts": time.time()
        })
        
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]
    
    def retrieve(self, task: str, limit: int = 10) -> List[Dict]:
        relevant = [x for x in self.data if x["task"] == task]
        if relevant:
            return relevant[-limit:]
        return self.data[-limit:]
