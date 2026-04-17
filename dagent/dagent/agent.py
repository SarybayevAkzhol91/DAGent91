"""
DAGent - Main Agent Class
"""

import time
from typing import Any, Dict, List

from dagent.memory import Memory
from dagent.router import UCBRouter
from dagent.planner import Planner
from dagent.executor import Executor, DAGEngine, Step, Result


class DAGent:
    def __init__(self, llm, tools: Dict[str, Any]):
        self.memory = Memory()
        self.router = UCBRouter()
        self.planner = Planner(llm)
        self.executor = Executor(tools)
        self.engine = DAGEngine(self.executor)
        self.strategies = ["fast", "safe", "deep"]
        self.max_replans = 2
    
    def run(self, task: str, verbose: bool = True) -> Dict:
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Task: {task}")
            print(f"{'='*60}")
        
        strategy = self.router.select(self.strategies)
        if verbose:
            print(f"🎯 Strategy: {strategy}")
        
        memory = self.memory.retrieve(task)
        if verbose:
            print(f"💾 Memory: {len(memory)} episodes")
        
        plan = self.planner.plan(task, memory)
        if verbose:
            print(f"📋 Plan: {len(plan)} steps")
        
        results = self.engine.execute(plan)
        
        reward = sum(r.confidence for r in results.values() if r.success)
        self.router.update(strategy, reward)
        
        for step in plan:
            if step.id in results:
                self.memory.add(task, step, results[step.id], strategy)
        
        successful = [r for r in results.values() if r.success]
        
        return {
            "success": all(r.success for r in results.values()),
            "output": successful[-1].output if successful else None,
            "strategy": strategy,
            "steps_total": len(results),
            "steps_successful": len(successful),
            "duration_sec": time.time() - start_time,
        }
