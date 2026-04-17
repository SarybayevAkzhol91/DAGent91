"""
DAGent - Main Agent Class
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from dagent.memory import Memory
from dagent.router import UCBRouter
from dagent.planner import Planner
from dagent.executor import Executor, DAGEngine, Step, Result


class DAGent:
    """
    Main DAGent class for executing tasks with LLM and tools.
    """
    
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
        
        for attempt in range(self.max_replans + 1):
            results = self.engine.execute(plan)
            success = all(r.success for r in results.values())
            if success or attempt == self.max_replans:
                break
            if verbose:
                print(f"🔄 Replan attempt {attempt + 1}")
            plan = self._replan_failed(task, plan, results)
        
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
            "results": {k: {"success": v.success, "confidence": v.confidence} 
                       for k, v in results.items()}
        }
    
    def _replan_failed(self, task: str, plan: List[Step], 
                       results: Dict[str, Result]) -> List[Step]:
        failed_ids = [sid for sid, r in results.items() if not r.success]
        failed_steps = [s for s in plan if s.id in failed_ids]
        if not failed_steps:
            return plan
        return self.planner.replan(task, failed_steps)
