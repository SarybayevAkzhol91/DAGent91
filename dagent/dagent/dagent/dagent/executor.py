import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import threading


@dataclass(frozen=True)
class Step:
    id: str
    tool: str
    input: Dict
    deps: List[str] = field(default_factory=list)
    timeout: float = 30.0
    retries: int = 2


@dataclass
class Result:
    step_id: str
    output: Any
    success: bool
    confidence: float
    error: Optional[str] = None
    duration_ms: float = 0.0


class Executor:
    def __init__(self, tools: Dict[str, Any]):
        self.tools = tools
    
    def run_step(self, step: Step, resolved_input: Dict) -> Result:
        start = time.time()
        tool = self.tools.get(step.tool)
        if not tool:
            return Result(step.id, None, False, 0.0, f"Tool '{step.tool}' not found")
        
        try:
            output = tool(resolved_input)
            if isinstance(output, dict):
                return Result(step.id, output.get("output", output), output.get("success", True), output.get("confidence", 0.8), output.get("error"))
            return Result(step.id, output, True, 0.8)
        except Exception as e:
            return Result(step.id, None, False, 0.0, str(e))


class DAGEngine:
    def __init__(self, executor: Executor, max_workers: int = 4):
        self.executor = executor
        self.max_workers = max_workers
    
    def execute(self, steps: List[Step]) -> Dict[str, Result]:
        results = {}
        lock = threading.Lock()
        
        def resolve_input(step: Step) -> Dict:
            resolved = {}
            for k, v in step.input.items():
                if isinstance(v, dict) and "ref" in v:
                    ref_id = v["ref"]
                    with lock:
                        if ref_id in results:
                            resolved[k] = results[ref_id].output
                        else:
                            resolved[k] = None
                else:
                    resolved[k] = v
            return resolved
        
        def is_ready(step: Step) -> bool:
            if not step.deps:
                return True
            with lock:
                return all(dep in results and results[dep].success for dep in step.deps)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            while len(results) < len(steps):
                for step in steps:
                    if step.id not in results and step.id not in futures:
                        if is_ready(step):
                            resolved = resolve_input(step)
                            future = pool.submit(self.executor.run_step, step, resolved)
                            futures[future] = step.id
                for future in list(futures.keys()):
                    if future.done():
                        step_id = futures.pop(future)
                        result = future.result()
                        with lock:
                            results[step_id] = result
        return results
