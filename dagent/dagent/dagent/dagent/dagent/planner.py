import uuid
import json
from typing import Any, Dict, List

from dagent.executor import Step


class Planner:
    def __init__(self, llm):
        self.llm = llm
    
    def plan(self, task: str, memory: List[Dict]) -> List[Step]:
        prompt = self._build_prompt(task, memory)
        raw_plan = self.llm.plan(prompt)
        return self._parse_plan(raw_plan)
    
    def _build_prompt(self, task: str, memory: List[Dict]) -> str:
        memory_text = "\n".join([f"- {m['tool']}: success={m['success']}" for m in memory[:5]])
        return f"Task: {task}\nMemory:\n{memory_text}\nReturn JSON array of steps with id, tool, input, deps."
    
    def _parse_plan(self, raw_plan: Any) -> List[Step]:
        if isinstance(raw_plan, str):
            try:
                raw_plan = json.loads(raw_plan)
            except:
                raw_plan = []
        steps = []
        for p in raw_plan:
            steps.append(Step(
                id=p.get("id", str(uuid.uuid4())),
                tool=p["tool"],
                input=p.get("input", {}),
                deps=p.get("deps", [])
            ))
        return steps
