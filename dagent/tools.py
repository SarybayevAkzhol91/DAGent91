from typing import Dict


def search_tool(input_data: Dict) -> Dict:
    query = input_data.get("query", "")
    return {
        "output": f"Search results for '{query}'",
        "success": True,
        "confidence": 0.9
    }


def summarize_tool(input_data: Dict) -> Dict:
    text = input_data.get("text", "")
    return {
        "output": f"Summary: {text[:100]}",
        "success": True,
        "confidence": 0.85
    }


DEFAULT_TOOLS = {
    "search": search_tool,
    "summarize": summarize_tool,
  }
