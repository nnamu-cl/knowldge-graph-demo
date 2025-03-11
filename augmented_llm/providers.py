from enum import Enum
from typing import Dict, Any, List
import json

class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"

def format_tool_result(tool_result: Any) -> str:
    """Format tool result for consistent output"""
    if isinstance(tool_result, (dict, list)):
        return json.dumps(tool_result, indent=2)
    return str(tool_result)

def get_tool_config(name: str, description: str, input_schema: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    """Get provider-specific tool configuration"""
    if provider == LLMProvider.ANTHROPIC:
        # Extract required fields and clean properties similar to OpenAI
        required = [k for k, v in input_schema.items() if v.get("required", False)]
        properties = {
            k: {key: value for key, value in v.items() if key != "required"}
            for k, v in input_schema.items()
        }
        
        return {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    else:  # OpenAI
        # Ensure required fields are properly extracted
        required = [k for k, v in input_schema.items() if v.get("required", False)]
        
        # Remove 'required' field from individual properties
        properties = {
            k: {key: value for key, value in v.items() if key != "required"}
            for k, v in input_schema.items()
        }
        
        return {
            "type": "function",  # Ensure type is set for OpenAI
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        } 