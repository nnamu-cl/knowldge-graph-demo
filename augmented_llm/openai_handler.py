from openai import OpenAI
from typing import Dict, Any, Generator, List
import json
from .model_costs import get_model_type

def process_openai_stream(stream, messages: List[Dict[str, Any]], debug_tools: bool = False) -> Generator[str, None, Dict[str, Any]]:
    """Process OpenAI message stream and handle tool usage"""
    current_content = ""
    tool_calls = {}
    has_tool_calls = False
    
    try:
        for chunk in stream:
            if not hasattr(chunk, 'choices'):
                continue
                
            delta = chunk.choices[0].delta
            
            # Handle content
            if hasattr(delta, 'content') and delta.content is not None:
                current_content += delta.content
                # Always yield actual content, but handle debug messages differently
                if debug_tools:
                    if not delta.content.startswith("[Debug]"):
                        yield delta.content
                    # Print debug messages to console directly
                    else:
                        print(delta.content)
                else:
                    yield delta.content

            # Handle tool calls
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                has_tool_calls = True
                for tool_call in delta.tool_calls:
                    index = tool_call.index
                    if index not in tool_calls:
                        tool_calls[index] = {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": ""
                            }
                        }
                    if tool_call.function.arguments:
                        tool_calls[index]["function"]["arguments"] += tool_call.function.arguments
    except Exception as e:
        print(f"\n[Debug] Error processing stream: {str(e)}")
        raise

    # Prepare message to return
    message = {
        "role": "assistant",
        "content": current_content if not has_tool_calls else None,
        "tool_calls": list(tool_calls.values()) if has_tool_calls else None
    }
    
    messages.append(message)
    return {
        "message": message,
        "has_tool_calls": has_tool_calls,
        "tool_calls": tool_calls,
        "content": current_content
    }

def create_openai_stream(client: OpenAI, debug_tools: bool = False, **kwargs):
    """Create a stream using OpenAI's API"""
    # Ensure stream parameter is set
    kwargs["stream"] = True
    
    # Transform tools into correct format if present
    if "tools" in kwargs and kwargs["tools"]:
        transformed_tools = []
        for tool in kwargs["tools"]:
            if tool["type"] == "function":
                # Extract existing properties
                function_data = tool["function"]
                parameters = function_data["parameters"]
                
                # Clean properties by removing default values
                cleaned_properties = {}
                for prop_name, prop_data in parameters["properties"].items():
                    cleaned_prop = {k: v for k, v in prop_data.items() if k != "default"}
                    cleaned_properties[prop_name] = cleaned_prop
                
                # When strict is true, all properties must be required
                all_properties = list(cleaned_properties.keys())
                
                # Create new tool structure
                transformed_tool = {
                    "type": "function",
                    "function": {
                        "name": function_data["name"],
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "required": all_properties,
                            "properties": cleaned_properties,
                            "additionalProperties": False
                        },
                        "description": function_data["description"]
                    }
                }
                transformed_tools.append(transformed_tool)
        
        # Replace original tools with transformed ones
        kwargs["tools"] = transformed_tools
    
    return client.chat.completions.create(**kwargs)

def prepare_openai_messages(messages: List[Dict[str, Any]], system_prompt: str, model_name: str) -> List[Dict[str, Any]]:
    """Prepare messages for OpenAI format"""
    # Check if this is a reasoning model
    is_reasoning = get_model_type(model_name) == "reasoning"
    
    prepared_messages = [{
        "role": "developer" if is_reasoning else "system",
        "content": system_prompt
    }]
    
    for msg in messages:
        if msg["role"] == "user":
            prepared_messages.append({
                "role": "user",
                "content": msg["content"]
            })
        elif msg["role"] == "assistant":
            if msg.get("tool_calls"):
                prepared_messages.append({
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": msg["tool_calls"]
                })
            else:
                prepared_messages.append({
                    "role": "assistant",
                    "content": msg.get("content", "")
                })
        elif msg["role"] == "tool":
            prepared_messages.append({
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "content": msg["content"]
            })
            
    return prepared_messages

def format_tool_result_message(tool_call: Dict[str, Any], result: str) -> Dict[str, Any]:
    """Format tool result message for OpenAI"""
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": result
    } 