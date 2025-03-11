import anthropic
from typing import Dict, Any, Generator, List
import json

def process_anthropic_stream(stream, messages: List[Dict[str, Any]], debug_tools: bool = False) -> Generator[str, None, Dict[str, Any]]:
    """Process Anthropic message stream and handle tool usage"""
    current_message = None
    current_block = None
    current_json_accumulator = ""
    stop_reason = None
    current_content = ""
    
    for event in stream:
        if event.type == "message_start":
            current_message = {
                "role": "assistant",
                "content": []
            }
            current_content = ""
            if debug_tools:
                print("[Debug] Stream Started")
            else:
                yield "[Stream Started]\n"
            
        elif event.type == "content_block_start":
            if hasattr(event.content_block, 'type'):
                if event.content_block.type == "text":
                    current_block = {"type": "text", "text": ""}
                    current_message["content"].append(current_block)
                elif event.content_block.type == "tool_use":
                    current_block = {
                        "type": "tool_use",
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input": {}
                    }
                    current_message["content"].append(current_block)
                    if debug_tools:
                        print(f"\n[Debug] Tool Use Started: {current_block['name']}")
                    else:
                        yield f"\n[Tool Use Started: {current_block['name']}]\n"
            
        elif event.type == "content_block_delta":
            if hasattr(event.delta, 'text'):
                text = event.delta.text
                current_content += text
                if not debug_tools or not text.startswith("[Debug]"):
                    yield text
                if current_block and current_block["type"] == "text":
                    current_block["text"] += text
            elif hasattr(event.delta, 'partial_json'):
                if event.delta.partial_json.strip():
                    current_json_accumulator += event.delta.partial_json
                    if current_json_accumulator.endswith("}"):
                        try:
                            tool_input = json.loads(current_json_accumulator)
                            if current_block and current_block["type"] == "tool_use":
                                current_block["input"] = tool_input
                                if debug_tools:
                                    print(f"[Debug] Tool Input: {json.dumps(tool_input, indent=2)}")
                                else:
                                    yield f"\n[Tool Input: {tool_input}]\n"
                        except json.JSONDecodeError:
                            pass
            
        elif event.type == "content_block_stop":
            if current_block and current_block["type"] == "tool_use":
                current_json_accumulator = ""
            
        elif event.type == "message_delta":
            if event.delta.stop_reason:
                stop_reason = event.delta.stop_reason
                if debug_tools:
                    print(f"[Debug] Stop Reason: {stop_reason}")
                else:
                    yield f"\n[Stop Reason: {stop_reason}]\n"
            
        elif event.type == "message_stop":
            if debug_tools:
                print("[Debug] Message Complete")
            else:
                yield "\n[Message Complete]\n"
                
            if current_message:
                messages.append(current_message)
                return {
                    "message": current_message,
                    "stop_reason": stop_reason,
                    "content": current_content
                }

def create_anthropic_stream(client: anthropic.Anthropic, **kwargs):
    """Create a stream using Anthropic's API"""
    return client.messages.create(**kwargs)

def format_tool_result_message(tool_block: Dict[str, Any], result: str) -> Dict[str, Any]:
    """Format tool result message for Anthropic"""
    return {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_block["id"],
            "content": result
        }]
    } 