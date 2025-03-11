import anthropic
from openai import OpenAI
from typing import Dict, Any, List, Optional, Callable, Union, Generator
import json
from dotenv import load_dotenv
import os
from datetime import datetime
from .token_debugger import TokenDebugger
from .providers import LLMProvider, get_tool_config, format_tool_result
from .error_logger import ToolErrorLogger
from .model_costs import get_model_type
from .anthropic_handler import process_anthropic_stream, create_anthropic_stream, format_tool_result_message as format_anthropic_result
from .openai_handler import (
    process_openai_stream,
    create_openai_stream,
    prepare_openai_messages,
    format_tool_result_message as format_openai_result
)

# Load environment variables
load_dotenv()

react_prompt = """
You are a highly capable and thoughtful assistant that employs a ReAct (Reasoning and Acting) strategy. For every query, follow this iterative process:

1. **Initial Reasoning:** Start by writing a detailed chain-of-thought enclosed within <thinking> and </thinking> tags. This should include your initial reasoning, hypotheses, and any uncertainties.
2. **Tool Invocation:** If you determine that further information is needed, or that a specific computation or external lookup is required.
   Then, wait for the result of that tool call.
3. **Iterative Refinement:** Once you receive the tool's output, update your chain-of-thought inside a new <thinking>...</thinking> block that integrates the tool's result. Continue to iterate—refining your chain-of-thought and making additional tool calls if necessary. Do not finalize your answer until you have thoroughly considered all angles and feel that you have thought about the best response.
4. **Final Answer:** After multiple iterations of reasoning and tool use, and once you are confident that you have fully addressed the query, provide your final answer clearly labeled as FINAL ANSWER.
5. **Clarity and Transparency:** Include all your reasoning steps (the <thinking>...</thinking> blocks with any tool invocations and their results) along with the final answer so that the user can follow your complete thought process.

Use this structured approach to handle complex queries, ensuring that each stage of your internal reasoning is transparent by using the <thinking> tags. Bellow if your specific role in this case:   
"""

class AugmentedLLM:
    def __init__(
        self,
        system_prompt: str,
        provider: Union[LLMProvider, str],
        model_name: Optional[str] = None,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        use_react: bool = False,
        debug_tools: bool = False,
        debug_tokens: bool = False,
        reasoning_effort: Optional[str] = "medium",
        debug_settings: bool = False,
        debug_messages: bool = False
    ):
        """Initialize AugmentedLLM with configuration"""
        # Convert string provider to enum if needed
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        self.provider = provider
        
        # Validate reasoning_effort if provided
        if reasoning_effort and reasoning_effort not in ["low", "medium", "high"]:
            raise ValueError("reasoning_effort must be one of: 'low', 'medium', 'high'")
        self.reasoning_effort = reasoning_effort
        
        # Initialize error logger
        self.error_logger = ToolErrorLogger()
        
        # Initialize based on provider
        if provider == LLMProvider.ANTHROPIC:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            self.client = anthropic.Anthropic()
            self.model_name = model_name or "claude-3-5-sonnet-20241022"
            self.max_tokens = max_tokens or 8192
        else:  # OpenAI
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = OpenAI()
            self.model_name = model_name or "gpt-4"
            self.max_tokens = max_tokens or 4096
            
        self.temperature = temperature
        self.debug_tools = debug_tools
        self.debug_tokens = debug_tokens
        self.debug_settings = debug_settings
        self.debug_messages = debug_messages
        
        # Initialize token debugger if enabled
        if self.debug_tokens:
            self.token_debugger = TokenDebugger(self.model_name)
        
        # Add current date and time to system prompt
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_time_info = f"\nCurrent Date and Time: {current_time}\n"
        
        # Combine prompts if using react
        if use_react:
            self.system_prompt = react_prompt + system_prompt + date_time_info 
        else:
            self.system_prompt = system_prompt + date_time_info
            
        # Log system prompt tokens if debugging
        if self.debug_tokens:
            self.token_debugger.log_message("system", self.system_prompt)
        
        # Initialize message history and tools
        self.messages = []
        self.tools: List[Dict[str, Any]] = []
        self.tool_registry: Dict[str, Callable] = {}
        
    def add_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable
    ) -> None:
        """Register a new tool with the LLM"""
        tool_config = get_tool_config(name, description, input_schema, self.provider)
        self.tools.append(tool_config)
        self.tool_registry[name] = handler
        
    def log_tools(self) -> None:
        """Save the current tools configuration to a JSON file."""
        try:
            os.makedirs("logs/tools", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tools_config_{timestamp}.json"
            filepath = os.path.join("logs/tools", filename)
            
            # Create a clean version of tools without handler references
            tools_for_logging = []
            for tool in self.tools:
                tool_copy = tool.copy()
                # Remove any callable objects that can't be serialized
                if "handler" in tool_copy:
                    del tool_copy["handler"]
                tools_for_logging.append(tool_copy)
            
            with open(filepath, "w") as f:
                json.dump({
                    "model": self.model_name,
                    "provider": self.provider.value,
                    "registered_tools": tools_for_logging,
                    "total_tools": len(self.tools)
                }, f, indent=2)
                
            if self.debug_tools:
                print(f"\n[Debug] Tools configuration saved to: {filepath}")
        except Exception as e:
            if self.debug_tools:
                print(f"\n[Debug] Error saving tools configuration: {str(e)}")
        
    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a registered tool with the given input"""
        if tool_name not in self.tool_registry:
            error_msg = f"Tool '{tool_name}' not found in registry"
            if self.debug_tools:
                print(f"\n[Debug] Error: {error_msg}")
            return error_msg
            
        tool_handler = self.tool_registry[tool_name]
        try:
            if self.debug_tools:
                print(f"\n[Debug] Executing tool: {tool_name}")
                print(f"[Debug] Tool input: {json.dumps(tool_input, indent=2)}")
                
            result = tool_handler(**tool_input)
            
            # Log tool result tokens if debugging
            if self.debug_tokens:
                self.token_debugger.log_message("tool", str(result), is_tool_result=True)
            
            if self.debug_tools:
                print(f"[Debug] Tool result: {format_tool_result(result)}\n")
                
            return str(result)
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            if self.debug_tools:
                print(f"\n[Debug] {error_msg}")
                
            # Log the error with full context
            context = {
                "model_name": self.model_name,
                "debug_mode": self.debug_tools,
                "messages": self.messages,  # Include conversation context
                "tools_registered": list(self.tool_registry.keys())
            }
            
            error_log_path = self.error_logger.log_error(
                tool_name=tool_name,
                tool_input=tool_input,
                error=e,
                provider=self.provider.value,
                context=context
            )
            
            # Generate a test file for this error
            self.error_logger.create_test_from_error(error_log_path)
            
            return error_msg

    def process_stream(self, stream) -> Generator[str, None, None]:
        """Process a message stream and handle tool usage"""
        if self.provider == LLMProvider.ANTHROPIC:
            result = yield from process_anthropic_stream(stream, self.messages, self.debug_tools)
            
            # Log assistant message tokens if debugging
            if self.debug_tokens and result.get("content"):
                self.token_debugger.log_message("assistant", result["content"])
            
            if result["stop_reason"] == "tool_use":
                tool_block = next(
                    (block for block in result["message"]["content"] 
                     if block["type"] == "tool_use"),
                    None
                )
                
                if tool_block:
                    tool_result = self.execute_tool(
                        tool_block["name"],
                        tool_block["input"]
                    )
                    if not self.debug_tools:
                        yield f"\n[Tool Result]\n{tool_result}\n"
                    
                    # Add tool result to messages
                    self.messages.append(format_anthropic_result(tool_block, tool_result))
                    
                    params = {
                    "messages": self.messages,
                    "model": self.model_name,
                    "system": self.system_prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "tools": self.tools,
                    "stream": True
                    }
                    # Create new stream with updated messages
                    new_stream = create_anthropic_stream(
                        self.client,
                       **params
                    )
                    
                    # Process the new stream
                    if not self.debug_tools:
                        yield "\n[Continuing conversation with tool result...]\n"
                    yield from self.process_stream(new_stream)
            else:
                # Only log tokens for complete messages without tool calls
                if self.debug_tokens and result.get("content"):
                    self.token_debugger.log_message("assistant", result["content"])
                    
        else:  # OpenAI
            result = yield from process_openai_stream(stream, self.messages, self.debug_tools)
            
            # Log assistant message tokens if debugging
            if self.debug_tokens and result.get("content"):
                self.token_debugger.log_message("assistant", result["content"])
            
            if result["has_tool_calls"]:
                for tool_call in result["tool_calls"].values():
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        tool_result = self.execute_tool(
                            tool_call["function"]["name"],
                            args
                        )
                        
                        # Add tool result to messages
                        self.messages.append(format_openai_result(tool_call, tool_result))
                        
                        if not self.debug_tools:
                            yield f"\n[Tool Result]\n{tool_result}\n"
                            
                    except json.JSONDecodeError as e:
                        error_msg = f"Error parsing tool arguments: {e}"
                        if self.debug_tools:
                            print(f"\n[Debug] {error_msg}")
                        yield f"\n[Error] {error_msg}\n"
                
                # Continue conversation with tool results
                params = {
                    "messages": prepare_openai_messages(self.messages, self.system_prompt, self.model_name),
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "tools": self.tools,
                    "stream": True
                }
                new_stream = create_openai_stream(
                    self.client,
                    debug_tools=self.debug_tools,
                    **params
                )
                
                if not self.debug_tools:
                    yield "\n[Continuing conversation with tool result...]\n"
                yield from self.process_stream(new_stream)
            else:
                # Only log tokens for complete messages without tool calls
                if self.debug_tokens and result.get("content"):
                    self.token_debugger.log_message("assistant", result["content"])
        
    def generate(self, message: str):
        """Generate a response to the given message, with streaming by default"""
        # Log user message tokens if debugging
        if self.debug_tokens:
            self.token_debugger.log_message("user", message)
            
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": message
        })
        
        # Create the stream based on provider
        if self.provider == LLMProvider.ANTHROPIC:
            params = {
                "messages": self.messages,
                "model": self.model_name,
                "system": self.system_prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "tools": self.tools,
                "stream": True
            }
            if self.debug_settings:
                debug_params = params.copy()
                del debug_params["messages"]  # Remove messages from debug output
                print("\n[Debug Settings] Anthropic API call parameters:")
                print(json.dumps(debug_params, indent=2))
            stream = create_anthropic_stream(self.client, **params)
        else:  # OpenAI
            # Prepare messages with model name for reasoning check
            prepared_messages = prepare_openai_messages(
                self.messages,
                self.system_prompt,
                self.model_name
            )
            
            # Add reasoning_effort for reasoning models if specified
            params = {
                "messages": prepared_messages,
                "model": self.model_name,
                "temperature": self.temperature,
                "tools": self.tools,
                "stream": True
            }
            
            # Handle max tokens parameter based on model type
            if get_model_type(self.model_name) == "reasoning":
                params["max_completion_tokens"] = self.max_tokens
                if self.reasoning_effort:
                    params["reasoning_effort"] = self.reasoning_effort
            else:
                params["max_tokens"] = self.max_tokens
                
            if self.debug_settings:
                debug_params = {k: v for k, v in params.items() if k != "messages"}
                print("\n[Debug Settings] OpenAI API call parameters:")
                print(json.dumps(debug_params, indent=2))
            
            stream = create_openai_stream(self.client, debug_tools=self.debug_tools, **params)
        
        try:
            # Process the stream
            yield from self.process_stream(stream)
        finally:
            # Print token debug info at the end if enabled
            if self.debug_tokens:
                self.token_debugger.print_debug_info()
            
            # Save messages to file if debug_messages is enabled
            if self.debug_messages:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"debug_messages_{timestamp}.json"
                os.makedirs("logs/messages", exist_ok=True)
                filepath = os.path.join("logs/messages", filename)
                
                with open(filepath, "w") as f:
                    json.dump({
                        "model": self.model_name,
                        "provider": self.provider.value,
                        "system_prompt": self.system_prompt,
                        "messages": self.messages
                    }, f, indent=2)
                print(f"\n[Debug Messages] Conversation saved to: {filepath}")
        
    def clear_history(self) -> None:
        """Clear message history except system prompt"""
        self.messages = []
        if self.debug_tokens:
            self.token_debugger = TokenDebugger(self.model_name)
            self.token_debugger.log_message("system", self.system_prompt) 