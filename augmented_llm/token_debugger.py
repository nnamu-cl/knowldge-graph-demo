import tiktoken
from typing import Dict, List, Optional, Union, Any
import json
from .model_costs import get_model_costs, get_model_type, MODEL_NAME_MAPPING

class TokenDebugger:
    def __init__(self, model_name: str):
        """Initialize token debugger with model information."""
        self.original_model_name = model_name
        self.model_name = self.normalize_model_name(model_name)
        self.model_type = get_model_type(self.model_name)
        self.costs = get_model_costs(self.model_name)
        
        # Initialize counters
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tool_tokens = 0
        self.message_count = 0
        self.tool_call_count = 0
        self.conversation_history: List[Dict[str, Any]] = []
        
    @staticmethod
    def normalize_model_name(model_name: str) -> str:
        """Normalize model name by removing date suffixes and standardizing format."""
        # Remove date suffix if present (e.g., -20241022)
        model_name = model_name.split('-2024')[0]
        
        # Convert to lowercase and remove spaces/dots for comparison
        normalized = model_name.lower().replace(" ", "").replace(".", "")
        
        # Check against known model variations
        for standard_name, variations in MODEL_NAME_MAPPING.items():
            if normalized in [v.lower().replace(" ", "").replace(".", "") for v in variations]:
                return standard_name
                
        return model_name
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a string using tiktoken."""
        enc = tiktoken.encoding_for_model("gpt-4")  # Use GPT-4 encoding as default
        return len(enc.encode(text))
        
    def log_message(self, role: str, content: Union[str, List[Dict]], is_tool_result: bool = False):
        """Log a message and count its tokens."""
        if isinstance(content, list):
            # Handle structured content (like tool calls)
            content_str = json.dumps(content)
        else:
            content_str = str(content)
            
        token_count = self.count_tokens(content_str)
        
        message_data = {
            "role": role,
            "token_count": token_count,
            "is_tool_result": is_tool_result
        }
        
        self.conversation_history.append(message_data)
        self.message_count += 1
        

        if role == "assistant":
            self.total_output_tokens += token_count
        else:
            self.total_input_tokens += token_count
            
        if is_tool_result:
            self.total_tool_tokens += token_count
            self.tool_call_count += 1
            
    def calculate_costs(self) -> Dict[str, float]:
        """Calculate costs based on token usage.
        
        Costs are calculated per 1000 tokens using the following formula:
        - Input cost = (total_input_tokens / 1000) * input_rate
        - Output cost = (total_output_tokens / 1000) * output_rate
        - Tool tokens are counted as input tokens but may use cached rate if applicable
        
        Returns:
            Dict containing input_cost, output_cost, and total_cost (all in USD)
        """
        # Calculate regular input cost (excluding tool tokens)
        regular_input_tokens = self.total_input_tokens - self.total_tool_tokens
        regular_input_cost = (regular_input_tokens / 1000) * self.costs["input"]
        
        # Calculate tool input cost (using cached rate if available)
        tool_input_cost = (self.total_tool_tokens / 1000) * self.costs.get("cached_input", self.costs["input"])
        
        # Calculate total input cost
        input_cost = regular_input_cost + tool_input_cost
        
        # Calculate output cost
        output_cost = (self.total_output_tokens / 1000) * self.costs["output"]
        
        # Calculate total cost
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": round(input_cost, 6),
            "input_details": {
                "regular_input_cost": round(regular_input_cost, 6),
                "tool_input_cost": round(tool_input_cost, 6)
            },
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6)
        }
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of token usage and costs."""
        costs = self.calculate_costs()
        
        return {
            "model": {
                "original": self.original_model_name,
                "normalized": self.model_name
            },
            "message_stats": {
                "total_messages": self.message_count,
                "tool_calls": self.tool_call_count
            },
            "token_stats": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "tool_tokens": self.total_tool_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens
            },
            "costs": costs,
            "message_history": [
                {
                    "role": msg["role"],
                    "tokens": msg["token_count"],
                    "is_tool": msg["is_tool_result"]
                }
                for msg in self.conversation_history
            ]
        }
        
    def print_debug_info(self):
        """Print detailed debug information about token usage and costs."""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("TOKEN USAGE AND COST SUMMARY")
        print("="*50)
        
        print(f"\nModel: {summary['model']['original']}")
        print(f"Normalized Model: {summary['model']['normalized']}")
        
        print("\nMessage Statistics:")
        print(f"- Total Messages: {summary['message_stats']['total_messages']}")
        print(f"- Tool Calls: {summary['message_stats']['tool_calls']}")
        
        print("\nToken Statistics:")
        print(f"- Input Tokens: {summary['token_stats']['input_tokens']:,}")
        print(f"  • Regular Input: {summary['token_stats']['input_tokens'] - summary['token_stats']['tool_tokens']:,}")
        print(f"  • Tool Input: {summary['token_stats']['tool_tokens']:,}")
        print(f"- Output Tokens: {summary['token_stats']['output_tokens']:,}")
        print(f"- Total Tokens: {summary['token_stats']['total_tokens']:,}")
        
        print("\nCosts (USD):")
        print(f"- Input Costs: ${summary['costs']['input_cost']:.6f}")
        print(f"  • Regular Input: ${summary['costs']['input_details']['regular_input_cost']:.6f}")
        print(f"  • Tool Input: ${summary['costs']['input_details']['tool_input_cost']:.6f}")
        print(f"- Output Cost: ${summary['costs']['output_cost']:.6f}")
        print(f"- Total Cost: ${summary['costs']['total_cost']:.6f}")
        
        print("\nMessage History:")
        for i, msg in enumerate(summary['message_history'], 1):
            role = msg['role'].ljust(10)
            tool_indicator = " (tool)" if msg['is_tool'] else "      "
            print(f"{i:2d}. {role}{tool_indicator} - {msg['tokens']:,} tokens")
            
        print("\n" + "="*50) 