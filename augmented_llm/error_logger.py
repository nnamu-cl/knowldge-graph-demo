import json
import os
from datetime import datetime
from typing import Dict, Any
import traceback

class ToolErrorLogger:
    def __init__(self, log_dir: str = "logs/tool_errors"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def log_error(self, 
                  tool_name: str, 
                  tool_input: Dict[str, Any], 
                  error: Exception,
                  provider: str,
                  context: Dict[str, Any] = None) -> str:
        """Log a tool execution error with full context for debugging.
        
        Args:
            tool_name: Name of the tool that failed
            tool_input: Input parameters passed to the tool
            error: The exception that occurred
            provider: The LLM provider being used (Anthropic/OpenAI)
            context: Additional context about the error
            
        Returns:
            Path to the error log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{tool_name}_error.json"
        filepath = os.path.join(self.log_dir, filename)
        
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "tool_input": tool_input,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            },
            "provider": provider,
            "context": context or {}
        }
        
        with open(filepath, "w") as f:
            json.dump(error_data, f, indent=2, default=str)
            
        print(f"\n[Error Log] Tool execution error logged to: {filepath}")
        return filepath
        
    def create_test_from_error(self, error_log_path: str) -> str:
        """Generate a test file from an error log to help reproduce and fix the issue.
        
        Args:
            error_log_path: Path to the error log file
            
        Returns:
            Path to the generated test file
        """
        with open(error_log_path) as f:
            error_data = json.load(f)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_filename = f"test_{error_data['tool_name']}_{timestamp}.py"
        test_filepath = os.path.join("tests", "tool_error_tests", test_filename)
        
        os.makedirs(os.path.dirname(test_filepath), exist_ok=True)
        
        test_code = f'''import pytest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from utils.augmented_llm import AugmentedLLM
from utils.augmented_llm.providers import LLMProvider

def test_{error_data["tool_name"]}_error_reproduction():
    """Test to reproduce tool execution error from {error_data["timestamp"]}"""
    # Setup
    llm = AugmentedLLM(
        system_prompt="Test prompt",
        provider=LLMProvider.{error_data["provider"].upper()},
    )
    
    # Tool input that caused the error
    tool_input = {json.dumps(error_data["tool_input"], indent=4)}
    
    # Original error: {error_data["error"]["message"]}
    # Attempt to execute the tool
    try:
        result = llm.execute_tool(
            tool_name="{error_data["tool_name"]}",
            tool_input=tool_input
        )
        assert False, f"Expected error but got result: {{result}}"
    except Exception as e:
        # Verify we get the same type of error
        assert type(e).__name__ == "{error_data["error"]["type"]}"
        # TODO: Add more specific assertions about the error

if __name__ == "__main__":
    # This allows running the test directly with python
    pytest.main([__file__])
'''
        
        with open(test_filepath, "w") as f:
            f.write(test_code)
            
        print(f"\n[Test Generated] Test file created at: {test_filepath}")
        return test_filepath 