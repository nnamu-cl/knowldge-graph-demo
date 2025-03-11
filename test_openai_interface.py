import unittest
from openai_interacter import OpenAIChatInterface
from pydantic import BaseModel
from typing import List

# Define a Pydantic model for structured output testing
class WeatherInfo(BaseModel):
    temperature: float
    conditions: str
    forecast: List[str]

def main():
    # # Test 1: Basic chat completion
    # print("\n=== Test 1: Basic Chat Completion ===")
    # chat = OpenAIChatInterface()
    # chat.add_message("system", "You are a helpful assistant.")
    # chat.add_message("user", "What is the capital of France?")
    
    # response = chat.get_completion()
    # print(f"Response: {response.content}")
    
    # # Continue the conversation
    # chat.add_message("user", "And what is its population?")
    # response = chat.get_completion()
    # print(f"Response: {response.content}")
    
    # # Test 2: Using initial messages
    # print("\n=== Test 2: Using Initial Messages ===")
    # initial_messages = [
    #     {"role": "system", "content": "You are a helpful assistant that responds in a concise manner."},
    #     {"role": "user", "content": "What are the three primary colors?"}
    # ]
    
    # chat2 = OpenAIChatInterface(initial_messages=initial_messages)
    # response = chat2.get_completion()
    # print(f"Response: {response.content}")
    
    # Test 3: Structured output with Pydantic
    print("\n=== Test 3: Structured Output with Pydantic ===")
    chat3 = OpenAIChatInterface()
    chat3.add_message("system", "You provide weather information in a structured format.")
    chat3.add_message("user", "Give a random possible weather forecast for 5 days from now.")
    
    # Get structured response
    structured_response = chat3.parse_structured_output(WeatherInfo)
    print(f"Temperature: {structured_response.temperature}°C")
    print(f"Conditions: {structured_response.conditions}")
    print(f"Forecast: {', '.join(structured_response.forecast)}")
    
    # Follow up question
    chat3.add_message("user", "How would this weather affect outdoor activities?")
    follow_up_response = chat3.get_completion()
    print(f"\nFollow-up Response: {follow_up_response.content}")
    
    # Test 4: Different model
    #print("\n=== Test 4: Using a Different Model ===")
    chat4 = OpenAIChatInterface(model_name="gpt-4o-mini")
    chat4.add_message("system", "You are a helpful assistant.")
    chat4.add_message("user", "Explain quantum computing in one sentence.")
    
    response = chat4.get_completion()
    #print(f"Response: {response.content}")

if __name__ == "__main__":
    main() 