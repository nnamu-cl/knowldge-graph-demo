from openai import OpenAI
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel

class OpenAIChatInterface:
    def __init__(self, model_name: str = "gpt-4o", initial_messages: Optional[List[Dict[str, str]]] = None, temperature: float = 1.0):
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.response_format = None
        self.messages = initial_messages or []
        self.schema_class = None
        
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
        
    def get_completion(self) -> Any:
        """Get a completion from the OpenAI API and save it to the messages list."""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature,
        )
        assistant_message = completion.choices[0].message
        
        # Save the assistant's response to the messages list
        self.messages.append({
            "role": "assistant",
            "content": assistant_message.content
        })
        
        return assistant_message
    
    def enable_structured_output(self, schema: Type[BaseModel]) -> None:
        """Enable structured output using a Pydantic model."""
        self.schema_class = schema
        self.response_format = {"type": "json_schema", "schema": schema}
    
    def parse_structured_output(self, schema: Optional[Type[BaseModel]] = None) -> Any:
        """Send messages and parse the response using the provided schema."""
        # Use the provided schema or fall back to the previously set schema
        if schema is not None:
            self.enable_structured_output(schema)
        elif self.schema_class is None:
            raise ValueError("No schema provided. Call enable_structured_output first or provide a schema.")
            
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=self.messages,
            response_format=self.schema_class,
            temperature=self.temperature,
        )
        
        # Save the assistant's response to the messages list
        parsed_response = completion.choices[0].message.parsed
        self.messages.append({
            "role": "assistant",
            "content": str(parsed_response)  # Convert the parsed object to string for message history
        })
        
        return parsed_response
