"""
Model cost configurations and categorizations for different AI models.
Costs are specified in USD per 1,000,000 tokens.
"""

MODEL_COSTS = {
    # =====================
    # REASONING MODELS
    # =====================
    "reasoning": {
        "o1": {
            "input": 0.015,
            "cached_input": 0.0075,
            "output": 0.060
        },
        "o1-mini": {
            "input": 0.0011,
            "cached_input": 0.00055,
            "output": 0.0044
        },
        "o3-mini": {
            "input": 0.0011,
            "cached_input": 0.00055,
            "output": 0.0044
        }
    },

    # =====================
    # NON-REASONING MODELS
    # =====================
    "non_reasoning": {
        # GPT Models
        "gpt-4o": {
            "input": 0.0025,
            "cached_input": 0.00125,
            "output": 0.010
        },
        "chatgpt-4o-latest": {
            "input": 0.0025,
            "cached_input": 0.00125,
            "output": 0.010
        },
        "gpt-4o-mini": {
            "input": 0.00015,
            "cached_input": 0.000075,
            "output": 0.0006
        },
        # Claude Models
        "claude-3.5-sonnet": {
            "input": 0.003,
            "cached_input": 0.00375,
            "output": 0.015
        },
        "claude-3.5-haiku": {
            "input": 0.001,
            "cached_input": 0.001,
            "output": 0.005
        },
        "claude-3-opus": {
            "input": 0.015,
            "cached_input": 0.01875,
            "output": 0.075
        }
    }
}

# Mapping of model name variations to their standardized names
MODEL_NAME_MAPPING = {
    # Reasoning models
    "o1": ["o1", "o-1"],
    "o1-mini": ["o1mini", "o1-mini", "o1_mini"],
    "o3-mini": ["o3mini", "o3-mini", "o3_mini"],
    
    # GPT models
    "gpt-4o": ["gpt4o", "gpt-4o", "gpt_4o"],
    "chatgpt-4o-latest": ["chatgpt4o", "chatgpt-4o", "chatgpt_4o"],
    "gpt-4o-mini": ["gpt4omini", "gpt-4o-mini", "gpt_4o_mini"],
    
    # Claude models
    "claude-3.5-sonnet": ["claude-3-5-sonnet", "claude35sonnet", "claude3.5sonnet"],
    "claude-3.5-haiku": ["claude-3-5-haiku", "claude35haiku", "claude3.5haiku"],
    "claude-3-opus": ["claude3opus", "claude-3opus", "claude3-opus"]
}

def get_model_type(model_name: str) -> str:
    """
    Determine if a model is a reasoning or non-reasoning model.
    
    Args:
        model_name: The name of the model to check
        
    Returns:
        'reasoning' or 'non_reasoning'
    """
    # Check reasoning models
    for model in MODEL_COSTS["reasoning"].keys():
        if model in model_name.lower():
            return "reasoning"
            
    # Default to non-reasoning
    return "non_reasoning"

def get_model_costs(model_name: str) -> dict:
    """
    Get the cost configuration for a specific model.
    
    Args:
        model_name: The name of the model to get costs for
        
    Returns:
        Dict containing input, cached_input, and output costs
    """
    model_type = get_model_type(model_name)
    
    # Search through models in the appropriate category
    for model, costs in MODEL_COSTS[model_type].items():
        if model in model_name.lower():
            return costs
            
    # Return zero costs if model not found
    return {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0
    } 