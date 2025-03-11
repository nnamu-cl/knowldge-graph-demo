"""
Knowledge Graph Generator

This module generates knowledge graphs using OpenAI's structured output feature.
It uses schemas of varying complexity from the schemas package.
"""

from typing import Optional
from openai import OpenAI
import json
from datetime import datetime

# Import the schema provider
from schemas import GetKnowledgeGraphSchema, ComplexityLevel, GetSchemaDescription
# Import graph utilities
from graph_utils import dict_to_graph_documents, upload_kg_to_neo4j, print_graph_document_summary

# Initialize the OpenAI client
client = OpenAI()

def generate_knowledge_graph(topic: str, complexity: ComplexityLevel = "standard") -> dict:
    """
    Generate a knowledge graph on the given topic using OpenAI's structured output.
    
    Args:
        topic: The topic to generate a knowledge graph about
        complexity: The complexity level of the schema ("basic", "standard", or "advanced")
        
    Returns:
        A knowledge graph as a dictionary
    """
    # Get the appropriate schema
    KnowledgeGraph = GetKnowledgeGraphSchema(complexity)
    
    # Get a description of the schema for the system prompt
    schema_description = GetSchemaDescription(complexity)
    
    print(f"Generating {complexity} knowledge graph about: {topic}")
    
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system", 
                    "content": f"""
                    You are an expert at knowledge graph generation. 
                    Create a detailed knowledge graph on the given topic.
                    
                    {schema_description}
                    
                    Follow these guidelines:
                    - Each node should have a unique ID, a name, and at least one label/type
                    - Provide a brief description for each node
                    - Nodes can have optional properties (key-value pairs)
                    - For properties, always specify a data_type as one of: "string", "number", "boolean", "date", "url"
                    - Relationships connect two nodes with a specific type (e.g., "CREATED", "LOCATED_IN", "INFLUENCED")
                    - For relationships, specify source, target, type, and set bidirectional to true or false
                    - Add weight values (0-1) to relationships to indicate their strength
                    - Add temporal information where relevant (when relationships began/ended)
                    - Make the graph rich and interconnected
                    - Include metadata like confidence scores where appropriate
                    """
                },
                {
                    "role": "user", 
                    "content": f"Generate a knowledge graph about: {topic}"
                }
            ],
            response_format=KnowledgeGraph,
        )
        
        # Convert to dict for easier manipulation
        kg = completion.choices[0].message.parsed.model_dump(mode='json')
        
        # Add metadata
        if complexity != "basic" and "metadata" in kg:
            kg["metadata"] = {
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "source": "OpenAI GPT-4o",
                "extraction_method": "LLM-generated"
            }
        
        kg["domain"] = topic
        kg["version"] = "1.0"
        
        return kg
        
    except Exception as e:
        print(f"Error generating knowledge graph: {e}")
        raise

def print_knowledge_graph(kg: dict):
    """
    Print a knowledge graph in a readable format.
    
    Args:
        kg: The knowledge graph as a dictionary
    """
    print("=== KNOWLEDGE GRAPH ===")
    if "domain" in kg:
        print(f"Domain: {kg['domain']}")
    if "version" in kg:
        print(f"Version: {kg['version']}")
    
    print(f"\n=== NODES ({len(kg['nodes'])}) ===")
    for node in kg['nodes']:
        print(f"Node: {node['name']} (ID: {node['id']})")
        print(f"  Labels: {', '.join(node['labels'])}")
        if "description" in node and node["description"]:
            print(f"  Description: {node['description']}")
        if "properties" in node and node["properties"]:
            print("  Properties:")
            for prop in node["properties"]:
                confidence_str = f" (confidence: {prop['confidence']})" if "confidence" in prop and prop["confidence"] is not None else ""
                print(f"    {prop['key']}: {prop['value']} ({prop['data_type']}){confidence_str}")
        if "external_ids" in node and node["external_ids"]:
            print("  External IDs:")
            for system, ext_id in node["external_ids"].items():
                print(f"    {system}: {ext_id}")
        print()
    
    print(f"=== RELATIONSHIPS ({len(kg['relationships'])}) ===")
    for rel in kg['relationships']:
        direction = "<-->" if rel['bidirectional'] else "-->"
        weight_str = f" [weight: {rel['weight']}]" if "weight" in rel and rel["weight"] is not None else ""
        print(f"Relationship: {rel['source']} {direction}[{rel['type']}]{weight_str} {rel['target']}")
        
        temporal = []
        if "start_date" in rel and rel["start_date"]:
            temporal.append(f"started: {rel['start_date']}")
        if "end_date" in rel and rel["end_date"]:
            temporal.append(f"ended: {rel['end_date']}")
        if "duration" in rel and rel["duration"]:
            temporal.append(f"duration: {rel['duration']}")
        
        if temporal:
            print(f"  Temporal: {', '.join(temporal)}")
            
        if "properties" in rel and rel["properties"]:
            print("  Properties:")
            for prop in rel["properties"]:
                confidence_str = f" (confidence: {prop['confidence']})" if "confidence" in prop and prop["confidence"] is not None else ""
                print(f"    {prop['key']}: {prop['value']} ({prop['data_type']}){confidence_str}")
        print()

def save_knowledge_graph(kg: dict, filename: str = "knowledge_graph.json"):
    """
    Save a knowledge graph to a JSON file.
    
    Args:
        kg: The knowledge graph as a dictionary
        filename: The filename to save to
    """
    with open(filename, "w") as f:
        json.dump(kg, f, indent=2)
    
    print(f"\nKnowledge graph saved to {filename}")

def upload_to_neo4j(kg: dict, neo4j_url: str, neo4j_username: str, neo4j_password: str) -> bool:
    """
    Upload a knowledge graph to Neo4j.
    
    Args:
        kg: The knowledge graph as a dictionary
        neo4j_url: The Neo4j database URL
        neo4j_username: The Neo4j username
        neo4j_password: The Neo4j password
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from langchain_community.graphs import Neo4jGraph
        
        # Initialize Neo4j graph
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Convert dictionary to GraphDocument and upload
        return upload_kg_to_neo4j(kg, graph)
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    topic = "a simple family of 3"
    
    # Generate a knowledge graph
    complexity = "basic"  # Choose from: "basic", "standard", "advanced"
    kg = generate_knowledge_graph(topic, complexity)
    
    # Print the knowledge graph
    print_knowledge_graph(kg)
    
    # Save the knowledge graph to a JSON file
    save_knowledge_graph(kg, f"{complexity}_knowledge_graph.json")
    
    # Convert to LangChain GraphDocument format
    graph_documents = dict_to_graph_documents(kg)
    
    # Print summary of the GraphDocument
    print("\nGraph Document Summary:")
    print_graph_document_summary(graph_documents)
    
    # Optionally upload to Neo4j
    # Uncomment and provide your Neo4j credentials to upload

    neo4j_url = "bolt+s://your-neo4j-instance.databases.neo4j.io:7687"
    neo4j_username = "neo4j"
    neo4j_password = "your-password"
    
    if upload_to_neo4j(kg, neo4j_url, neo4j_username, neo4j_password):
        print("Successfully uploaded to Neo4j!")
    else:
        print("Failed to upload to Neo4j.")
    
            
        