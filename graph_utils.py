"""
Graph Utilities

This module provides utility functions for working with knowledge graphs,
including conversion between different formats and database operations.
"""

from typing import Dict, List, Any, Optional
from langchain_community.graphs.graph_document import (
    Node as LCNode,
    Relationship as LCRelationship,
    GraphDocument
)

def dict_to_graph_documents(kg_dict: Dict[str, Any]) -> List[GraphDocument]:
    """
    Convert a dictionary-based knowledge graph to a list of LangChain GraphDocument objects.
    
    Args:
        kg_dict: A dictionary containing 'nodes' and 'relationships' keys
        
    Returns:
        A list containing a single GraphDocument object that can be used with 
        graph.add_graph_documents()
    """
    # Convert dictionary nodes to LangChain Node objects
    lc_nodes = []
    for node_dict in kg_dict.get('nodes', []):
        # Extract node properties
        node_id = node_dict.get('id')
        node_type = None
        
        # Handle different schema formats for node type
        if 'type' in node_dict:
            node_type = node_dict.get('type')
        elif 'labels' in node_dict and node_dict['labels']:
            # If we have labels list, use the first one as the type
            node_type = node_dict['labels'][0]
            
        # Skip nodes without ID or type
        if not node_id or not node_type:
            continue
            
        # Extract properties
        properties = {}
        
        # Add name if available
        if 'name' in node_dict:
            properties['name'] = node_dict['name']
            
        # Add description if available
        if 'description' in node_dict and node_dict['description']:
            properties['description'] = node_dict['description']
            
        # Add any custom properties
        if 'properties' in node_dict and node_dict['properties']:
            for prop in node_dict['properties']:
                if 'key' in prop and 'value' in prop:
                    properties[prop['key']] = prop['value']
        
        # Create LangChain Node
        lc_node = LCNode(
            id=node_id,
            type=node_type,
            properties=properties if properties else None
        )
        lc_nodes.append(lc_node)
    
    # Convert dictionary relationships to LangChain Relationship objects
    lc_relationships = []
    for rel_dict in kg_dict.get('relationships', []):
        # Extract relationship properties
        source_id = rel_dict.get('source')
        target_id = rel_dict.get('target')
        rel_type = rel_dict.get('type')
        
        # Skip relationships without source, target, or type
        if not source_id or not target_id or not rel_type:
            continue
            
        # Find source and target nodes
        source_node = next((node for node in lc_nodes if node.id == source_id), None)
        target_node = next((node for node in lc_nodes if node.id == target_id), None)
        
        # Skip if source or target node not found
        if not source_node or not target_node:
            continue
            
        # Extract properties
        properties = {}
        
        # Add weight if available
        if 'weight' in rel_dict and rel_dict['weight'] is not None:
            properties['weight'] = rel_dict['weight']
            
        # Add temporal information if available
        if 'start_date' in rel_dict and rel_dict['start_date']:
            properties['start_date'] = rel_dict['start_date']
        if 'end_date' in rel_dict and rel_dict['end_date']:
            properties['end_date'] = rel_dict['end_date']
        if 'duration' in rel_dict and rel_dict['duration']:
            properties['duration'] = rel_dict['duration']
            
        # Add any custom properties
        if 'properties' in rel_dict and rel_dict['properties']:
            for prop in rel_dict['properties']:
                if 'key' in prop and 'value' in prop:
                    properties[prop['key']] = prop['value']
        
        # Create LangChain Relationship
        lc_relationship = LCRelationship(
            source=source_node,
            target=target_node,
            type=rel_type,
            properties=properties if properties else {}
        )
        lc_relationships.append(lc_relationship)
    
    # Create a source dictionary with information about the knowledge graph
    source_dict = {
        "type": "knowledge_graph",
        "id": kg_dict.get("domain", "unknown_domain"),
        "version": kg_dict.get("version", "1.0"),
    }
    
    # Add metadata if available
    if "metadata" in kg_dict and kg_dict["metadata"]:
        if "source" in kg_dict["metadata"]:
            source_dict["origin"] = kg_dict["metadata"]["source"]
        if "created_at" in kg_dict["metadata"]:
            source_dict["created_at"] = kg_dict["metadata"]["created_at"]
    
    # Create GraphDocument with source as a dictionary
    graph_document = GraphDocument(
        nodes=lc_nodes,
        relationships=lc_relationships,
        # source=source_dict  # Source must be a dictionary
    )
    
    return [graph_document]

def upload_kg_to_neo4j(kg_dict: Dict[str, Any], graph) -> bool:
    """
    Upload a dictionary-based knowledge graph to Neo4j.
    
    Args:
        kg_dict: A dictionary containing 'nodes' and 'relationships' keys
        graph: A Neo4jGraph instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert dictionary to GraphDocument
        graph_documents = dict_to_graph_documents(kg_dict)
        
        # Upload to Neo4j
        graph.add_graph_documents(graph_documents, include_source=False)
        return True
    except Exception as e:
        print(f"Error uploading knowledge graph to Neo4j: {e}")
        return False

def print_graph_document_summary(graph_documents: List[GraphDocument]) -> None:
    """
    Print a summary of the GraphDocument objects.
    
    Args:
        graph_documents: A list of GraphDocument objects
    """
    for i, doc in enumerate(graph_documents):
        print(f"Graph Document #{i+1}:")
        print(f"  Source: {doc.source}")
        print(f"  Nodes: {len(doc.nodes)}")
        print("  Node Types:")
        node_types = {}
        for node in doc.nodes:
            node_types[node.type] = node_types.get(node.type, 0) + 1
        for node_type, count in node_types.items():
            print(f"    - {node_type}: {count}")
            
        print(f"  Relationships: {len(doc.relationships)}")
        rel_types = {}
        for rel in doc.relationships:
            rel_types[rel.type] = rel_types.get(rel.type, 0) + 1
        print("  Relationship Types:")
        for rel_type, count in rel_types.items():
            print(f"    - {rel_type}: {count}")
        print() 