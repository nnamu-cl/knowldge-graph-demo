"""
Basic Knowledge Graph Schema

This module defines the minimal schema required for a functional knowledge graph.
It includes only the essential fields needed to represent nodes, relationships,
and their connections.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict

class Property(BaseModel):
    """
    A basic property representing a key-value pair with data type.
    
    This is the simplest form of property that can be attached to nodes or relationships.
    """
    # The property name/key
    key: str
    
    # The property value
    value: str
    
    # The data type of the value (string, number, boolean, date, url)
    # No default value to comply with OpenAI API requirements
    data_type: str

class Node(BaseModel):
    """
    A basic node in the knowledge graph representing an entity.
    
    Contains only the essential fields needed to identify and describe an entity.
    """
    # Unique identifier for the node
    id: str
    
    # List of types/labels for this node (e.g., ['Person', 'Author'])
    labels: List[str]
    
    # Human-readable name for display purposes
    name: str
    
    # Brief description of this entity
    description: Optional[str] = None
    
    # Optional list of properties as key-value pairs
    properties: Optional[List[Property]] = None

class Relationship(BaseModel):
    """
    A basic relationship between two nodes in the knowledge graph.
    
    Represents the simplest form of connection between two nodes.
    """
    # ID of the source node
    source: str
    
    # ID of the target node
    target: str
    
    # Type of relationship (e.g., 'CREATED', 'KNOWS')
    type: str
    
    # Whether this relationship applies in both directions
    # No default value to comply with OpenAI API requirements
    bidirectional: bool

class KnowledgeGraph(BaseModel):
    """
    A basic knowledge graph with entities and relationships.
    
    The minimal structure needed to represent a knowledge graph.
    """
    # List of nodes in the knowledge graph
    nodes: List[Node]
    
    # List of relationships in the knowledge graph
    relationships: List[Relationship] 