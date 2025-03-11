"""
Standard Knowledge Graph Schema

This module defines a balanced schema for knowledge graphs with a good mix of
essential fields and useful additional features. This is the recommended schema
for most use cases.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict

class Property(BaseModel):
    """
    A standard property representing a key-value pair with metadata.
    
    Includes data type information and confidence score.
    """
    # The property name/key
    key: str
    
    # The property value
    value: str
    
    # The data type of the value (string, number, boolean, date, url)
    # No default value to comply with OpenAI API requirements
    data_type: str
    
    # Confidence score for this property (0-1)
    confidence: Optional[float] = None
    
    # Source of this property information
    source: Optional[str] = None

class Metadata(BaseModel):
    """
    Metadata for nodes, relationships, or the entire graph.
    
    Contains information about creation, sources, and confidence.
    """
    # Creation date in ISO format (YYYY-MM-DD)
    created_at: Optional[str] = None
    
    # Last modification date in ISO format (YYYY-MM-DD)
    last_modified: Optional[str] = None
    
    # Source of the information
    source: Optional[str] = None
    
    # Overall confidence score (0-1)
    confidence: Optional[float] = None
    
    # Method used to extract or create this entity
    extraction_method: Optional[str] = None

class Node(BaseModel):
    """
    A standard node in the knowledge graph representing an entity.
    
    Includes essential fields plus metadata and external references.
    """
    # Unique identifier for the node
    id: str
    
    # List of types/labels for this node (e.g., ['Person', 'Author'])
    labels: List[str]
    
    # Human-readable name for display purposes
    name: str
    
    # Brief description of this entity
    description: Optional[str] = None
    
    # List of properties as key-value pairs
    properties: Optional[List[Property]] = None
    
    # Metadata about this node
    metadata: Optional[Metadata] = None
    
    # IDs in external systems (e.g., {'wikidata': 'Q937'})
    external_ids: Optional[Dict[str, str]] = None

class Relationship(BaseModel):
    """
    A standard relationship between two nodes in the knowledge graph.
    
    Includes direction, weight, and temporal information.
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
    
    # Strength of the relationship (0-1)
    weight: Optional[float] = None
    
    # When this relationship began (YYYY-MM-DD)
    start_date: Optional[str] = None
    
    # When this relationship ended (YYYY-MM-DD)
    end_date: Optional[str] = None
    
    # List of properties as key-value pairs
    properties: Optional[List[Property]] = None
    
    # Metadata about this relationship
    metadata: Optional[Metadata] = None

class KnowledgeGraph(BaseModel):
    """
    A standard knowledge graph with entities, relationships, and metadata.
    
    Includes domain information and versioning.
    """
    # List of nodes in the knowledge graph
    nodes: List[Node]
    
    # List of relationships in the knowledge graph
    relationships: List[Relationship]
    
    # Metadata about the entire graph
    metadata: Optional[Metadata] = None
    
    # Domain or subject area of this knowledge graph
    domain: Optional[str] = None
    
    # Version of this knowledge graph
    version: Optional[str] = None 