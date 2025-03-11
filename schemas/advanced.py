"""
Advanced Knowledge Graph Schema

This module defines a comprehensive schema for knowledge graphs with all possible
fields and features. This schema is suitable for complex knowledge graph applications
that require rich metadata, visualization properties, and detailed attributes.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Property(BaseModel):
    """
    An advanced property representing a key-value pair with comprehensive metadata.
    
    Includes data type information, confidence score, source, and validation.
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
    
    # Whether this property has been validated
    validated: Optional[bool] = None
    
    # Who validated this property
    validated_by: Optional[str] = None
    
    # When this property was last updated (YYYY-MM-DD)
    last_updated: Optional[str] = None

class Metadata(BaseModel):
    """
    Comprehensive metadata for nodes, relationships, or the entire graph.
    
    Contains detailed information about creation, sources, confidence, and more.
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
    
    # Additional notes about this entity
    notes: Optional[str] = None
    
    # Creator of this entity
    created_by: Optional[str] = None
    
    # Last person to modify this entity
    modified_by: Optional[str] = None
    
    # Tags for categorization
    tags: Optional[List[str]] = None
    
    # Verification status
    verification_status: Optional[str] = None

class Node(BaseModel):
    """
    An advanced node in the knowledge graph representing an entity.
    
    Includes all possible fields for rich entity representation.
    """
    # Unique identifier for the node
    id: str
    
    # List of types/labels for this node (e.g., ['Person', 'Author'])
    labels: List[str]
    
    # Human-readable name for display purposes
    name: str
    
    # Brief description of this entity
    description: Optional[str] = None
    
    # Detailed long-form description
    detailed_description: Optional[str] = None
    
    # List of properties as key-value pairs
    properties: Optional[List[Property]] = None
    
    # Additional structured attributes for this node
    attributes: Optional[Dict[str, str]] = None
    
    # Metadata about this node
    metadata: Optional[Metadata] = None
    
    # IDs in external systems (e.g., {'wikidata': 'Q937'})
    external_ids: Optional[Dict[str, str]] = None
    
    # Properties for visualization (e.g., color, size, position)
    display_properties: Optional[Dict[str, str]] = None
    
    # URLs to images representing this node
    images: Optional[List[str]] = None
    
    # URLs to related resources
    urls: Optional[List[str]] = None
    
    # Hierarchical position information
    hierarchy: Optional[Dict[str, str]] = None
    
    # Geographic coordinates if applicable
    geo_coordinates: Optional[Dict[str, float]] = None

class Relationship(BaseModel):
    """
    An advanced relationship between two nodes in the knowledge graph.
    
    Includes comprehensive metadata, temporal information, and attributes.
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
    
    # Duration of the relationship
    duration: Optional[str] = None
    
    # List of properties as key-value pairs
    properties: Optional[List[Property]] = None
    
    # Additional structured attributes for this relationship
    attributes: Optional[Dict[str, str]] = None
    
    # Metadata about this relationship
    metadata: Optional[Metadata] = None
    
    # Properties for visualization (e.g., color, thickness)
    display_properties: Optional[Dict[str, str]] = None
    
    # Qualifiers that provide context to the relationship
    qualifiers: Optional[Dict[str, str]] = None
    
    # Provenance information about this relationship
    provenance: Optional[Dict[str, str]] = None
    
    # Certainty level about this relationship (beyond confidence)
    certainty: Optional[str] = None

class KnowledgeGraph(BaseModel):
    """
    An advanced knowledge graph with comprehensive features.
    
    Includes detailed metadata, statistics, and configuration options.
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
    
    # Statistics about this graph (e.g., node count by type)
    stats: Optional[Dict[str, str]] = None
    
    # Configuration settings for this graph
    config: Optional[Dict[str, str]] = None
    
    # Namespace definitions
    namespaces: Optional[Dict[str, str]] = None
    
    # Ontology information
    ontology: Optional[Dict[str, Any]] = None
    
    # License information
    license: Optional[str] = None
    
    # Citation information
    citation: Optional[str] = None 