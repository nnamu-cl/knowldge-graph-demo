"""
Knowledge Graph Schema Provider

This module provides a utility function to get knowledge graph schemas of varying complexity.
It serves as a central access point for all schema definitions.
"""

from typing import Tuple, Type, Literal
from pydantic import BaseModel

# Import all schema definitions
from schemas.basic import Property as BasicProperty
from schemas.basic import Node as BasicNode
from schemas.basic import Relationship as BasicRelationship
from schemas.basic import KnowledgeGraph as BasicKnowledgeGraph

from schemas.standard import Property as StandardProperty
from schemas.standard import Node as StandardNode
from schemas.standard import Relationship as StandardRelationship
from schemas.standard import KnowledgeGraph as StandardKnowledgeGraph
from schemas.standard import Metadata as StandardMetadata

from schemas.advanced import Property as AdvancedProperty
from schemas.advanced import Node as AdvancedNode
from schemas.advanced import Relationship as AdvancedRelationship
from schemas.advanced import KnowledgeGraph as AdvancedKnowledgeGraph
from schemas.advanced import Metadata as AdvancedMetadata

# Define a type for complexity levels
ComplexityLevel = Literal["basic", "standard", "advanced"]

def GetKnowledgeGraphSchema(complexity: ComplexityLevel = "standard") -> Type[BaseModel]:
    """
    Get a knowledge graph schema of the specified complexity.
    
    Args:
        complexity: The complexity level of the schema ("basic", "standard", or "advanced")
        
    Returns:
        The KnowledgeGraph Pydantic model class for the specified complexity level
        
    Raises:
        ValueError: If an invalid complexity level is provided
    """
    if complexity == "basic":
        return BasicKnowledgeGraph
    elif complexity == "standard":
        return StandardKnowledgeGraph
    elif complexity == "advanced":
        return AdvancedKnowledgeGraph
    else:
        raise ValueError(f"Invalid complexity level: {complexity}. Must be one of: basic, standard, advanced")

def GetAllSchemaClasses(complexity: ComplexityLevel = "standard") -> Tuple:
    """
    Get all schema classes (Property, Node, Relationship, KnowledgeGraph) for the specified complexity.
    
    Args:
        complexity: The complexity level of the schema ("basic", "standard", or "advanced")
        
    Returns:
        A tuple of (Property, Node, Relationship, KnowledgeGraph) classes
        
    Raises:
        ValueError: If an invalid complexity level is provided
    """
    if complexity == "basic":
        return (BasicProperty, BasicNode, BasicRelationship, BasicKnowledgeGraph)
    elif complexity == "standard":
        return (StandardProperty, StandardNode, StandardRelationship, StandardKnowledgeGraph, StandardMetadata)
    elif complexity == "advanced":
        return (AdvancedProperty, AdvancedNode, AdvancedRelationship, AdvancedKnowledgeGraph, AdvancedMetadata)
    else:
        raise ValueError(f"Invalid complexity level: {complexity}. Must be one of: basic, standard, advanced")

def GetSchemaDescription(complexity: ComplexityLevel = "standard") -> str:
    """
    Get a description of the schema at the specified complexity level.
    
    Args:
        complexity: The complexity level of the schema ("basic", "standard", or "advanced")
        
    Returns:
        A string describing the schema
        
    Raises:
        ValueError: If an invalid complexity level is provided
    """
    if complexity == "basic":
        return """
        Basic Knowledge Graph Schema
        
        A minimal schema with only essential fields:
        - Nodes with ID, labels, name, and optional description and properties
        - Relationships with source, target, type, and bidirectional flag
        - Properties with key, value, and data type
        
        Suitable for simple knowledge graphs with minimal metadata requirements.
        """
    elif complexity == "standard":
        return """
        Standard Knowledge Graph Schema
        
        A balanced schema with essential fields plus useful metadata:
        - Nodes with ID, labels, name, description, properties, metadata, and external IDs
        - Relationships with source, target, type, bidirectional flag, weight, and temporal information
        - Properties with key, value, data type, confidence, and source
        - Metadata with creation/modification dates, source, confidence, and extraction method
        
        Suitable for most knowledge graph applications.
        """
    elif complexity == "advanced":
        return """
        Advanced Knowledge Graph Schema
        
        A comprehensive schema with all possible fields:
        - Nodes with extensive metadata, visualization properties, and detailed attributes
        - Relationships with comprehensive metadata, temporal information, and qualifiers
        - Properties with validation information and detailed metadata
        - Graph-level configuration, statistics, namespaces, and ontology information
        
        Suitable for complex knowledge graph applications requiring rich metadata.
        """
    else:
        raise ValueError(f"Invalid complexity level: {complexity}. Must be one of: basic, standard, advanced") 