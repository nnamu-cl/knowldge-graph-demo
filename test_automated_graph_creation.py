"""
Test Automated Graph Creation

This script demonstrates how to construct a knowledge graph from unstructured text
using LangChain's LLMGraphTransformer and Neo4j.

This is a direct implementation of the example from LangChain's documentation:
https://python.langchain.com/v0.1/docs/use_cases/graph/constructing_knowledge_graphs/
"""

import os
import getpass
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph

# Set up OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Set up Neo4j credentials
url = "neo4j+s://8e6af5fb.databases.neo4j.io"
username = "neo4j"
password = "6Q7v8t1FcCKs6UkuOeUk83v2GVKqw8xZPvMk6g0hYCg"

# Initialize Neo4j graph
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

# Initialize LLM
llm = ChatOpenAI(temperature=1, model_name="gpt-4o")

# Initialize LLMGraphTransformer
llm_transformer = LLMGraphTransformer(llm=llm)

# Example text
text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""

# Convert text to document
documents = [Document(page_content=text)]

# Convert document to graph document
print("Basic Graph Extraction:")
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes: {graph_documents[0].nodes}")
print(f"Relationships: {graph_documents[0].relationships}")
print("\n" + "-"*80 + "\n")

# Filtered graph extraction
print("Filtered Graph Extraction:")
llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Country", "Organization"],
    allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
)
graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)
print(f"Nodes: {graph_documents_filtered[0].nodes}")
print(f"Relationships: {graph_documents_filtered[0].relationships}")
print("\n" + "-"*80 + "\n")

# Graph extraction with node properties
print("Graph Extraction with Node Properties:")
llm_transformer_props = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Country", "Organization"],
    allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
    node_properties=["born_year"],
)
graph_documents_props = llm_transformer_props.convert_to_graph_documents(documents)
print(f"Nodes: {graph_documents_props[0].nodes}")
print(f"Relationships: {graph_documents_props[0].relationships}")
print("\n" + "-"*80 + "\n")

# Store to graph database
print("Storing to Neo4j Graph Database...")
try:
    graph.add_graph_documents(graph_documents_props)
    print("Successfully stored graph documents to Neo4j!")
except Exception as e:
    print(f"Error storing graph documents to Neo4j: {e}")
    print("Make sure your Neo4j database is running and credentials are correct.")

print("\nTest completed!") 