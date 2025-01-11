# TODO: abstract all of this into a function that takes in a PDF file name 

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter, SimpleNodeParser
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
import os

from typing import List, Optional


# def get_doc_tools(
#     file_path: str,
#     name: str,
# ) -> str:
#     """Get vector query and summary query tools from a document."""
    
#     # load documents
#     documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
#     splitter = SentenceSplitter(chunk_size=1024)
#     nodes = splitter.get_nodes_from_documents(documents)
#     vector_index = VectorStoreIndex(nodes)

#     def vector_query(
#         query: str, 
#         filter_key_list: List[str],
#         filter_value_list: List[str]
#     ) -> str:
#         """Perform a vector search over an index.

#         query (str): the string query to be embedded.
#         filter_key_list (List[str]): A list of metadata filter field names
#             Must specify ['page_label'] or empty list. Please leave empty
#             if there are no explicit filters to specify.
#         filter_value_list (List[str]): List of metadata filter field values 
#             (corresponding to names specified in filter_key_list) 

#         """
#         metadata_dicts = [
#             {"key": k, "value": v} for k, v in zip(filter_key_list, filter_value_list)
#         ]

#         query_engine = vector_index.as_query_engine(
#             similarity_top_k=2,
#             filters=MetadataFilters.from_dicts(metadata_dicts)
#         )
#         response = query_engine.query(query)
#         return response

#     vector_query_tool = FunctionTool.from_defaults(
#         fn=vector_query,
#         name=f"vector_query_{name}"
#     )

#     summary_index = SummaryIndex(nodes)
#     summary_query_engine = summary_index.as_query_engine(
#         response_mode="tree_summarize",
#         use_async=True,
#     )
#     summary_tool = QueryEngineTool.from_defaults(
#         query_engine=summary_query_engine,
#         name=f"summary_query_{name}",
#         description=(
#             f"Useful for summarization questions related to {name}"
#         ),
#     )
#     return vector_query_tool, summary_tool



def get_doc_tools(
    file_path: str,
    name: str,
    embed_model=None
) -> str:
    """Get vector query and summary query tools from a document.
    
    Args:
        file_path (str): Path to the document
        name (str): Name identifier for the tools
        embed_model: Optional embedding model (defaults to OpenAI if None)
    """
    
    # Load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    # Parse documents into nodes
    nodes = SimpleNodeParser().get_nodes_from_documents(documents)
    
    # Clean nodes with better error handling
    for node in nodes:
        try:
            node.text = node.text.encode('utf-8', 'replace').decode('utf-8')
        except Exception as e:
            node.text = ''.join(char for char in node.text if ord(char) < 128)
    
    # Create a Gemini LLM instance
    llm = Gemini(
        model="models/gemini-1.5-flash", 
        temperature=0.1, 
        api_key=os.environ["GOOGLE_API_KEY"]
    )
    Settings.llm = llm
    
    # Use provided embed_model or default to OpenAI
    if embed_model is None:
        embed_model = OpenAIEmbedding()
    
    # Initialize indices with cleaned nodes and embedding model
    vector_index = VectorStoreIndex(
        nodes,
        embed_model=embed_model
    )
    summary_index = SummaryIndex(
        nodes,
        embed_model=embed_model
    )
    
    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over a given paper.
    
        Useful if you have specific questions over the paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
    
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        # More robust Unicode handling
        try:
            # First attempt: direct string conversion
            return str(response)
        except UnicodeEncodeError:
            try:
                # Second attempt: normalize Unicode and remove problematic characters
                import unicodedata
                normalized = unicodedata.normalize('NFKD', str(response))
                return normalized.encode('ascii', 'ignore').decode('ascii')
            except:
                # Fallback: aggressive character replacement
                return str(response).encode('ascii', 'replace').decode('ascii')
        
    
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {name}"
        ),
    )

    return vector_query_tool, summary_tool