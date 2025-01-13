import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.llms.gemini import Gemini
from llama_index.core.agent import ReActAgent
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from utils import get_doc_tools
from pathlib import Path
from llama_index.embeddings.gemini import GeminiEmbedding
gemini_embed_model = GeminiEmbedding()  

# Load environment variables: first try .env file, then Streamlit secrets
load_dotenv()

# Try to get API key from environment first, then from Streamlit secrets
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key

# Set up the Streamlit page
st.set_page_config(page_title="Expert Assistant", page_icon="🤖", layout="wide")

# Sidebar for uploading documents
with st.sidebar:
    st.title("Upload Source Documents")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf", "txt", "docx"])

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist you today?"}]

# Load documents and create tools
if uploaded_files:
    paper_to_tools_dict = {}
    for uploaded_file in uploaded_files:
        paper_name = uploaded_file.name
        with open(paper_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        vector_tool, summary_tool = get_doc_tools(paper_name, Path(paper_name).stem, embed_model=gemini_embed_model)
        paper_to_tools_dict[paper_name] = [vector_tool, summary_tool]

    all_tools = [t for paper in paper_to_tools_dict for t in paper_to_tools_dict[paper]]

    # Create an object index and retriever
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
        embed_model=gemini_embed_model
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)

    # Set up the language model with increased max iterations
    llm = Gemini(
        model="models/gemini-1.5-flash",
        temperature=0.1,
        max_output_tokens=100,
        api_key=os.environ["GOOGLE_API_KEY"]
    )

    # Create the ReActAgent with modified parameters
    agent = ReActAgent.from_tools(
        tool_retriever=obj_retriever,
        llm=llm,
        verbose=True,
        max_iterations=10,  # Increase max iterations (default is usually 5)
        system_prompt=""" \
        You are an agent designed to answer queries over a set of given papers.
        Please always use the tools provided to answer a question. Use prior knowledge or search the internet if you can't find enough information from the given tools.
        If you still cannot find an answer, please state that clearly.
        Try to be concise and direct in your responses.\
        """
    )

    # Create Agent Runner for openai and other llms that support function calling
    #agent_worker = FunctionCallingAgentWorker.from_tools(
    #    tool_retriever=obj_retriever,
    #    llm=llm, 
    #    system_prompt=""" \
    #    You are an agent designed to answer queries over a set of given papers.
    #    Please always use the tools provided to answer a question. Use prior knowledge or search the internet if you can't find enough information from the given tools.
    #    If you still cannot find an answer, please state that clearly.
    #    Try to be concise and direct in your responses.\
    #    """,
    #    verbose=True
    #   )
    #agent = AgentRunner(agent_worker)

# Main section for conversation
st.title("Expert Assistant")
st.caption("Interact with the assistant using the documents you uploaded.")

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# User input for queries
if prompt := st.chat_input("Ask a question related to the uploaded documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if uploaded_files:
        try:
            # Include chat history in the query
            chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            full_prompt = f"{chat_history}\nUser: {prompt}\nAssistant:"
            
            response = agent.query(full_prompt)
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
            st.chat_message("assistant").write(str(response))
            
        except ValueError as e:
            if "Reached max iterations" in str(e):
                error_message = ("I apologize, but I'm having trouble providing a complete answer. "
                               "Could you please rephrase your question or break it down into smaller parts?")
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.chat_message("assistant").write(error_message)
            else:
                raise e
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.chat_message("assistant").write(error_message)
    else:
        st.info("Please upload documents to enable the assistant.") 