import nest_asyncio
nest_asyncio.apply()
#
import os
from llama_index.embeddings.gemini import GeminiEmbedding
gemini_embed_model = GeminiEmbedding()  

GOOGLE_API_KEY = "AIzaSyAV1_kxWDHugmlJtnjI9F6U2AfuHv5Tndg"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem,  embed_model=gemini_embed_model)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.llms.gemini import Gemini

llm = Gemini(
    model="models/gemini-1.5-flash", 
    temperature=0.1, 
    max_output_tokens=100,
    api_key=os.environ["GOOGLE_API_KEY"]
)

from llama_index.core.agent import ReActAgent

# Create the agent with your tools
agent = ReActAgent.from_tools(
    tools=initial_tools,  # Your list of tools
    llm=llm,
    verbose=True,
    max_iterations=10  # Optional: limit the number of iterations
)

response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print(str(response))