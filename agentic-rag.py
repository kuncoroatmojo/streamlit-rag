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
    "https://openreview.net/pdf?id=LzPWWPAdY4",
    "https://openreview.net/pdf?id=VTF8yNQM66",
    "https://openreview.net/pdf?id=hSyW5go0v8",
    "https://openreview.net/pdf?id=9WD9KwssyT",
    "https://openreview.net/pdf?id=yV6fD7LYkF",
    "https://openreview.net/pdf?id=hnrB5YHoYu",
    "https://openreview.net/pdf?id=WbWtOYIzIK",
    "https://openreview.net/pdf?id=c5pwL0Soay",
    "https://openreview.net/pdf?id=TpD2aG1h0D"
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf",
    "swebench.pdf",
    "selfrag.pdf",
    "zipformer.pdf",
    "values.pdf",
    "finetune_fair_diffusion.pdf",
    "knowledge_card.pdf",
    "metra.pdf",
    "vr_mcl.pdf"
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    # Pass the GeminiEmbeddings instance to get_doc_tools
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem, embed_model=gemini_embed_model)  
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
    embed_model=gemini_embed_model  # Pass GeminiEmbedding here
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

tools = obj_retriever.retrieve(
    "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
)


from llama_index.llms.gemini import Gemini

llm = Gemini(
    model="models/gemini-1.5-flash", 
    temperature=0.1, 
    max_output_tokens=100,
    api_key=os.environ["GOOGLE_API_KEY"]
)

from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    tool_retriever=obj_retriever,  # Your list of tools
    llm=llm,
    verbose=True,
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

"""
)

response = agent.query(
    "Tell me about the evaluation dataset used "
    "in MetaGPT and compare it against SWE-Bench"
)
print(str(response))

response = agent.query(
    "Compare and contrast the LoRA papers (LongLoRA, LoftQ). "
    "Analyze the approach in each paper first. "
)

print(str(response))