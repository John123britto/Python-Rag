import streamlit as st
import os
from typing import List
from typing_extensions import TypedDict

# ---------------- FIXED IMPORTS ----------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langchain_community.llms import Ollama

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RAG Agent (Ollama)", layout="wide")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KNOWLEDGE_BASE_DIR = "data"
PERSIST_DIRECTORY = "chroma_db"

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------- INGEST PDFs ----------------
@st.cache_resource
def ingest_pdfs():
    documents = []

    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        return 0

    for file in os.listdir(KNOWLEDGE_BASE_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(KNOWLEDGE_BASE_DIR, file))
            documents.extend(loader.load())

    if not documents:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = splitter.split_documents(documents)

    db = Chroma.from_documents(
        docs,
        embedding=get_embeddings(),
        persist_directory=PERSIST_DIRECTORY
    )

    db.persist()
    return len(documents)

# ---------------- RETRIEVER ----------------
@st.cache_resource
def get_retriever():
    if not os.path.exists(PERSIST_DIRECTORY):
        return None

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embeddings()
    )

    return db.as_retriever(search_kwargs={"k": 2})

# ---------------- GRAPH STATE ----------------
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    answer: str

# ---------------- OLLAMA MODEL ----------------
def get_llm():
    return Ollama(model="mistral")  # use mistral (works on your system)

# ---------------- ROUTER ----------------
def router_node(state: GraphState) -> str:
    question = state["question"]

    prompt = f"""
Decide:
- 'vectorstore' → PDF related
- 'web_search' → latest info

Question: {question}
Answer only one word.
"""

    llm = get_llm()
    decision = llm.invoke(prompt).lower()

    return "web_search" if "web" in decision else "vectorstore"

# ---------------- RETRIEVE ----------------
def retrieve_node(state: GraphState):
    retriever = get_retriever()
    if not retriever:
        return {"documents": []}

    docs = retriever.invoke(state["question"])
    return {"documents": docs}

# ---------------- WEB SEARCH ----------------
def web_search_node(state: GraphState):
    search = TavilySearchResults(max_results=2)
    results = search.invoke({"query": state["question"]})

    docs = []
    for r in results:
        loader = WebBaseLoader(r["url"])
        docs.extend(loader.load())

    return {"documents": docs}

# ---------------- GENERATE ----------------
def generate_node(state: GraphState):
    question = state["question"]
    docs = state["documents"]

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer based on context.

Context:
{context}

Question:
{question}

If not found say "I don't know".
"""

    llm = get_llm()
    answer = llm.invoke(prompt)

    return {"answer": answer}

# ---------------- GRAPH ----------------
@st.cache_resource
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web", web_search_node)
    workflow.add_node("generate", generate_node)

    workflow.add_conditional_edges(
        START,
        router_node,
        {
            "vectorstore": "retrieve",
            "web_search": "web"
        }
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# ---------------- UI ----------------
def main():
    st.title("🤖 RAG Agent with Ollama")

    if st.button("📥 Ingest PDFs"):
        count = ingest_pdfs()
        st.success(f"Ingested {count} documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if question := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                app = build_graph()
                result = app.invoke({"question": question})

                answer = result["answer"]
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

if __name__ == "__main__":
    main()