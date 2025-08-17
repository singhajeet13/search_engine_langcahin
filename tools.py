from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

## Custom tool - RAG
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model import create_HF_embedding_model

from langchain.tools.retriever import create_retriever_tool


## Inbuild Wikkipedia Tool

def create_wiki_tool():
    """Create a Wikipedia tool for querying articles.  """
    api_wrapper_wiki = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=250
    )

    tool_wiki = WikipediaQueryRun(
        api_wrapper=api_wrapper_wiki,
    )

    return tool_wiki

def create_arxiv_tool():
    """Create an Arxiv tool for querying articles."""
    # Create an instance of the ArxivAPIWrapper with desired parameters
    api_wrapper_arxiv = ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=250
    )

    tool_arxiv = ArxivQueryRun(
        api_wrapper=api_wrapper_arxiv,
    )
    return tool_arxiv

def create_web_page_tool(page_url: str):
    """Create a web page tool for querying a specific URL."""

    loader = WebBaseLoader("https://docs.smith.langchain.com/")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embeddings = create_HF_embedding_model()
    vectorstore = faiss.FAISS.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever()

    retriver_tool = create_retriever_tool(
        retriever=retriever,
        name="webpage-search",
        description="Search anything related to the provided webpage url. Use this tool when you need to find information in the webpage url provided",
    )

    return retriver_tool

def create_search_tool():
    """Create a search tool for querying the web."""
    # Create an instance of the DuckDuckGoSearchRun with desired parameters
    search_tool = DuckDuckGoSearchRun(
        max_results=1,
        doc_content_chars_max=250,
        name = "web-search",
        description="Search anything on the web. Use this tool when you need to find information on the web."
    )
    return search_tool

    


