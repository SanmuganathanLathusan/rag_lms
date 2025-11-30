# vectorstore.py
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
from typing import List, Dict
from utils import CHROMA_PERSIST_DIR, EMBED_MODEL, OPENAI_API_KEY
import openai

# configure OpenAI key for chroma embedding function (if using OpenAI)
openai.api_key = OPENAI_API_KEY

def get_client():
    # persistent directory
    persist = CHROMA_PERSIST_DIR
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist))
    return client

def create_collection(collection_name="pdf_docs"):
    client = get_client()
    if collection_name in [c.name for c in client.list_collections()]:
        return client.get_collection(collection_name)
    # use OpenAI embeddings via chromadb helper
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)
    return client.create_collection(name=collection_name, embedding_function=openai_ef)

def add_documents(docs: List[Dict], collection_name="pdf_docs"):
    """
    docs: list of dicts with keys: id, text, metadata (dict)
    """
    coll = create_collection(collection_name)
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]
    coll.add(documents=texts, metadatas=metadatas, ids=ids)

def query_top_k(query: str, k=4, collection_name="pdf_docs"):
    coll = create_collection(collection_name)
    results = coll.query(query_texts=[query], n_results=k)
    # results: dict with ids, distances, documents, metadatas
    return results
