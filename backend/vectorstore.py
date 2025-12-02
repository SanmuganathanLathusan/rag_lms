from cfhromadb.config import Settings
import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict
from utils import CHROMA_PERSIST_DIR, EMBED_MODEL, OPENAI_API_KEY


# create Chroma client with persistent storage
def get_client():
persist = CHROMA_PERSIST_DIR
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist))
return client




def create_collection(collection_name: str = "pdf_docs"):
client = get_client()
# if collection exists, return it
names = [c.name for c in client.list_collections()]
if collection_name in names:
return client.get_collection(collection_name)
# use OpenAI embeddings via chromadb helper
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)
coll = client.create_collection(name=collection_name, embedding_function=openai_ef)
return coll




def add_documents(docs: List[Dict], collection_name: str = "pdf_docs"):
coll = create_collection(collection_name)
ids = [d["id"] for d in docs]
texts = [d["text"] for d in docs]
metadatas = [d.get("metadata", {}) for d in docs]
coll.add(documents=texts, metadatas=metadatas, ids=ids)




def query_top_k(query: str, k: int = 4, collection_name: str = "pdf_docs"):
coll = create_collection(collection_name)
results = coll.query(query_texts=[query], n_results=k)
return results