from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pypdf import PdfReader
import tempfile
import os

load_dotenv()



app = FastAPI()

# Allow your frontend origin
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


VECTOR_STORE = None


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF"""
    global VECTOR_STORE

    # Validate PDF
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files allowed"}

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    # Extract text from PDF
    text = ""
    reader = PdfReader(temp_path)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    if text.strip() == "":
        return {"error": "No extractable text in PDF"}

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # Embed + store in FAISS
    embeddings = OpenAIEmbeddings()
    VECTOR_STORE = FAISS.from_texts(chunks, embeddings)

    return {"message": "PDF processed successfully", "chunks": len(chunks)}


from pydantic import BaseModel

class AskModel(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(data: AskModel):
    global VECTOR_STORE

    question = data.question

    if VECTOR_STORE is None:
        return {"error": "No PDF uploaded yet"}

    docs = VECTOR_STORE.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke(
        f"Answer based on the document only:\n\nContext:\n{context}\n\nQuestion: {question}"
    )

    return { "answer": response.content }
