# server.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from ingest_pdf import ingest_pdf_file
from vectorstore import query_top_k
import openai
from utils import OPENAI_API_KEY, CHAT_MODEL

openai.api_key = OPENAI_API_KEY

app = FastAPI(title="RAG PDF Q&A (LMS)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = ingest_pdf_file(file_path, source_name=file.filename)
    return {"message": "uploaded and ingested", "result": result}

@app.post("/query")
async def query_pdf(q: dict):
    """
    Request JSON: { "question": "...", "top_k": 4 }
    """
    question = q.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")
    top_k = q.get("top_k", 4)
    results = query_top_k(question, k=top_k)
    # Extract retrieved texts and metadata
    docs = []
    # results structure depends on chroma version; handle safely:
    try:
        retrieved_texts = results["documents"][0]
        retrieved_ids = results["ids"][0]
        retrieved_mds = results["metadatas"][0]
    except Exception:
        # fallback: return raw results
        return {"error": "unexpected results format", "raw": results}

    for idx, text in enumerate(retrieved_texts):
        docs.append({
            "id": retrieved_ids[idx],
            "text": text,
            "metadata": retrieved_mds[idx]
        })

    # Build prompt for the LLM to answer based only on retrieved docs.
    system_prompt = (
        "You are an assistant that must answer questions ONLY using the provided source excerpts. "
        "If the answer is not found in the excerpts, say you cannot find it in the document."
    )
    context_text = "\n\n".join([f"[{d['id']}] {d['text']}" for d in docs])
    user_prompt = f"Context excerpts:\n{context_text}\n\nQuestion: {question}\n\nAnswer with a short clear response and reference the excerpt ids you used."

    # Call OpenAI ChatCompletion (or completions)
    resp = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=400,
        temperature=0.0,
    )
    answer = resp["choices"][0]["message"]["content"].strip()

    return {"answer": answer, "retrieved": docs}
