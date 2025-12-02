import pdfplumber
import uuid
import os
from typing import List, Dict
from nltk.tokenize import sent_tokenize
from vectorstore import add_documents
from tqdm import tqdm


CHUNK_SIZE = 800 # characters per chunk (tuneable)
CHUNK_OVERLAP = 120




def extract_text_from_pdf(path: str) -> str:
text = []
with pdfplumber.open(path) as pdf:
for page in pdf.pages:
t = page.extract_text()
if t:
text.append(t)
return "\n".join(text)




def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
sentences = sent_tokenize(text)
chunks = []
current = ""
for s in sentences:
if len(current) + len(s) + 1 <= chunk_size:
current = current + " " + s if current else s
else:
chunks.append(current.strip())
# carry overlap chars to new chunk
if overlap > 0:
carry = current[-overlap:] if len(current) >= overlap else current
current = (carry + " " + s).strip()
else:
current = s
if current:
chunks.append(current.strip())
return chunks




def ingest_pdf_file(path: str, source_name: str = None):
source_name = source_name or os.path.basename(path)
text = extract_text_from_pdf(path)
if not text.strip():
return {"status": "empty", "source": source_name, "n_chunks": 0}


chunks = chunk_text(text)
docs = []
for i, c in enumerate(tqdm(chunks, desc="Preparing chunks")):
docs.append({
"id": f"{source_name}_{i}_{uuid.uuid4().hex[:8]}",
"text": c,
"metadata": {"source": source_name, "chunk_index": i}
})
add_documents(docs)
return {"status": "ok", "source": source_name, "n_chunks": len(docs)}