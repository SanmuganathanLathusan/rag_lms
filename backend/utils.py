import os
from dotenv import load_dotenv
import nltk


load_dotenv()


# download punkt for sentence tokenization (runs once at import)
try:
nltk.data.find('tokenizers/punkt')
except Exception:
nltk.download('punkt', quiet=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


# sanity check
if not OPENAI_API_KEY:
print("WARNING: OPENAI_API_KEY is not set. Add it to backend/.env or export it.")