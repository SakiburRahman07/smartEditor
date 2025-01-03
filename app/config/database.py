from pymongo import MongoClient
from chromadb import Client as ChromaClient
from chromadb.config import Settings
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB setup
MONGODB_URL = os.getenv("MONGODB_URL")
if not MONGODB_URL:
    raise ValueError("MONGODB_URL not found in environment variables")

mongo_client = MongoClient(MONGODB_URL)
db = mongo_client[os.getenv("MONGO_DB_NAME", "kothakoli")]

# ChromaDB setup
chroma_client = ChromaClient(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_storage"
))

# Create or get collections
chat_collection = db['chat_history']

try:
    chroma_collection = chroma_client.get_collection(name="chat_embeddings")
except:
    chroma_collection = chroma_client.create_collection(
        name="chat_embeddings",
        metadata={"hnsw:space": "cosine"}
    ) 