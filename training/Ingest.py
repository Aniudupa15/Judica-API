import os
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Step 1: Load PDF Documents from a Directory
data_dir = "data"
if not os.path.exists(data_dir):
    logging.error(f"Directory '{data_dir}' does not exist. Please create it and add PDF files.")
    exit()

try:
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents from the '{data_dir}' directory.")
except Exception as e:
    logging.error(f"Error loading documents: {e}")
    exit()

# Step 2: Split Documents into Manageable Text Chunks
try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    logging.info(f"Split the documents into {len(texts)} text chunks.")
except Exception as e:
    logging.error(f"Error splitting documents: {e}")
    exit()

# Step 3: Initialize HuggingFace Embeddings Model
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"}
    )
    logging.info("Initialized HuggingFace embeddings model successfully.")
except Exception as e:
    logging.error(f"Error initializing embeddings model: {e}")
    exit()

# Step 4: Create FAISS Vector Database from Text Embeddings
try:
    faiss_db = FAISS.from_documents(texts, embeddings)
    logging.info("Created FAISS vector database from text embeddings.")
except Exception as e:
    logging.error(f"Error creating FAISS vector database: {e}")
    exit()

# Step 5: Save the FAISS Vector Database Locally
try:
    save_path = "ipc_vector_db"
    faiss_db.save_local(save_path)
    logging.info(f"FAISS vector database has been saved locally at '{save_path}'.")
except Exception as e:
    logging.error(f"Error saving FAISS vector database: {e}")
    exit()