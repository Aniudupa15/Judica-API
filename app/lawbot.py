from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Set the API key for Together.ai
TOGETHER_AI_API = os.getenv("TOGETHER_AI_API", "1c27fe0df51a29edee1bec6b4b648b436cc80cf4ccc36f56de17272d9e663cbd")

# Ensure proper cache directory is available for models
os.environ['TRANSFORMERS_CACHE'] = '/tmp/cache'

# Initialize FastAPI Router
router = APIRouter()

# Lazy loading of large models (only load embeddings and index when required)
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"},
)

index_path = Path("models/index.faiss")
if not index_path.exists():
    raise FileNotFoundError("FAISS index not found. Please generate it and place it in 'ipc_vector_db'.")

# Load the FAISS index
db = FAISS.load_local("models", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define the prompt template for the legal chatbot
prompt_template = """<s>[INST]This is a chat template and as a legal chatbot specializing in Indian Penal Code queries, your objective is to provide accurate and concise information.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

# Set up the LLM (Large Language Model) for the chatbot
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=TOGETHER_AI_API,
)

# Set up memory for conversational context
memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Create the conversational retrieval chain with the LLM and retriever
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
)

# Input schema for chat requests
class ChatRequest(BaseModel):
    question: str
    chat_history: str

# Input schema for FIR-related requests
class FIRDescriptionRequest(BaseModel):
    description_offense: str

# POST endpoint to handle chat requests
@router.post("/chat/")
async def chat(request: ChatRequest):
    try:
        # Prepare the input data
        inputs = {"question": request.question, "chat_history": request.chat_history}
        # Run the chain to get the answer
        result = qa_chain(inputs)
        return {"answer": result["answer"]}
    except Exception as e:
        # Return an error if something goes wrong
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# POST endpoint to handle FIR description processing
@router.post("/process-fir-description/")
async def process_fir_description(request: FIRDescriptionRequest):
    try:
        # Prepare the input data for LawGPT to process the offense description
        context = "FIR Description Processing"
        chat_history = ""  # Empty history for a fresh description

        # Send the description_offense to LawGPT for processing
        result = qa_chain({
            "question": request.description_offense,
            "chat_history": chat_history
        })

        # Get the processed response
        processed_description = result["answer"]
        return {"processed_description": processed_description}
    except Exception as e:
        # Return an error if something goes wrong
        raise HTTPException(status_code=500, detail=f"Error processing FIR description: {str(e)}")

# GET endpoint to check if the API is running
@router.get("/")
async def root():
    return {"message": "LawGPT API is running."}
