from fastapi import FastAPI
import uvicorn
from app.lawbot import router as lawgpt_router
from app.predict_pipeline import router as bail_reckoner_router
from app.fir_pdf_gen import router as fir_router
import os

# Set cache directory to a path you have permission to write to
# Set custom cache directories
os.makedirs("/app/.cache/huggingface/transformers", exist_ok=True)
os.makedirs("/app/.cache/sentence_transformers", exist_ok=True)
os.makedirs("/app/.cache/torch", exist_ok=True)
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/app/.cache/sentence_transformers"
os.environ["TORCH_HOME"] = "/app/.cache/torch"

app = FastAPI()

# Include routers with distinct prefixes
app.include_router(lawgpt_router, prefix="/lawgpt", tags=["LawGPT"])
app.include_router(bail_reckoner_router, prefix="/bail-reckoner", tags=["Bail Reckoner"])
app.include_router(fir_router, prefix="/generate-fir", tags=["Generate FIR"])

@app.get("/")
async def root():
    return {
        "message": "API Gateway is running",
        "routes": ["/lawgpt", "/bail-reckoner", "/generate-fir", "/process-fir-description"]
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set

    # Force the host to localhost only
    uvicorn.run("main:app", host="0.0.0.0", port=port)
