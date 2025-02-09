# Step 1: Use a base image with Python
FROM python:3.9-slim

# Step 2: Set environment variables for caching (optional, but included for consistency)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
ENV TORCH_HOME=/app/.cache/torch

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Copy the requirements file to the container
COPY requirements.txt ./

# Step 5: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Install faiss-gpu separately (if needed)
RUN pip install --no-cache-dir faiss-gpu

# Step 7: Disable caching for models by setting cache_dir=None
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True, cache_dir=None); \
    tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True, cache_dir=None)"

# Step 8: Copy the entire codebase into the container
COPY . /app

# Step 9: Set permissions for the entire app directory
RUN chmod -R 777 /app
RUN chmod -R 777 /app/.cache

# Step 10: Expose the port that FastAPI will run on (default: 7860)
EXPOSE 7860

# Step 11: Set the entry point to run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
