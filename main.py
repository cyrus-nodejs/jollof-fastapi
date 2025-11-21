from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import uvicorn

app = FastAPI(
    title="Embedding Service API",
    description="FastAPI service for text embeddings and cosine similarity using SentenceTransformer.",
    version="1.0.0",
)

# ───────────────────────────────
# Load model (lightweight)
# ───────────────────────────────
model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")

# ───────────────────────────────
# Request Models
# ───────────────────────────────
class TextRequest(BaseModel):
    text: str

class SimilarityRequest(BaseModel):
    userEmbedding: List[float]
    recipeEmbeddings: List[List[float]]

# ───────────────────────────────
# Routes
# ───────────────────────────────
@app.get("/")
async def root() -> Dict[str, str]:
    """Health check route"""
    return {"status": "FastAPI embedding service running 🚀"}

@app.post("/embed")
async def generate_embedding(req: TextRequest) -> Dict[str, List[float]]:
    """Generate embedding for a given text"""
    try:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Missing text input")

        # Explicitly annotate the type
        embedding: List[float] = model.encode(req.text, convert_to_numpy=True).tolist()  # type: ignore
        return {"embedding": embedding}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/similarity")
async def compute_similarity(req: SimilarityRequest) -> Dict[str, List[float]]:
    """Compute cosine similarity between user embedding and recipe embeddings"""
    try:
        user_embedding = np.array(req.userEmbedding)
        recipe_embeddings = np.array(req.recipeEmbeddings)

        if user_embedding.size == 0 or recipe_embeddings.size == 0:
            raise HTTPException(status_code=400, detail="Missing embeddings")

        # Normalize and compute cosine similarities
        user_norm = user_embedding / np.linalg.norm(user_embedding)
        recipe_norms = recipe_embeddings / np.linalg.norm(recipe_embeddings, axis=1, keepdims=True)
        similarities = np.dot(recipe_norms, user_norm)

        return {"similarities": similarities.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {str(e)}")

# ───────────────────────────────
# Server Entry Point
# ───────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
