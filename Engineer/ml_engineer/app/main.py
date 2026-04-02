from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Literal
from app import model
from app.schemas import PredictResponse, HealthResponse, RegisterRequest

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg", "image/avif"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load_models()
    yield

app = FastAPI(title="Face Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================
# API Endpoints
# ===================================

# Get health and model status
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        **model.is_ready()
    }

# predict endpoint - accepts image file and returns identity
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), mode: Literal["knn", "cnn", "ensemble"] = Query(default="ensemble"), ):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    
    image_bytes = await file.read()
    result = model.predict(image_bytes=image_bytes, mode=mode)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    
    return result

# register endpoint - accepts image file and label to add to KNN database
@app.post("/register")
async def register(file: UploadFile = File(...), label: str = Form(...), ):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    
    image_bytes = await file.read()
    result = model.register(image_bytes=image_bytes, label=label)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    
    return result

# debug endpoint - accepts image file and returns raw model outputs (for development only)
@app.post("/debug")
async def debug(request: Request):
    form = await request.form()
    return {
        "form_keys": list(form.keys()),
        "content_type": request.headers.get("content-type"),
        "form_data": {k: str(v) for k, v in form.items()}
    }


# usage: run app
import uvicorn
if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )