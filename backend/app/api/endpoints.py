# API endpoints for LSH nearest neighbor search.
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from app.services.classification_engine import ClassificationEngine
from app.services.embedding_service import compute_embedding
import json

engine = ClassificationEngine()


router = APIRouter()

# Schema for document input
class DocumentInput(BaseModel):
    content: str
    document_id: Optional[str] = None
    metadata: Optional[dict] = None



# Insert a document into the database and LSH tables
@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image → convert to embedding → classify.
    """

    # Load PIL image
    image = load_image_file(file)

    # Compute embedding (you will implement in embedding_service.py)
    embedding = compute_embedding(image)   # shape: (D,)

    if embedding is None:
        raise HTTPException(status_code=500, detail="Embedding failed.")

    # Predict
    result = engine.predict(embedding)

    return {
        "filename": file.filename,
        "predicted_label": result["label"],
        "confidence": result["confidence"],
    }

# Upload a document file and insert it into the database
@router.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    try:
        # ---------------------------
        # 1. Generate document ID
        # ---------------------------
        document_id = str(uuid.uuid4())

        # ---------------------------
        # 2. Upload raw file to S3
        # ---------------------------
        s3_key = f"documents/{document_id}/{file.filename}"

        # upload_file_to_s3 returns full s3 URI like: s3://bucket/key
        s3_uri = await upload_file_to_s3(
            file=file,
            bucket=settings.s3_bucket_documents,
            key=s3_key
        )

        # ---------------------------
        # 3. Build metadata
        # ---------------------------
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "s3_uri": s3_uri
        }

        # ---------------------------
        # 4. Insert metadata into database
        # ---------------------------
        db = await get_db()
        async with db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO documents (document_id, s3_uri, metadata)
                VALUES ($1, $2, $3::jsonb)
                """,
                document_id,
                s3_uri,
                json.dumps(metadata)
            )

        return {
            "status": "success",
            "document_id": document_id,
            "s3_uri": s3_uri,
            "message": "Document uploaded to S3 and metadata stored successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Query for k nearest neighbors of a given document
@router.post("/query", response_model=List[NearestNeighborResponse])
async def query_nearest_neighbors(query: QueryInput):
    try:
        if query.k <= 0:
            raise HTTPException(status_code=400, detail="k must be positive")
        
        engine = LSHEngine()
        results = await engine.find_nearest_neighbors(
            query_document=query.document,
            k=query.k
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retrieve a document by ID
@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    try:
        engine = LSHEngine()
        document = await engine.get_document(document_id)
        
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# List all documents with pagination
@router.get("/documents")
async def list_documents(limit: int = 10, offset: int = 0):
    try:
        engine = LSHEngine()
        documents = await engine.list_documents(limit=limit, offset=offset)
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

