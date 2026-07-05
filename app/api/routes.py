# app/api/routes.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from app.schemas.request import AskRequest, URLUploadRequest
from app.schemas.response import AskResponse, MessageResponse

from app.rag.pipeline import RAGService

router = APIRouter(tags=["RAG APIs"])

# Create singleton service instance
rag_service = RAGService()


@router.post("/upload-pdf", response_model=MessageResponse)
async def upload_pdf(files: List[UploadFile] = File(...)):
    """
    Upload PDF files and ingest into vector DB.
    """
    try:
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No PDF files uploaded"
            )

        rag_service.process_pdfs(files)

        return MessageResponse(
            message="PDF files uploaded successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/upload-url", response_model=MessageResponse)
def upload_url(payload: URLUploadRequest):
    """
    Upload website URLs and ingest into vector DB.
    """
    try:
        if not payload.urls:
            raise HTTPException(
                status_code=400,
                detail="No URLs provided"
            )

        rag_service.process_urls(payload.urls)

        return MessageResponse(
            message="URLs processed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/upload-image", response_model=MessageResponse)
def upload_image(files: List[UploadFile] = File(...)):

    try:
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No image files uploaded"
            )

        rag_service.process_images(files)
        return MessageResponse(
            message="Images processed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    """
    Ask question against uploaded data.
    """
    try:
        answer = rag_service.ask(payload.question)

        return AskResponse(answer=answer)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )