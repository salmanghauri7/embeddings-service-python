from fastapi import APIRouter, HTTPException
from app.schemas.request import EmbeddingRequest, EmbeddingResponse
from app.services.embedding import embeddingService

router = APIRouter()
embeddingService = embeddingService()


@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings from a PDF link
    """
    pdfLink = request.pdf_url
    try:
        extractedText = await embeddingService.downloadPdf(pdfLink)
        # TODO: Implement PDF download and embedding generation
        return EmbeddingResponse(
            success=True,
            message="Embeddings generated successfully",
            embeddings=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
