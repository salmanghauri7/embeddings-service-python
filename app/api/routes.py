from fastapi import APIRouter, HTTPException
from app.schemas.request import EmbeddingRequest, EmbeddingResponse
from app.services.embedding import EmbeddingService

router = APIRouter()
embedding_service = EmbeddingService()


@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings from a PDF link
    """
    pdf_link = request.pdf_url
    try:
        extracted_docs = await embedding_service.download_pdf(pdf_link)

        chunks = embedding_service.split_text(extracted_docs)

        embeddings = embedding_service.generate_embeddings(chunks)
        await embedding_service.upload_chunks_to_db(chunks)

        return EmbeddingResponse(
            success=True,
            message="Embeddings generated successfully",
            embeddings=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
