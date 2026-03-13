from fastapi import APIRouter, HTTPException
from arq import create_pool
from arq.connections import RedisSettings
from app.schemas.request import EmbeddingRequest, EmbeddingResponse
from app.services.embedding import EmbeddingService


router = APIRouter()
embedding_service = EmbeddingService()

# Maintain a global redis pool for ARQ
redis_pool = None

@router.on_event("startup")
async def startup_event():
    global redis_pool
    # Initialize connection to Redis for the ARQ worker queue
    redis_pool = await create_pool(RedisSettings(host='localhost', port=6379))

@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings from a PDF link
    """
    pdf_link = request.pdf_url
    paper_id = request.paper_id
    try:
        # download_pdf now returns both the documents (for chunks) and the full_text (for summary)
        extracted_docs, full_text = await embedding_service.download_pdf(pdf_link)

        # ---------------------------------------------------------
        # 1. Enqueue the heavy summary task in the background
        # ---------------------------------------------------------
        if redis_pool:
            print(f"Enqueuing summary task for paper_id: {paper_id}")
            await redis_pool.enqueue_job('generate_pdf_summary_task', full_text, paper_id)
        
        # ---------------------------------------------------------
        # 2. Continue with the standard sync chunk generation
        # ---------------------------------------------------------
        chunks = embedding_service.split_text(extracted_docs)


        embeddings = embedding_service.generate_embeddings(chunks)
        await embedding_service.upload_chunks_to_db(chunks, paper_id)

        return EmbeddingResponse(
            success=True,
            message="Embeddings generated successfully",
            embeddings=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
