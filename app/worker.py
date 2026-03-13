from arq.connections import RedisSettings
from app.services.embedding import EmbeddingService
from app.db.database import db
from app.config import settings


async def startup(ctx):
    await db.connect()
    # Summary jobs do not need the embedding model; avoid loading it in the worker.
    ctx["embedding_service"] = EmbeddingService(load_embedding_model=False)


async def shutdown(ctx):
    await db.disconnect()


async def generate_pdf_summary_task(ctx, pdf_text: str, paper_id: str):
    """
    Background task to generate a summary of the PDF and eventually store it.
    """
    print(f"Background Task Started: Generating summary for paper {paper_id}")
    embedding_service = ctx["embedding_service"]
    
    try:
        summary = await embedding_service.generate_summary_of_pdf(pdf_text)
        print(f"Successfully generated summary for paper {paper_id}. Length: {len(summary)}")
        
        # Store the summary in the database
        await embedding_service.upload_summary_to_db(paper_id, summary)
        print(f"Successfully saved summary to database for paper {paper_id}")
        
        return summary
    except Exception as e:
        print(f"Failed to generate summary for paper {paper_id}: {e}")
        raise e

class WorkerSettings:
    functions = [generate_pdf_summary_task]
    redis_settings = RedisSettings(host='localhost', port=6379)
    on_startup = startup
    on_shutdown = shutdown
