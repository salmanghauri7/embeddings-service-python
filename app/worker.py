from arq.connections import RedisSettings
from app.services.embedding import EmbeddingService
from app.db.database import db
import os

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

async def save_chat_messages_task(ctx, paper_id: str, user_id: str, question: str, answer: str):
    """
    Background task to save the chat messages (user's question and assistant's answer) to the database.
    """
    print(f"Background Task Started: Saving chat messages for paper {paper_id}")
    try:
        from bson import ObjectId
        from datetime import datetime
        collection = db.get_collection("conversations")
        now = datetime.utcnow()
        documents = [
            {
                "paperId": ObjectId(paper_id),
                "userId": ObjectId(user_id),
                "role": "user",
                "message": question,
                "createdAt": now,
                "updatedAt": now
            },
            {
                "paperId": ObjectId(paper_id),
                "userId": ObjectId(user_id),
                "role": "assistant",
                "message": answer,
                "createdAt": now,
                "updatedAt": now
            }
        ]
        await collection.insert_many(documents)
        print(f"Successfully saved chat messages to database for paper {paper_id}")
    except Exception as e:
        print(f"Failed to save chat messages for paper {paper_id}: {e}")
        raise e

class WorkerSettings:
    functions = [generate_pdf_summary_task, save_chat_messages_task]

    redis_host = os.getenv("REDIS_HOST", "127.0.0.1").strip()
    redis_port = int(os.getenv("REDIS_PORT", "6379").strip())
    redis_settings = RedisSettings(host=redis_host, port=redis_port)
    on_startup = startup
    on_shutdown = shutdown


# to run the worker, use the command: arq app.worker.WorkerSettings