from fastapi import APIRouter, HTTPException
from arq import create_pool
from arq.connections import RedisSettings
import os
from app.db.database import db
from app.schemas.request import EmbeddingRequest, EmbeddingResponse, chatRequest
from app.services.embedding import EmbeddingService
from bson import ObjectId


router = APIRouter()
embedding_service = EmbeddingService()

# Maintain a global redis pool for ARQ
redis_pool = None

@router.on_event("startup")
async def startup_event():
    global redis_pool
    # Initialize connection to Redis for the ARQ worker queue
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_pool = await create_pool(RedisSettings(host=redis_host, port=6379))

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

@router.post("/paperChat")
async def chat_with_paper(request: chatRequest):
    '''
    Chat with the paper using the question and paper_id to retrieve relevant chunks and generate a response.
    '''
    try:
        structured_query = await embedding_service.restructure_query(
            question=request.question,
            conversation_history=request.conversation_history
        )
        
        summary = structured_query.get("summary", False)
        revised_question = structured_query.get("revised_question", "")

        if summary:
            # If the user asked for a summary, fetch it from the savedpapers collection
            paper_summary = await embedding_service.get_summary_from_db(request.paperId)
            
            if not paper_summary:
                return {"answer": "Summary is currently being generated or not available. Please try again later."}
            
            return {"answer": paper_summary}
        else:
            # 1. Create embeddings for the string question
            query_embedding = embedding_service.generate_embeddings([{"content": revised_question}])[0]["embedding"]

            # 2. Perform vector search and full text search separately and combine using RRF in Python
            collection = db.get_collection("chunkembeddings")
            
            vector_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_search_for_research_zone",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": 20,
                        "filter": {
                            "paperId": request.paper_id
                        }
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "paperId": 1,
                        "score": { "$meta": "searchScore" }
                    }
                }
            ]

            text_pipeline = [
                {
                    "$search": {
                        "index": "text_search_for_research_zone",
                        "compound": {
                            "must": [
                                {
                                    "text": {
                                        "query": revised_question,
                                        "path": "content"
                                    }
                                }
                            ],
                            "filter": [
                                {
                                    "equals": {
                                        "value": request.paper_id,
                                        "path": "paperId"
                                    }
                                }
                            ]
                        }
                    }
                },
                { "$limit": 20 },
                {
                    "$project": {
                        "content": 1,
                        "paperId": 1,
                        "score": { "$meta": "searchScore" }
                    }
                }
            ]

            import asyncio
            vector_results, text_results = await asyncio.gather(
                collection.aggregate(vector_pipeline).to_list(length=20),
                collection.aggregate(text_pipeline).to_list(length=20)
            )

            # Combine using Reciprocal Rank Fusion (RRF)
            combined_results = {}
            k = 60 # RRF constant

            for rank, doc in enumerate(vector_results):
                doc_id = str(doc["_id"])
                if doc_id not in combined_results:
                    combined_results[doc_id] = {"doc": doc, "rrf_score": 0.0}
                combined_results[doc_id]["rrf_score"] += 1.0 / (k + rank + 1)
                
            for rank, doc in enumerate(text_results):
                doc_id = str(doc["_id"])
                if doc_id not in combined_results:
                    combined_results[doc_id] = {"doc": doc, "rrf_score": 0.0}
                # Weighting text search slightly less (e.g. 0.5) if desired, but standard RRF just sums them
                combined_results[doc_id]["rrf_score"] += (0.5) * (1.0 / (k + rank + 1))

            # Sort by RRF score descending and take top 5
            sorted_docs = sorted(combined_results.values(), key=lambda x: x["rrf_score"], reverse=True)
            results = [item["doc"] for item in sorted_docs[:5]]
            
            # Use the chunks to answer the revised_question (RAG)
            context = "\n".join([doc.get("content", "") for doc in results])
            
            answer = await embedding_service.generate_answer_from_context(question=revised_question, context=context)

            if redis_pool:
                await redis_pool.enqueue_job("save_chat_messages_task", request.paper_id, request.user_id, request.question, answer)

            return {"answer": answer, "results": [str(d) for d in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


