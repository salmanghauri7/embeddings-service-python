import httpx
import os
import pymupdf
import tempfile
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from app.config import settings
from app.db.database import db
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from bson import ObjectId

# Schema definitions based on the Mongoose model
class ChunkMetadata(BaseModel):
    page: Optional[int] = None
    totalPages: Optional[int] = Field(None, alias="total_pages")
    chunkLength: Optional[int] = None

class ChunkDBItem(BaseModel):
    paperId: Optional[str] = None
    content: str
    embedding: List[float]
    metadata: ChunkMetadata
    createdAt: datetime = Field(default_factory=datetime.utcnow)

class EmbeddingService: # Capitalized Class name (PEP 8 standard)
    def __init__(self, load_embedding_model: bool = True):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embedding_model = None
        if load_embedding_model:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def _ensure_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    async def download_pdf(self, pdf_link: str) -> list:
        pdf_url = str(pdf_link)
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(pdf_url)
            response.raise_for_status()

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                temp_path = tmp.name

            documents = []
            full_text = ""
            with pymupdf.open(temp_path) as doc:
                # Get global PDF metadata (Author, Title, etc.)
                pdf_info = doc.metadata 

                for page_num, page in enumerate(doc):
                    # sort=True is vital for 2-column academic papers!
                    content = page.get_text("text", sort=True)
                    
                    if content.strip(): # Only add if the page isn't empty
                        full_text += "\n\n" + content
                        documents.append({
                            "content": content,
                            "metadata": {
                                "page": page_num + 1,
                                "total_pages": len(doc)
                            }
                        })

            # We optionally return the full text so the API route can enqueue it
            return documents, full_text
        
        except Exception as e:
            # Good to log the error for your FYP debugging
            print(f"Error processing PDF: {e}")
            raise 

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    async def generate_summary_of_pdf(self, content: str):
        if not content.strip():
            return ""

        messages = [
            {"role": "system", "content": settings.SUMMARY_PROMPT},
            {"role": "user", "content": content},
        ]

        headers = {
            "Authorization": f"Bearer {settings.HF_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": settings.SUMMARY_MODEL,
            "messages": messages,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        data = response.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Unexpected response format from Hugging Face chat completions") from exc

    async def generate_answer_from_context(self, question: str, context: str):
        if not context.strip():
            return "No context found to answer the question."

        messages = [
            {"role": "system", "content": "You are an expert AI assistant. Answer the user's question accurately using ONLY the provided context. If the context does not contain the answer, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ]

        headers = {
            "Authorization": f"Bearer {settings.HF_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": settings.SUMMARY_MODEL,
            "messages": messages,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        data = response.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Unexpected response format from Hugging Face chat completions") from exc

    async def restructure_query(self, question: str, conversation_history: list = None):
        if not question.strip():
            return {"summary": False, "revised_question": ""}

        system_prompt = """You are an AI assistant helping to structure queries for a RAG (Retrieval-Augmented Generation) system.
If the user's question asks for a general summary of the paper (e.g., "explain this paper", "what is this research about", "summarize the pdf"), set "summary" to true and leave "revised_question" empty.
If it is a specific question, set "summary" to false and rewrite the user's question into a highly optimized search query. 
Crucially:
1. Strip all conversational filler (e.g., "Can you explain", "What is", "Please tell me").
2. Focus strictly on the core entities, keywords, and concepts. 
3. Include relevant context from the conversation history if needed to make it standalone.
For example, if the user asks "Can you explain what the main contribution of the paper is?", the revised_question should be something like "main contribution proposed method novelty".

You MUST output strictly in JSON format matching this schema:
{
  "summary": boolean,
  "revised_question": "string"
}
Do not output any markdown code blocks or additional text."""

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if conversation_history:
            # Reformat incoming history into valid HF chat messages
            for msg in conversation_history:
                role = msg.get("role", "user")
                # The backend might send the text in "message" (DB schema) or "content"
                content = msg.get("message") or msg.get("content", "")
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question})

        headers = {
            "Authorization": f"Bearer {settings.HF_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": settings.SUMMARY_MODEL,
            "messages": messages,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"].strip()
            import json
            # Sanitize content in case the model returns markdown JSON blocks
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing JSON from LLM: {e}")
            return {"summary": False, "revised_question": question}

    async def upload_summary_to_db(self, paper_id: str, summary: str):
        """
        Placeholder for uploading the generated summary to the database.
        You can create a new collection for summaries or update an existing paper document with the summary field.
        """
        try:
            collection = db.get_collection("savedpapers")

            result = await collection.update_one(
                {"_id": ObjectId(paper_id)},
                {"$set": {"summary": summary, "summaryGenerated": True}}
            )
            print(f"Summary upload result for paper_id {paper_id}: {result.raw_result}")
            return result
        except Exception as e:
            print(f"Error uploading summary to database for paper_id {paper_id}: {e}")
            raise

    async def get_summary_from_db(self, paper_id: str):
        """
        Retrieves the generated summary for a given paper from the database.
        """
        try:
            collection = db.get_collection("savedpapers")
            document = await collection.find_one({"_id": ObjectId(paper_id)})
            
            if document and "summary" in document:
                return document["summary"]
            return None
        except Exception as e:
            print(f"Error fetching summary from database for paper_id {paper_id}: {e}")
            raise

    def split_text(self, documents: list) -> list:
        """
        Takes the list of page-based dicts from your downloadPdf 
        and returns a list of smaller chunks with metadata.
        """

        final_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["content"])

            for chunk in chunks:
                final_chunks.append({
                    "content": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_length":  len(chunk)
                    }
                })
        
        return final_chunks
    
    def generate_embeddings(self, chunks: list) -> list:
        """
        Placeholder for your embedding generation logic.
        You can integrate OpenAI, Hugging Face, or any other embedding model here.

        for now, i am going to use sentence-transformers/all-MiniLM-L6-v2 
        as the embedding model which is small and makes embeddings of 384 dimensions. I will use the sentence-transformers library to generate embeddings.

        but i am thinking of using sentence-transformers/multi-qa-mpnet-base-dot-v1
        which is a bit larger but makes embeddings of 768 dimensions and is more accurate for question-answering tasks. I will experiment with both models and see which one works better for my use case.
        """
        self._ensure_embedding_model()

        texts = [chunk["content"] for chunk in chunks]

        # 2. Encode EVERYTHING in one batch (No for-loop!)
        # On a CPU, this is significantly more efficient
        embeddings = self.embedding_model.encode(texts)

        # 3. Re-attach to your existing chunk objects
        for i, chunk in enumerate(chunks):
            # Convert NumPy array to Python list for Supabase
            chunk["embedding"] = embeddings[i].tolist()

        return chunks # Returns the full list of dicts ready for Supabase
    

    async def upload_chunks_to_db(self, chunks: list, paper_id: str):
        """
        Validates and uploads the processed chunks with their embeddings to MongoDB.
        """
        if not chunks:
            print("No chunks to upload.")
            return []

        validated_chunks = []
        for chunk in chunks:
            try:
                # Standardize the total_pages key to match Mongoose schema preference
                if "total_pages" in chunk.get("metadata", {}):
                    chunk["metadata"]["totalPages"] = chunk["metadata"].pop("total_pages")

                # Validate and parse the chunk data using Pydantic
                item = ChunkDBItem(
                    
                    paperId=paper_id,
                    content=chunk["content"],
                    embedding=chunk["embedding"],
                    metadata=chunk["metadata"]
                )
                
                # Exclude unset values and format for MongoDB
                validated_chunks.append(item.model_dump(by_alias=True, exclude_none=True))
            except ValidationError as e:
                print(f"Validation error for a chunk: {e}")
                raise ValueError(f"Chunk validation failed: {e}")

        try:
            collection = db.get_collection("chunkembeddings")
            
            # Enforce indexes similarly to Mongoose
        
            
            # Insert the generated chunks into MongoDB
            result = await collection.insert_many(validated_chunks)
            print(f"Successfully inserted {len(result.inserted_ids)} chunks into the database.")
            return result.inserted_ids
        except Exception as e:
            print(f"Error uploading chunks to database: {e}")
            raise
        

