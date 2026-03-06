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

# Schema definitions based on the Mongoose model
class ChunkMetadata(BaseModel):
    page: Optional[int] = None
    totalPages: Optional[int] = Field(None, alias="total_pages")
    chunk_length: Optional[int] = None

class ChunkDBItem(BaseModel):
    workspaceId: str
    userId: str
    paperId: Optional[str] = None
    content: str
    embedding: List[float]
    metadata: ChunkMetadata
    createdAt: datetime = Field(default_factory=datetime.utcnow)

class EmbeddingService: # Capitalized Class name (PEP 8 standard)
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    async def download_pdf(self, pdf_link: str) -> list:
        pdf_url = str(pdf_link)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(pdf_url)
            response.raise_for_status()

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                temp_path = tmp.name

            documents = []
            with pymupdf.open(temp_path) as doc:
                # Get global PDF metadata (Author, Title, etc.)
                pdf_info = doc.metadata 

                for page_num, page in enumerate(doc):
                    # sort=True is vital for 2-column academic papers!
                    content = page.get_text("text", sort=True)
                    
                    if content.strip(): # Only add if the page isn't empty
                        documents.append({
                            "content": content,
                            "metadata": {
                                "page": page_num + 1,
                                "total_pages": len(doc)
                            }
                        })

            return documents
        
        except Exception as e:
            # Good to log the error for your FYP debugging
            print(f"Error processing PDF: {e}")
            raise 

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

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
        texts = [chunk["content"] for chunk in chunks]

        # 2. Encode EVERYTHING in one batch (No for-loop!)
        # On a CPU, this is significantly more efficient
        embeddings = self.embedding_model.encode(texts)

        # 3. Re-attach to your existing chunk objects
        for i, chunk in enumerate(chunks):
            # Convert NumPy array to Python list for Supabase
            chunk["embedding"] = embeddings[i].tolist()

        return chunks # Returns the full list of dicts ready for Supabase
    

    async def upload_chunks_to_db(self, chunks: list, workspace_id: str, user_id: str, paper_id: Optional[str] = None):
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
                    workspaceId=workspace_id,
                    userId=user_id,
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
            await collection.create_index(
                [("workspaceId", 1), ("userId", 1)],
                background=True
            )
            
            # Insert the generated chunks into MongoDB
            result = await collection.insert_many(validated_chunks)
            print(f"Successfully inserted {len(result.inserted_ids)} chunks into the database.")
            return result.inserted_ids
        except Exception as e:
            print(f"Error uploading chunks to database: {e}")
            raise
        

