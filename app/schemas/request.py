from pydantic import BaseModel, HttpUrl
from typing import List, Optional


class EmbeddingRequest(BaseModel):
    pdf_url: HttpUrl
    

class EmbeddingResponse(BaseModel):
    success: bool
    message: str
    embeddings: List[float]
