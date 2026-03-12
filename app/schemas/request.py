from pydantic import BaseModel, HttpUrl
from typing import List, Optional


class EmbeddingRequest(BaseModel):
    pdf_url: HttpUrl
    paper_id: str
    

class EmbeddingResponse(BaseModel):
    success: bool
    message: str
    embeddings: List[float]
