import httpx
import os
import pymupdf
import tempfile

class embeddingService:
    async def downloadPdf(self, pdfLink:str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(pdfLink)
            response.raise_for_status()

        tempPath = None
        try:

            
            with tempFile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                tempPath = tmp.name

        
            text = ''
            with pymupdf.open(tempPath) as doc:
                for page in doc:
                    text += page.get_text()

            return text
        

        finally:
            if tempPath and os.path.exists(tempPath):
                os.remove(tempPath)

       

