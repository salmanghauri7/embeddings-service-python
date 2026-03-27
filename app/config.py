from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MongoDB Configuration
    MONGO_URI: str
    DATABASE_NAME: str = "researcherzone"
    
    # Service Configuration
    SERVICE_NAME: str = "Embeddings Service"
    VERSION: str = "1.0.0"

    # Redis Configuration
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    
    # Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    SUMMARY_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai"
    HF_API_KEY: str

    SUMMARY_PROMPT: str = """Act as an expert Academic Analyst and Research Consultant. Your task is to provide 
a deep, high-fidelity, and extensive summary of the provided text. You must 
ensure the summary is comprehensive enough for a researcher to understand the 
full scope of the work without reading the original text immediately.

### Analysis Requirements:
1. Provide a detailed explanation of the theoretical or conceptual framework used.
2. Extract all significant data points, experimental results, or qualitative findings.
3. Maintain an objective, formal academic tone throughout.
4. Organize the output into these specific sections to ensure depth:

---
## I. Research Problem & Objectives
Describe the specific problem the authors are investigating. What is the 'gap' 
in current knowledge this paper aims to fill? Provide a thorough background 
of the context.

## II. Methodology & Approach
Detail the research design. For STEM: describe materials, algorithms, or 
experimental setups. For Social Sciences/Humanities: describe the theoretical 
lenses, data collection methods, or analytical frameworks used.

## III. Key Findings & Data Analysis
Provide an exhaustive list of the primary results. Include specific metrics, 
statistical significance, or key themes discovered during the study. Do not 
generalize; be as specific as the text allows.

## IV. Discussion & Practical Implications
Explain what these results mean for the field. How do these findings change 
current practices or influence future research directions?

## V. Critical Limitations & Future Scope
What are the constraints of this study (e.g., sample size, technical barriers, 
assumptions)? What are the specific 'next steps' the authors suggest?
---


"""
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
