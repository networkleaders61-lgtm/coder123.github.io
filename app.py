import os
import io
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TailoredResponse(BaseModel):
    match_percentage: int = Field(description="A score from 0 to 100 matching the resume to the job.")
    missing_skills: List[str] = Field(description="Key skills missing from the resume.")
    improved_bullet_points: List[str] = Field(description="3-5 highly optimized tailored bullet points.")
    mock_interview_questions: List[str] = Field(description="3 highly targeted interview questions.")

@app.post("/api/tailor", response_model=TailoredResponse)
async def tailor_resume(file: UploadFile, job_description: str = Form(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        resume_text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not read PDF text.")

        system_prompt = "You are an expert recruiter. Analyze the resume against the job description. Give a match score, missing skills, improved bullet points, and 3 interview questions."
        user_content = f"JOB:\n{job_description}\n\nRESUME:\n{resume_text}"

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format=TailoredResponse,
        )
        return completion.choices.message.parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

