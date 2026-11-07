import os
import io
import json
import requests
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
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

@app.post("/api/tailor")
async def tailor_resume(file: UploadFile, job_description: str = Form(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        resume_text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not read PDF text.")

        system_prompt = (
            "You are an expert recruiter. Analyze the resume against the job description. "
            "You MUST respond ONLY with a valid JSON object matching this structure: "
            '{"match_percentage": 85, "missing_skills": ["Skill1"], "improved_bullet_points": ["Bullet1"], "mock_interview_questions": ["Q1"]}'
        )
        
        user_content = f"JOB:\n{job_description}\n\nRESUME:\n{resume_text}"

        response = requests.post(
            url="https://openrouter.ai",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "response_format": {"type": "json_object"}
            })
        )
        
        result_json = response.json()
        ai_response_string = result_json['choices']['message']['content']
        return json.loads(ai_response_string)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
