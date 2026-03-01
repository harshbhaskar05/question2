from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Create FastAPI app
app = FastAPI()

# Create OpenAI client (pointing to AI Pipe)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# -------------------------
# Request Model
# -------------------------
class CommentRequest(BaseModel):
    comment: str

# -------------------------
# Response Model
# -------------------------
class SentimentResponse(BaseModel):
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    rating: int = Field(..., ge=1, le=5)

# -------------------------
# POST /comment
# -------------------------
@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(request: CommentRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis API. Respond ONLY with valid JSON."
                },
                {
                    "role": "user",
                    "content": f"""
Analyze this comment and return JSON in this exact format:

{{
  "sentiment": "positive|negative|neutral",
  "rating": 1-5
}}

Comment: {request.comment}
"""
                }
            ],
            response_format={"type": "json_object"}
        )

        import json
        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))