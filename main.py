from fastapi import FastAPI
from src.pipeline import process_session

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Shiksha Netra ML Pipeline Running"}

@app.post("/process")
def process(data: dict):
    return process_session(data)
