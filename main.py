#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import pickle
import numpy as np
import logging
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

from bs4 import BeautifulSoup

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can later restrict to your React domain if you wish
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

class URLRequest(BaseModel):
    urls: List[str]

class QuestionRequest(BaseModel):
    question: str

FILE_PATH = "faiss_store.pkl"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def fetch_html(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch {url}: {e}")

def extract_main_text(html):
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n\n".join(p.get_text() for p in paragraphs)

@app.get("/")
def root():
    return {"message": "News Research API is running!"}

@app.post("/process_urls")
def process_urls(request: URLRequest):
    urls = request.urls
    if not urls:
        raise HTTPException(status_code=400, detail="Please provide at least one URL.")

    try:
        all_texts = []
        for url in urls:
            html = fetch_html(url)
            text = extract_main_text(html)
            if text.strip():
                all_texts.append(text)

        if not all_texts:
            raise HTTPException(status_code=400, detail="No text could be extracted.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = []
        for text in all_texts:
            splits = text_splitter.split_text(text)
            chunks.extend(splits)

        embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks])

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        with open(FILE_PATH, "wb") as f:
            pickle.dump((index, chunks), f)

        return {"message": f"Processed {len(chunks)} chunks successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question
    if not os.path.exists(FILE_PATH):
        raise HTTPException(status_code=400, detail="No data indexed yet. Please call /process_urls first.")

    try:
        with open(FILE_PATH, "rb") as f:
            index, chunks = pickle.load(f)

        question_embedding = embedding_model.encode(question).reshape(1, -1)
        D, I = index.search(question_embedding, k=7)  # increased from 3 to 7

        relevant_texts = " ".join([chunks[i] for i in I[0]])

        # Log retrieved context for inspection
        logging.info(f"Context retrieved for question '{question}':\n{relevant_texts}")

        prompt = f"""You are a helpful assistant that answers questions strictly based on the provided context below.
If the answer cannot be found in the context, reply with exactly: "Not found in the article."

Context:
{relevant_texts}

Question:
{question}

Answer:
"""

        response = generator(prompt, max_length=300, do_sample=False)
        answer = response[0]["generated_text"].strip()

        # Relax strict fallback for testing
        if answer.lower() in ["not found in the article.", "not found"]:
            answer = "Sorry, I couldn't find the answer explicitly, but here's some info related to your question."

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# In[ ]:





# In[ ]:





# In[ ]:




