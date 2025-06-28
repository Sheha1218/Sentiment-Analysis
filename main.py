from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Form
import uvicorn
import pandas as pd
import numpy as np
import pickle
import string
import re
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open('D:/Way to denmark/Projects/Sentiment-Analysis/model/twitter_model.pickle', 'rb') as f:
    model = pickle.load(f)

class_names = {0: 'hate_speech', 1: 'offensive_language', 2: 'Normal'}



def remove_rt(utext):
    return utext.replace('rt :', '')

def remove_punctu(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data['tweet'] = data['tweet'].apply(lambda x: x.lower())
    data['tweet'] = data['tweet'].str.replace('!', '', regex=False)
    data['tweet'] = data['tweet'].apply(lambda x: re.sub(r'@\w+', '', x))
    data['tweet'] = data['tweet'].apply(remove_rt)
    data['tweet'] = data['tweet'].apply(remove_punctu)
    return data['tweet'][0]  # return cleaned string



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": "", "review": ""})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, review: str = Form(...)):
    cleaned_text = preprocessing(review)
    prediction = model.predict([cleaned_text])[0]
    result = class_names.get(prediction, "Unknown")
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "review": review})


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
