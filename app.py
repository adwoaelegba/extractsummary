from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import torch
from bs4 import BeautifulSoup
import requests
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Loading sentence transformer model
sent_model = SentenceTransformer("all-MiniLM-L6-v2")

# Privacy-related keywords and legal phrases the extraction must prioritize when looking for key sentences
legal_phrases = {"third-party sharing", "right to be forgotten", "legitimate interest"}

class URLRequest(BaseModel):
    url: str

def preprocess_text(text):
    for phrase in legal_phrases:
        text = text.replace(phrase, phrase.replace(" ", "_"))
    return text

#TF-IDF extraction function
def td_extract_summary(text, boost_factor=2.0):
    text = preprocess_text(text)
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    tdidf_matrix = vectorizer.fit_transform(sentences)
    word_to_tdidf = dict(zip(vectorizer.get_feature_names_out(), np.asarray(tdidf_matrix.mean(axis=0)).flatten()))

    sentence_weights = [sum(word_to_tdidf.get(word, 0) for word in sentence.lower().split()) for sentence in sentences]
    sentence_embeddings = sent_model.encode(sentences, convert_to_tensor=True)
    doc_embedding = torch.mean(sentence_embeddings, dim=0)
    similarities = torch.nn.functional.cosine_similarity(sentence_embeddings, doc_embedding.unsqueeze(0))
    top_indices = torch.topk(similarities, min(len(sentences), 5)).indices.cpu().numpy()
    top_sentences = [sentences[i] for i in sorted(top_indices)]
    
    return " ".join(top_sentences)

 # Fetch the HTML content from the provided URL
def extraction_function(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch the URL: {url}")
    content = response.text
    
    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")
    sections = {}
    current_heading = None
    current_text = []

    for elem in soup.find_all(["h1", "h2", "h3", "p"]):
        if elem.name in ["h1", "h2", "h3"]:
            if current_heading:
                sections[current_heading] = " ".join(current_text).strip()
            current_heading = elem.get_text().strip()
            current_text = []
        else:
            current_text.append(elem.get_text().strip())

    if current_heading and current_text:
        sections[current_heading] = " ".join(current_text).strip()

    return sections

#API endpoint
@app.post("/extract_summary")
async def summarize(request: URLRequest):
    # Pass the URL from the request to the extraction function
    extracted_summary = extraction_function(request.url) 
    
    if not extracted_summary:
        return {"error": "No text extracted from the document."}
    
    final_summary = {}
    
    for heading, text in extracted_summary.items():
        extractive_mod_summary = td_extract_summary(text)
        
        # Cleaning the text i.e removing any extra spaces
        cleaned_text = extractive_mod_summary.replace("\n", " ").strip()
        final_summary[heading] = cleaned_text
    
    return final_summary
