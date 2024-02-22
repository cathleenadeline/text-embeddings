import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/sentence-t5-base')

def encode_sentences(sentences):
    embeddings = model.encode(sentences)
    return embeddings

def write_index(sentences, embeddings, path="/tmp/index.json"):
    index = {}
    i = 0
    for s, e in zip(sentences, embeddings):
        index[i] = {
            "sentence": s,
            "embedding": e.tolist()
        }
        i += 1
    with open(path, 'w') as f:
        json.dump(index, f)

sentences = [
    "This is an example sentence",
    "Each sentence is converted",
    "kacang ijo",
    "kacang kelinci",
    "cabe merah"
]

embeddings = encode_sentences(sentences)

write_index(sentences, embeddings, "/tmp/myindex.json")
