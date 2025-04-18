# bm25_script.py
from rank_bm25 import BM25Okapi
import sys
import json

# Ambil input dari argumen atau stdin
documents = json.loads(sys.argv[1])
query = sys.argv[2]

tokenized_corpus = [doc.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
tokenized_query = query.split(" ")
scores = bm25.get_scores(tokenized_query)

print(json.dumps(scores.tolist()))
