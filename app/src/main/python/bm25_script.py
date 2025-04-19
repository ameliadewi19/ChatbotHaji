from rank_bm25 import BM25Okapi
import sys
import json

def log(msg):
    print(f"[PY_LOG] {msg}", file=sys.stderr)
    
def get_hajj_requirements():
    syarat_haji = """
    1. Islam: Menjadi seorang Muslim adalah syarat utama untuk menunaikan ibadah haji.
    2. Baligh: Menunaikan haji hanya diwajibkan bagi yang sudah baligh (dewasa).
    3. Berakal Sehat: Seseorang yang akan melaksanakan haji harus memiliki akal sehat.
    4. Mampu: Mampu secara fisik dan finansial untuk melakukan perjalanan haji.
    5. Merdeka: Haji hanya diwajibkan bagi orang yang bebas (tidak menjadi budak).
    6. Memiliki biaya: Memiliki biaya yang cukup untuk perjalanan, termasuk biaya perjalanan dan nafkah keluarga yang ditinggalkan.
    """
    return syarat_haji

# Fungsi untuk mendapatkan dokumen dari sumber lain
def get_documents():
    # Ini contoh dokumen yang bisa berasal dari sumber lain
    return [
        """1. Islam: Menjadi seorang Muslim adalah syarat utama untuk menunaikan ibadah haji.
            2. Baligh: Menunaikan haji hanya diwajibkan bagi yang sudah baligh (dewasa).
            3. Berakal Sehat: Seseorang yang akan melaksanakan haji harus memiliki akal sehat.
            4. Mampu: Mampu secara fisik dan finansial untuk melakukan perjalanan haji.
            5. Merdeka: Haji hanya diwajibkan bagi orang yang bebas (tidak menjadi budak).
            6. Memiliki biaya: Memiliki biaya yang cukup untuk perjalanan, termasuk biaya perjalanan dan nafkah keluarga yang ditinggalkan.""",
        "Kalimat kedua ada di sini.",
        "Dan ini adalah kalimat ketiga.",
        "Kalimat keempat yang lebih panjang dan relevan.",
        "Kalimat kelima juga relevan dalam konteks ini."
    ]

def get_bm25_scores(query):
    try:
        # Ambil dokumen dari fungsi lain
        documents = get_documents()
        
        # Proses BM25 untuk mengambil 3 dokumen teratas
        tokenized_corpus = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.split())

        # Gabungkan dokumen dan skor
        doc_scores = list(zip(documents, scores))
        
        # Urutkan berdasarkan skor BM25 dan ambil 3 dokumen teratas
        sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:3]
        
        # Ambil hanya dokumen dari 3 teratas
        top_docs = [doc for doc, score in sorted_docs]
        
        # Mengembalikan dokumen teratas
        return top_docs
    
    except Exception as e:
        print(f"[PY_ERROR] {str(e)}")
        return []
