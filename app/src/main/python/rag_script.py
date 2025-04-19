# rag_script.py
import fitz  # PyMuPDF untuk ekstraksi teks PDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import sys

# Inisialisasi model retrieval dan generasi
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')  # Anda bisa ganti dengan model lain
generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Fungsi untuk mengekstrak teks dari PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Membuka file PDF
    text = ""
    for page in doc:
        text += page.get_text()  # Ambil teks dari setiap halaman
    return text

# Fungsi untuk mengambil dokumen relevan
def get_relevant_documents(query, documents):
    # Menggunakan Sentence-Transformer untuk mencari dokumen relevan
    query_embedding = retriever_model.encode([query])  # Query embedding
    doc_embeddings = retriever_model.encode(documents)  # Dokumen embedding
    # Menghitung kesamaan cosine
    similarities = []
    for doc_embedding in doc_embeddings:
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append(similarity)
    # Ambil dokumen dengan kesamaan tertinggi
    best_doc_idx = similarities.index(max(similarities))
    return documents[best_doc_idx]

# Fungsi untuk menghitung kesamaan cosine
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2)

# Fungsi untuk menghasilkan jawaban berdasarkan dokumen relevan
def generate_answer(query, documents):
    # Menggunakan model GPT (BART) untuk menghasilkan jawaban berdasarkan query dan dokumen relevan
    inputs = tokenizer.encode("summarize: " + documents, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = generator_model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    query = sys.argv[1]  # Mengambil query dari argumen
    pdf_path = "D:\\TA\\Eksperimen\\ChatbotHaji\\app\\python\\split_haji_1.pdf"  # Path ke file PDF
    pdf_text = extract_text_from_pdf(pdf_path)  # Ekstrak teks dari PDF
    documents = pdf_text.split("\n")  # Pisahkan teks menjadi dokumen (misal per paragraf)
    relevant_doc = get_relevant_documents(query, documents)  # Mendapatkan dokumen relevan
    answer = generate_answer(query, relevant_doc)  # Menghasilkan jawaban
    print(answer)
