from django.shortcuts import render, redirect
from .forms import PDFDocumentForm, SearchForm
from .models import PDFDocument
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the FAISS index and embedding model
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
DIMENSION = 384  # Dimension of the embeddings
faiss_index = faiss.IndexFlatL2(DIMENSION)  # FAISS index for storing embeddings
chunk_data = []  # List to store chunk text and metadata

def upload_pdf(request):
    global faiss_index, chunk_data
    if request.method == 'POST':
        form = PDFDocumentForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_document = form.save()
            text = extract_text_from_pdf(pdf_document.file.path)
            chunks = chunk_text(text)
            store_chunks_in_faiss(chunks)
            return render(request, 'result.html', {'text': text, 'chunks': chunks})
    else:
        form = PDFDocumentForm()
    return render(request, 'upload.html', {'form': form})

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Size of each chunk (in characters)
        chunk_overlap=200,  # Overlap between chunks (in characters)
        length_function=len  # Function to calculate chunk size
    )
    return text_splitter.split_text(text)

def store_chunks_in_faiss(chunks):
    global faiss_index, chunk_data
    # Generate embeddings for each chunk
    embeddings = EMBEDDING_MODEL.encode(chunks)
    # Add embeddings to the FAISS index
    faiss_index.add(np.array(embeddings))
    # Store chunk text and metadata
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            'id': len(chunk_data),
            'text': chunk,
            'embedding': embeddings[i]
        })

def search_chunks(query, top_k=5):
    global faiss_index, chunk_data
    # Generate embedding for the query
    query_embedding = EMBEDDING_MODEL.encode([query])
    # Search the FAISS index
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    # Retrieve the top-k chunks
    results = []
    for i in indices[0]:
        if i >= 0:  # FAISS may return -1 for invalid indices
            results.append(chunk_data[i])
    return results

def search(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            results = search_chunks(query)
            return render(request, 'search_results.html', {'query': query, 'results': results})
    else:
        form = SearchForm()
    return render(request, 'search.html', {'form': form})