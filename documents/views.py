from django.shortcuts import render, redirect
from .forms import PDFDocumentForm, SearchForm
from .models import PDFDocument
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the FAISS index and embedding model
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
DIMENSION = 384  # Dimension of the embeddings
faiss_index = faiss.IndexFlatL2(DIMENSION)  # FAISS index for storing embeddings
chunk_data = []  # List to store chunk text and metadata

# Load Llama 2 model and tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Use a smaller model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True,  # Quantize the model to 8-bit
)

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

def generate_response(prompt, max_length=200):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # Move inputs to the model's device
    # Generate text using the model
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,  # Controls randomness
        top_p=0.9,  # Nucleus sampling
        do_sample=True,
    )
    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def summarize_text(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            # Generate a summary using the self-hosted LLM
            prompt = f"Summarize the following text: {query}"
            summary = generate_response(prompt)
            return render(request, 'summary.html', {'summary': summary})
    else:
        form = SearchForm()
    return render(request, 'summarize.html', {'form': form})

def answer_question(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            # Generate an answer using the self-hosted LLM
            prompt = f"Answer the following question: {query}"
            answer = generate_response(prompt)
            return render(request, 'answer.html', {'answer': answer})
    else:
        form = SearchForm()
    return render(request, 'ask.html', {'form': form})