#Build vector database using FREE local embeddings (no quota limits!)

from langchain_community.embeddings import HuggingFaceEmbeddings   # NEW: Local & free
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
import os

print("Building vector database with free local embeddings (no API limits)...\n")

# PDF settings
PDF_FOLDER = "pdfs"
PDF_FILES = [
    "CAF-2-MP.pdf",
    "CAF-2-QB.pdf",
    "CAF-2-ST.pdf",
    "CAF-2-Autumn-2024.pdf",
    "CAF-2-TAX-Autumn-2025.pdf",
    "CAF-2-TAX-Spring-2024.pdf",
    "CAF-2-TAX-Spring-2025.pdf"
]

all_docs = []

print("Loading and chunking PDFs (fast strategy)...")

for file_name in PDF_FILES:
    file_path = os.path.join(PDF_FOLDER, file_name)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    print(f"Processing {file_name}...")

    loader = UnstructuredPDFLoader(
        file_path,
        strategy="fast",
        infer_table_structure=True,
        languages=["eng"]
    )
    docs = loader.load()

    for doc in docs:
        doc.metadata["source_book"] = file_name.replace(".pdf", "")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = splitter.split_documents(docs)

    # Content type detection
    for doc in split_docs:
        content = doc.page_content.lower()
        if any(k in content for k in ["mcq", "multiple choice", "choose the correct"]):
            doc.metadata["content_type"] = "MCQ"
        elif any(k in content for k in ["solution", "answer", "explanation", "working"]):
            doc.metadata["content_type"] = "Solution"
        elif any(k in content for k in ["question", "required:", "compute", "calculate"]):
            doc.metadata["content_type"] = "Numerical_Question"
        elif "section" in content:
            doc.metadata["content_type"] = "Theory"
        else:
            doc.metadata["content_type"] = "General"

    all_docs.extend(split_docs)
    print(f"   → {len(split_docs)} chunks")

print(f"\nTotal chunks: {len(all_docs)}")

# FREE Local Embeddings (downloads once)
print("\nLoading local embedding model (first time downloads ~90MB, be patient)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build DB
print("Building vector database (this takes 1-3 minutes)...")
db = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    persist_directory="vector_db"
)

db.persist()
print("\nSUCCESS! Vector database saved in 'vector_db' folder")
print("Ready for chatbot!")

# Quick test
print("\nTest search: 'taxable salary income'")
results = db.similarity_search("taxable salary income", k=4)
for i, doc in enumerate(results):
    ctype = doc.metadata.get("content_type", "Unknown")
    source = doc.metadata.get("source_book", "Unknown")
    print(f"\nResult {i+1} [{ctype} | {source}]")
    print(doc.page_content[:500] + "...")