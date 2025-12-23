# load_pdfs.py
# Part 1: Loading the 3 Tax PDFs properly with table support

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
import os

# Folder where your PDFs are stored
PDF_FOLDER = "pdfs"

# List of your PDF files (change names if different)
PDF_FILES = [
    "CAF-2-MP.pdf",
    "CAF-2-QB.pdf",
    "CAF-2-ST.pdf"
]

# This will store all loaded content
all_documents = []

print("Starting to load your 3 Tax PDFs...")

for file_name in PDF_FILES:
    file_path = os.path.join(PDF_FOLDER, file_name)
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        continue
    
    print(f"Loading: {file_name} ...")
    
    # UnstructuredPDFLoader is smart – it detects tables, headings, text
    # Note: Parameter support varies by version - using compatible approach
    try:
        # Try with mode parameter (supported in newer versions)
        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",  # Use "elements" for better structure detection
        )
    except (TypeError, ValueError) as e:
        # Fallback if parameters are not supported - use basic loader
        print(f"   Warning: Using basic loader for {file_name} (advanced params not supported)")
        loader = UnstructuredPDFLoader(file_path)
    
    # Load the PDF
    try:
        docs = loader.load()
    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        continue
    
    # Optional: Add metadata so we know which book/page it came from
    for doc in docs:
        doc.metadata["source_book"] = file_name
        # Ensure page number exists in metadata (if not already present)
        if "page" not in doc.metadata:
            doc.metadata["page"] = "Unknown"
    
    all_documents.extend(docs)
    print(f"   Successfully loaded {len(docs)} pages/chunks from {file_name}")

print("\nAll done!")
print(f"Total chunks loaded: {len(all_documents)}")
print("Your PDFs are now ready for the next step (chunking + vector database)")

# Optional: See a sample of what was loaded
if all_documents:
    print("\nSample content from first chunk:")
    print(all_documents[0].page_content[:500] + "...")  # First 500 characters