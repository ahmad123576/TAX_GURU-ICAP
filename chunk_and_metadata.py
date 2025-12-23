# Smart chunking with metadata for better retrieval

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
import os

# Folder and file names
PDF_FOLDER = "pdfs"
PDF_FILES = [
    "CAF-2-MP.pdf",
    "CAF-2-QB.pdf",
    "CAF-2-ST.pdf"
]

# This will store final processed documents
all_docs_with_metadata = []

print("Starting Part 2: Loading PDFs again with better settings and adding metadata...\n")

for file_name in PDF_FILES:
    file_path = os.path.join(PDF_FOLDER, file_name)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    print(f"Processing: {file_name}")

    # Improved loader: English language + table support
    # Note: Parameter support varies by version - using compatible approach
    try:
        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",  # Use "elements" for better structure detection
        )
    except (TypeError, ValueError):
        # Fallback if parameters are not supported - use basic loader
        print(f"   Warning: Using basic loader for {file_name} (advanced params not supported)")
        loader = UnstructuredPDFLoader(file_path)
    
    # Load the PDF with error handling
    try:
        docs = loader.load()
    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        continue

    # Add basic source metadata
    for doc in docs:
        doc.metadata["source_book"] = file_name.replace(".pdf", "")  # e.g., CAF-2-QB
        # Ensure page number exists in metadata (if not already present)
        if "page" not in doc.metadata:
            doc.metadata["page"] = "Unknown"

    # Smart chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Good balance: big enough for context, small for accuracy
        chunk_overlap=200,     # Overlap helps connect related content
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(docs)
    
    # Skip if no documents were created
    if not split_docs:
        print(f"   Warning: No chunks created for {file_name}")
        continue

    # Simple auto-tagging: Detect type of content
    for split_doc in split_docs:
        content = split_doc.page_content.lower()

        # Detect content type based on keywords/patterns
        if any(keyword in content for keyword in ["mcq", "multiple choice", "choose the correct", "which of the following"]):
            split_doc.metadata["content_type"] = "MCQ"
        elif any(keyword in content for keyword in ["solution", "answer", "explanation", "working", "computation"]):
            split_doc.metadata["content_type"] = "Solution"
        elif any(keyword in content for keyword in ["question", "required:", "compute", "calculate", "determine", "prepare"]):
            split_doc.metadata["content_type"] = "Numerical_Question"
        elif "section" in content and ("income tax" in content or "sales tax" in content):
            split_doc.metadata["content_type"] = "Theory"
        else:
            split_doc.metadata["content_type"] = "General"

        # Optional: Mark past paper vs study material
        if "CAF-2-MP" in file_name or "past" in content:
            split_doc.metadata["is_past_paper"] = True
        else:
            split_doc.metadata["is_past_paper"] = False

    all_docs_with_metadata.extend(split_docs)
    
    print(f"   Split into {len(split_docs)} smart chunks")

print("\nAll done!")
print(f"Total final chunks: {len(all_docs_with_metadata)}")
print("Metadata added: source_book, page, content_type, is_past_paper")

# Show a few examples
print("\nSample chunks with metadata:")
for i, doc in enumerate(all_docs_with_metadata[:5]):
    print(f"\nChunk {i+1}:")
    print(f"   Source: {doc.metadata.get('source_book', 'Unknown')}")
    print(f"   Type: {doc.metadata.get('content_type', 'Unknown')}")
    # Replace form feed character (chr(12)) with space for cleaner display
    print(f"   Content preview: {doc.page_content[:200].replace(chr(12), ' ')}...")

# Save to a variable for next part (we'll use this list later)
# In next part, we'll save this to a file or directly build vector DB

print("\nReady for Part 3: Building the Vector Database with Gemini Embeddings!")