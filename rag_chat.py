"""
rag_chat.py - Chat with a PDF locally using Ollama + ChromaDB (no API key, no cloud).

Usage:
    python rag_chat.py path/to/document.pdf

Needs:
    ollama pull nomic-embed-text
    ollama pull llama3.2
    pip install -r requirements.txt

Tutorial: https://www.techprofree.com/rag-tutorial-python-ollama/
"""

import sys
import ollama
import chromadb
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 4             # how many chunks to give the model


# ---------- 1. Load ----------
def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""   # extract_text() can return None
        if text.strip():
            pages.append(text)
    full = "\n".join(pages)
    if not full.strip():
        sys.exit("No text found. This PDF is probably scanned images - run OCR first.")
    print(f"Loaded {len(reader.pages)} pages, {len(full):,} characters")
    return full


# ---------- 2. Split ----------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap   # step back so chunks overlap a little
    print(f"Split into {len(chunks)} chunks")
    return chunks


# ---------- 3. Embed + store ----------
def build_index(chunks: list[str], name: str = "my_pdf"):
    client = chromadb.PersistentClient(path="./chroma_db")   # saved to disk, reused next run
    # start fresh each time so re-running with a different PDF doesn't mix documents
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name)

    # embed in batches - one call per chunk is slow
    batch = 32
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        result = ollama.embed(model=EMBED_MODEL, input=part)
        collection.add(
            ids=[str(i + j) for j in range(len(part))],
            embeddings=result["embeddings"],
            documents=part,
        )
        print(f"  embedded {min(i + batch, len(chunks))}/{len(chunks)}", end="\r")
    print("\nIndex ready.")
    return collection


# ---------- 4 + 5. Retrieve + generate ----------
def ask(collection, question: str) -> str:
    q = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    hits = collection.query(query_embeddings=[q], n_results=TOP_K)
    context = "\n\n---\n\n".join(hits["documents"][0])

    prompt = f"""You are answering questions about a document.
Use ONLY the context below. If the answer is not in the context, say "I can't find that in the document."

Context:
{context}

Question: {question}
Answer:"""

    answer = ""
    for chunk in ollama.generate(model=CHAT_MODEL, prompt=prompt, stream=True):
        token = chunk["response"]
        answer += token
        print(token, end="", flush=True)
    print()
    return answer


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python rag_chat.py document.pdf")

    text = load_pdf(sys.argv[1])
    chunks = chunk_text(text)
    collection = build_index(chunks)

    print("\nChat with your PDF. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print("AI: ", end="", flush=True)
        ask(collection, question)
        print()


if __name__ == "__main__":
    main()
