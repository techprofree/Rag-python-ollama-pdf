# RAG in Python with Ollama — Chat with Your PDF Locally

A minimal, framework-free RAG (Retrieval-Augmented Generation) app: ask questions about any PDF and get answers grounded in its text. Everything runs on your own machine with [Ollama](https://ollama.com) — no API key, no cloud, no cost.

📖 **Full tutorial with screenshots, the mistakes I hit, and fixes:**
https://www.techprofree.com/rag-tutorial-python-ollama/

## What's inside

| File | What it does |
|------|--------------|
| `rag_chat.py` | The complete app: load PDF → chunk → embed → store → retrieve → answer |
| `no_rag_baseline.py` | Ask the same question *without* the PDF, to see why RAG is needed |
| `requirements.txt` | `ollama`, `chromadb`, `pypdf` |

About 90 lines of plain Python. No LangChain, no LlamaIndex — so you can see every step.

## Quick start

1. Install Ollama from https://ollama.com/download and pull the two models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
2. Install the Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run it on any text-based PDF:
   ```bash
   python rag_chat.py document.pdf
   ```

## How it works

```
PDF ──► text ──► chunks ──► embeddings ──► ChromaDB
                                              │
question ──► embedding ──► top-4 chunks ◄─────┘
                                  │
                         prompt + context ──► llama3.2 ──► answer
```

## Requirements

- Python 3.9+
- Ollama running locally
- 8 GB RAM (16 GB makes embedding noticeably faster)

## Common errors

See the [Common errors section](https://www.techprofree.com/rag-tutorial-python-ollama/#common-errors) of the tutorial.

## License

MIT
