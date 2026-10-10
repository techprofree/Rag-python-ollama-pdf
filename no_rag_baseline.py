"""
no_rag_baseline.py - Ask the model a question about your PDF WITHOUT giving it the PDF.
Run this first to see the model guess or hallucinate, then run rag_chat.py to see the difference.

Usage:  python no_rag_baseline.py "What does chapter 3 say about loops?"
Tutorial: https://www.techprofree.com/rag-tutorial-python-ollama/
"""

import sys
import ollama

question = " ".join(sys.argv[1:]) or "What is this document about?"

response = ollama.generate(model="llama3.2", prompt=question)
print(response["response"])
