# 🤖 RAG Chatbot From Scratch

A Retrieval-Augmented Generation (RAG) chatbot built from scratch using
Python, LangChain, ChromaDB, Hugging Face embeddings, and Ollama.

This project was developed step-by-step to understand how modern
RAG systems work internally, from document ingestion and embeddings
to retrieval, filtering, LLM generation, conversation memory,
and source citations.

---

## 📌 Project Overview

Traditional LLMs generate answers from their learned knowledge.

A RAG system first retrieves relevant information from a knowledge
base and then provides that information to an LLM as context.

This helps the chatbot generate answers that are grounded in the
provided documents.

### Basic RAG Flow

User Question
      ↓
Query Embedding
      ↓
ChromaDB Vector Search
      ↓
Relevant Documents
      ↓
Relevance Filtering
      ↓
Context Construction
      ↓
Ollama LLM
      ↓
Grounded Answer
      ↓
Source Citation

---

## 🚀 Features

- Document ingestion
- Text chunking
- Hugging Face embeddings
- ChromaDB vector database
- Similarity search
- Retrieval score evaluation
- Hit@K evaluation
- MMR retrieval
- Metadata inspection
- Metadata filtering
- Reusable retriever
- Grounded RAG prompting
- Similarity threshold filtering
- Source citations
- Source details
- Conversation memory
- Error handling
- Multiple document retrieval
- Reusable RAG functions
- Interactive terminal chatbot
- Unknown-question handling

---

## 🛠️ Technologies Used

- Python
- LangChain
- LangChain Core
- LangChain Hugging Face
- LangChain Chroma
- LangChain Ollama
- ChromaDB
- Sentence Transformers
- Ollama
- Qwen 2.5 Coder 7B

---

## 🧠 Architecture

```text
                    ┌─────────────────┐
                    │  Knowledge Base │
                    │  knowledge.txt  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Chunking    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Embeddings    │
                    │ Hugging Face    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    ChromaDB     │
                    │ Vector Database │
                    └────────┬────────┘
                             │
                             │
User Question ───────────────┘
        │
        ▼
┌─────────────────────┐
│ Query Embedding     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Similarity Search   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Relevance Filtering │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Retrieved Context   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Ollama LLM          │
│ Qwen 2.5 Coder 7B   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Grounded Answer     │
│ + Sources           │
└─────────────────────┘
