# app/rag/pipeline.py

import hashlib
from datetime import datetime

from app.rag.pdf_loader import PDFLoaderService
from app.rag.url_loader import URLLoaderService
from app.rag.chunker import ChunkingService
from app.rag.vectorstore import VectorStoreService
from app.rag.llm import get_llm
from app.core.config import settings
from app.rag.image_loader import ImageLoaderService

class RAGService:

    def __init__(self):
        self.pdf_loader = PDFLoaderService()
        self.url_loader = URLLoaderService()
        self.image_loader = ImageLoaderService()
        self.chunker = ChunkingService()
        self.db = VectorStoreService()
        self.llm = get_llm()

        # In-memory cache
        self.answer_cache = {}

        # Conversation memory
        self.chat_memory = []

    # ---------------------------------
    # Document Ingestion
    # ---------------------------------
    def process_pdfs(self, files):

        for file in files:
            docs = self.pdf_loader.load(file)
            chunks = self.chunker.split(docs)
            self.db.add_documents(chunks)

        self.answer_cache.clear()

    def process_urls(self, urls):

        for url in urls:
            docs = self.url_loader.load(str(url))
            chunks = self.chunker.split(docs)
            self.db.add_documents(chunks)

        self.answer_cache.clear()

    def process_images(self, files):

        for file in files:
            docs = self.image_loader.load(file)
            chunks = self.chunker.split(docs)
            self.db.add_documents(chunks)

        self.answer_cache.clear()
    # ---------------------------------
    # Helpers
    # ---------------------------------
    def _normalize_question(self, question: str):

        return question.strip().lower()

    def _cache_key(self, question: str):

        clean = self._normalize_question(question)

        return hashlib.md5(clean.encode()).hexdigest()

    def _get_chat_context(self):

        if not self.chat_memory:
            return ""

        last_items = self.chat_memory[-5:]

        history = ""

        for item in last_items:
            history += f"""
User: {item['question']}
Assistant: {item['answer']}
"""

        return history

    # ---------------------------------
    # Ask RAG
    # ---------------------------------
    def ask(self, question):

        key = self._cache_key(question)

        # ---------------------------------
        # Cache hit
        # ---------------------------------
        if key in self.answer_cache:
            return self.answer_cache[key]

        # ---------------------------------
        # Retrieve docs
        # ---------------------------------
        docs = self.db.search(
            question,
            settings.TOP_K
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        chat_context = self._get_chat_context()

        prompt = f"""
You are a helpful assistant.

Use ONLY the context below.

Previous Chat:
{chat_context}

Knowledge Context:
{context}

Current Question:
{question}

If answer not found, say:
"I could not find this in uploaded knowledge."
"""

        result = self.llm.invoke(prompt)

        answer = result.content

        # ---------------------------------
        # Save cache
        # ---------------------------------
        self.answer_cache[key] = answer

        # ---------------------------------
        # Save memory
        # ---------------------------------
        self.chat_memory.append(
            {
                "question": question,
                "answer": answer,
                "time": str(datetime.now())
            }
        )

        # Keep last 20 turns only
        self.chat_memory = self.chat_memory[-20:]

        return answer