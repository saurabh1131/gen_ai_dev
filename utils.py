# utils.py
"""
Financial Chatbot Utilities
Core functionality for RAG-based financial chatbot
"""

import os
import re
import nltk
from nltk.corpus import stopwords
from collections import deque
from typing import Tuple
import torch

import streamlit as st

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Models and ML
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

import sys

sys.path.append('/mount/src/gen_ai_dev')

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initialize NLTK stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
nltk.data.path.append('./nltk_data')  # Point to local NLTK data
stop_words = set(nltk.corpus.stopwords.words('english'))

# Configuration
DATA_PATH = "./Infy financial report/"
DATA_FILES = ["INFY_2022_2023.pdf", "INFY_2023_2024.pdf"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "microsoft/phi-2"

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_DISABLE_TELEMETRY"] = "true"

# Suppress specific warnings
import warnings

warnings.filterwarnings("ignore", message=".*oneDNN custom operations.*")
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*")


# ------------------------------
# Load and Chunk Documents
# ------------------------------
def load_and_chunk_documents():
    """Load and split PDF documents into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []
    for file in DATA_FILES:
        try:
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            pages = loader.load()
            all_chunks.extend(text_splitter.split_documents(pages))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return all_chunks


# ------------------------------
# Vector Store and Search Setup
# ------------------------------
text_chunks = load_and_chunk_documents()
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_vector_db():
    # Load and chunk documents
    text_chunks = load_and_chunk_documents()

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create and return Chroma vector store
    return Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

# Initialize vector_db
vector_db = load_vector_db()

# BM25 setup
bm25_corpus = [chunk.page_content for chunk in text_chunks]
bm25_tokenized = [doc.split() for doc in bm25_corpus]
bm25 = BM25Okapi(bm25_tokenized)

# Cross-encoder for re-ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# ------------------------------
# Conversation Memory
# ------------------------------
class ConversationMemory:
    """Stores recent conversation context"""

    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)

    def add_interaction(self, query: str, response: str) -> None:
        self.buffer.append((query, response))

    def get_context(self) -> str:
        return "\n".join(
            [f"Previous Q: {q}\nPrevious A: {r}" for q, r in self.buffer]
        )


memory = ConversationMemory(max_size=3)


# ------------------------------
# Hybrid Retrieval System
# ------------------------------
def hybrid_retrieval(query: str, top_k: int = 5) -> str:
    try:
        # Semantic search
        semantic_results = vector_db.similarity_search(query, k=top_k * 2)
        print(f"\n\n[For Debug Only] Semantic Results: {semantic_results}")

        # Keyword search
        keyword_results = bm25.get_top_n(query.split(), bm25_corpus, n=top_k * 2)
        print(f"\n\n[For Debug Only] Keyword Results: {keyword_results}\n\n")

        # Combine and deduplicate results
        combined = []
        seen = set()

        for doc in semantic_results:
            content = doc.page_content
            if content not in seen:
                combined.append((content, "semantic"))
                seen.add(content)

        for doc in keyword_results:
            if doc not in seen:
                combined.append((doc, "keyword"))
                seen.add(doc)

        # Re-rank results using cross-encoder
        pairs = [(query, content) for content, _ in combined]
        scores = cross_encoder.predict(pairs)

        # Sort by scores
        sorted_results = sorted(
            zip(combined, scores),
            key=lambda x: x[1],
            reverse=True
        )

        final_results = [f"[{source}] {content}" for (content, source), _ in sorted_results[:top_k]]

        memory_context = memory.get_context()
        if memory_context:
            final_results.append(f"[memory] {memory_context}")

        return "\n\n".join(final_results)

    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""


# ------------------------------
# Safety Guardrails
# ------------------------------
class SafetyGuard:
    """Validates input and filters output"""

    def __init__(self):
        self.financial_terms = {
            'revenue', 'profit', 'ebitda', 'balance', 'cash',
            'income', 'fiscal', 'growth', 'margin', 'expense'
        }
        self.blocked_topics = {
            'politics', 'sports', 'entertainment', 'religion',
            'medical', 'hypothetical', 'opinion', 'personal'
        }

    def validate_input(self, query: str) -> Tuple[bool, str]:
        query_lower = query.lower()
        if any(topic in query_lower for topic in self.blocked_topics):
            return False, "I only discuss financial topics."
        # if not any(term in query_lower for term in self.financial_terms):
        #     return False, "Please ask financial questions."
        return True, ""

    def filter_output(self, response: str) -> str:
        phrases_to_remove = {
            "I'm not sure", "I don't know", "maybe",
            "possibly", "could be", "uncertain", "perhaps"
        }
        for phrase in phrases_to_remove:
            response = response.replace(phrase, "")

        sentences = re.split(r'[.!?]', response)
        if len(sentences) > 2:
            response = '. '.join(sentences[:2]) + '.'

        return response.strip()


guard = SafetyGuard()

# ------------------------------
# LLM Initialization
# ------------------------------
try:
    @st.cache_resource(show_spinner=False)
    def load_generator():
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_4bit=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                device_map="cpu",
                torch_dtype=torch.float32
            )
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.3,
            top_k=30,
            top_p=0.9,
            repetition_penalty=1.2
        )


    # Later in your generate_answer function:
    generator = load_generator()
except Exception as e:
    print(f"Error loading model: {e}")
    raise


# ------------------------------
# Response Generation
# ------------------------------
def extract_final_response(full_response: str) -> str:
    parts = full_response.split("<|im_start|>assistant")
    if len(parts) > 1:
        response = parts[-1].split("<|im_end|>")[0]
        return re.sub(r'\s+', ' ', response).strip()
    return full_response


def generate_answer(query: str) -> Tuple[str, float]:
    try:
        # Input validation
        is_valid, msg = guard.validate_input(query)
        if not is_valid:
            return msg, 0.0

        # Retrieve context
        context = hybrid_retrieval(query)

        # Generate response
        prompt = f"""<|im_start|>system
You are a financial analyst. Provide a brief answer using the context.
Context: {context}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
Answer:"""

        response = generator(prompt)[0]['generated_text']
        clean_response = extract_final_response(response)
        clean_response = guard.filter_output(clean_response)

        # Calculate confidence
        query_embed = embeddings.embed_query(query)
        response_embed = embeddings.embed_query(clean_response)
        confidence = cosine_similarity([query_embed], [response_embed])[0][0]

        # Update memory
        memory.add_interaction(query, clean_response)

        return clean_response, round(confidence, 2)

    except Exception as e:
        return f"Error processing request: {e}", 0.0
