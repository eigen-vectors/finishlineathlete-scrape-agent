# config.py
# This file contains the dynamic configuration for the Analyst Agent.

import os
from dotenv import load_dotenv
from schemas import * # Import all static lists and schemas

load_dotenv()

# --- API Keys & Model Configuration ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_KEY_1 = os.getenv("MISTRAL_API_KEY_1")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
CSE_ID = os.getenv("CSE_ID")
MISTRAL_MODEL = "mistral-large-latest"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
SPACY_MODEL = 'en_core_web_sm'

# --- File & Execution Configuration ---
RACE_INPUT_FILE = "races.json"
OUTPUT_DIR = "outputs"
APP_VERSION = "v69.0-NoDeepSearch" # Version updated
CRAWL_CACHE_DIR = "crawl_cache"
KNOWLEDGE_CACHE_DIR = "knowledge_cache"
VECTOR_DB_PATH = "vector_db"

# --- Performance & Tuning Configuration ---
TOP_N_URLS_TO_PROCESS = 3
MAX_SEARCH_RESULTS = 10
MAX_SUBPAGES_PER_SITE = 3
MAX_CONCURRENT_CRAWLERS = 5
MAX_RETRIES = 3
DEBUG = True
MIN_CONFIDENCE_THRESHOLD = 0.65

# --- RAG & Re-ranking Configuration ---
RAG_CANDIDATE_POOL_SIZE = 50
RAG_FINAL_EVIDENCE_COUNT = 5