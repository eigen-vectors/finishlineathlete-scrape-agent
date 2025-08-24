# agent.py
# This file contains the core logic for the Crawl4AI Mistral Analyst Agent.

import os
import re
import json
import time
import hashlib
import sqlite3
import sys  # <-- ADDED IMPORT
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from urllib.parse import urljoin, urlparse

import requests
import ftfy
import faiss
import numpy as np
import dirtyjson
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage
from dateutil.parser import parse as date_parse

from markdown_it import MarkdownIt
from readability import Document
from lxml import etree
from rank_bm25 import BM25Okapi

import spacy

from config import (
    MISTRAL_MODEL, MAX_RETRIES, DEBUG, MAX_CONCURRENT_CRAWLERS, MAX_SEARCH_RESULTS,
    TOP_N_URLS_TO_PROCESS, CRAWL_CACHE_DIR, EMBEDDING_MODEL, SPACY_MODEL,
    VECTOR_DB_PATH, CROSS_ENCODER_MODEL, RAG_CANDIDATE_POOL_SIZE,
    RAG_FINAL_EVIDENCE_COUNT, MIN_CONFIDENCE_THRESHOLD
)
from schemas import (
    DEFAULT_BLANK_FIELDS, CHOICE_OPTIONS, INFERABLE_FIELDS, BLACKLISTED_DOMAINS
)


def retry(retries=MAX_RETRIES, delay=5):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if i < retries - 1:
                        is_rate_limit = "429" in str(e)
                        current_delay = delay * (2 ** i) if not is_rate_limit else delay * (3 ** i)
                        print(f"⚠️  Function '{f.__name__}' failed. Retrying in {current_delay:.1f}s...")
                        time.sleep(current_delay)
                    else:
                        print(f"❌  Function '{f.__name__}' failed after {retries} retries.");
                        return None

        return wrapper

    return decorator


class Field:
    def __init__(self, value=None, confidence=0.0, sources=None, inferred_by=""):
        self.value, self.confidence, self.sources, self.inferred_by = value, confidence, sources or [], inferred_by
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def to_dict(self):
        return {"value": self.value, "confidence": self.confidence, "sources": self.sources,
                "inferred_by": self.inferred_by, "last_updated": self.last_updated}


class MistralAnalystAgent:
    def __init__(self, mistral_key_1: str, mistral_key_2: str, search_key: str, cse_id: str, schema: list):
        if not all([mistral_key_1, mistral_key_2, search_key, cse_id]): raise ValueError("API keys missing.")
        self.llm_clients = [ChatMistralAI(api_key=key, model=MISTRAL_MODEL, temperature=0.0) for key in
                            [mistral_key_1, mistral_key_2]]
        self.llm_client_index = 0
        self.search_api_key, self.cse_id, self.schema = search_key, cse_id, schema
        self.field_instructions = self._generate_field_instructions()
        self.invalid_years = [str(y) for y in range(2015, 2025)]

        # --- MODIFIED: ROBUST SPACY MODEL LOADING FOR EXE ---
        print("🚀 Initializing ML models and VectorDB...")
        try:
            # Determine base path, supporting both normal execution and bundled .exe
            if getattr(sys, 'frozen', False):
                # If the application is run as a bundle ('frozen'), the base path is sys._MEIPASS
                # This is the temporary folder where PyInstaller unpacks the app.
                base_path = sys._MEIPASS
            else:
                # If run as a normal .py script, the base path is the script's directory
                base_path = os.path.dirname(os.path.abspath(__file__))

            # Construct the full path to the spaCy model data folder
            # SPACY_MODEL is 'en_core_web_sm' from config.py
            model_path = os.path.join(base_path, SPACY_MODEL)
            print(f"  - Attempting to load spaCy model from path: {model_path}")

            # Check if the model path actually exists before trying to load it
            if not os.path.isdir(model_path):
                 raise IOError(f"spaCy model directory not found at the expected path: {model_path}")

            self.nlp = spacy.load(model_path)
            print("  - ✅ spaCy model loaded successfully by path.")

        except Exception as e:
            # Provide a detailed error message if loading fails for any reason
            print(f"❌ FATAL: Failed to load spaCy model. Error: {e}")
            print("Please ensure the model data is correctly included in the build configuration.")
            # Exit gracefully if the model is essential
            sys.exit(1)
        # --- END OF MODIFICATION ---

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        self.md_parser = MarkdownIt()
        self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH,
                                                       settings=Settings(anonymized_telemetry=False))

        self.chroma_collection = None
        self.bm25_index, self.mission_corpus, self.corpus_map = None, [], {}
        self.type_validation_cache, self.variant_validation_cache, self.semantic_merge_cache = {}, {}, {}
        print("✅ Models and VectorDB initialized.")

    def get_caching_key(self, event_name: str) -> str:
        base_name = re.sub(r'sprint|standard|olympic|full iron|half iron|70\.3', '', event_name, flags=re.IGNORECASE)
        return re.sub(r'[^a-z0-9]+', '-', base_name.lower()).strip('-')

    def _generate_field_instructions(self) -> dict:
        instructions = {}
        for key in self.schema:
            if key in DEFAULT_BLANK_FIELDS: continue
            if key in CHOICE_OPTIONS:
                instructions[
                    key] = f"Extract the data for '{key}'. MUST be one of the following: {', '.join(CHOICE_OPTIONS[key])}."
            else:
                instructions[key] = f"Extract the data for '{key}'."
        return instructions

    @retry()
    def _call_llm(self, prompt: str) -> str:
        client = self.llm_clients[self.llm_client_index]
        self.llm_client_index = (self.llm_client_index + 1) % len(self.llm_clients)
        messages = [HumanMessage(content=prompt)]
        return client.invoke(messages).content

    def _google_search(self, query: str, num_results=10) -> list:
        print(f"  - Searching Google for: '{query}'")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.search_api_key, "cx": self.cse_id, "q": query, "num": num_results}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return [{"title": i.get("title"), "link": i.get("link"), "snippet": i.get("snippet")} for i in
                response.json().get("items", [])]

    def _step_1a_initial_search(self, race_info: dict) -> list:
        event_name = race_info.get("Festival")
        print(f"\n[STEP 1A] Performing initial search for '{event_name}'")
        query = f'{event_name} 2025 OR 2026'
        try:
            search_results = self._google_search(query, num_results=MAX_SEARCH_RESULTS)
            if not search_results: print("  - ❌ Google Search returned no results."); return []
        except requests.HTTPError as e:
            print(f"  - ❌ Google Search API call failed: {e}"); return []
        clean_results = [r for r in search_results if
                         r.get('link') and not any(d in r['link'] for d in BLACKLISTED_DOMAINS) and self._is_valid_url(
                             r['link'])]
        print(f"  - Found {len(clean_results)} potentially relevant URLs.")
        return clean_results

    def _step_1b_validate_and_select_urls(self, event_name: str, search_results: list, top_n: int) -> list:
        print(f"[STEP 1B] Validating search results with LLM...")
        prompt = f"You are an intelligence analyst. Identify the most relevant websites for '{event_name}' from the provided search results. Select the single best 'primary_url' (official page) and up to three 'secondary_urls' (news, registration sites).\n\nSearch Results:\n```json\n{json.dumps(search_results, indent=2)}\n```\n\nYour response MUST be a single valid JSON object with keys 'primary_url' and 'secondary_urls'."
        response_text = self._call_llm(prompt)

        # --- BUG FIX STARTS HERE: ROBUST URL SALVAGING ---
        try:
            match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if match:
                urls = dirtyjson.loads(match.group(0))
                primary = urls.get("primary_url")
                secondaries = urls.get("secondary_urls", [])
                final_urls = list(
                    dict.fromkeys([u for u in ([primary] + secondaries if primary else secondaries) if u]))
                print(f"  - LLM selected {len(final_urls)} relevant URLs.")
                return final_urls[:top_n]
        except (dirtyjson.error.Error, AttributeError, TypeError) as e:
            print(f"  - ⚠️  LLM JSON parsing failed: {e}. Attempting to salvage URLs from raw text.")

        # Fallback 1: Salvage URLs from the raw LLM response text using regex.
        # This is better than the blind fallback because it respects the LLM's reasoning.
        salvaged_urls = re.findall(r'https?://[^\s"\'\)\],]+', response_text)
        if salvaged_urls:
            clean_urls = list(dict.fromkeys(salvaged_urls))
            print(f"  - ✅ Salvaged {len(clean_urls)} URLs directly from the response.")
            return clean_urls[:top_n]

        # Fallback 2: Ultimate fallback to Google's top results if salvage fails.
        print(f"  - ⚠️  Salvage failed. Falling back to top Google search results.")
        return [r['link'] for r in search_results[:top_n] if r.get('link')]
        # --- BUG FIX ENDS HERE ---

    @retry(retries=2, delay=10)
    def _get_content_from_url(self, url: str) -> str | None:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(CRAWL_CACHE_DIR, f"{url_hash}.md")
        if os.path.exists(cache_path):
            if DEBUG: print(f"  - Using cached content for: {url}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        print(f"  - Crawling: {url}")
        try:
            response = requests.get(f"https://r.jina.ai/{url}", timeout=60)
            if response.status_code == 200 and response.text:
                with open(cache_path, 'w', encoding='utf-8') as f: f.write(response.text); return response.text
        except requests.RequestException as e:
            print(f"  - ⚠️  Jina crawl failed. Trying fallback ({e}).")
        try:
            response = requests.get(url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            doc = Document(response.content)
            html_content = etree.tostring(doc.summary_html(pretty_print=True))
            text_content = " ".join(etree.fromstring(html_content).xpath("//text()"))
            clean_text = re.sub(r'\s{2,}', ' ', text_content).strip()
            if clean_text:
                with open(cache_path, 'w', encoding='utf-8') as f: f.write(clean_text); return clean_text
        except Exception as e:
            print(f"  - ❌ Fallback crawl also failed for {url}: {e}")
        return None

    def _chunk_and_index_text(self, text: str, url: str, event_id_str: str):
        chunks = self._chunk_markdown_with_ast(text)
        if not chunks: return
        print(f"    - Semantically chunked into {len(chunks)} passages from {url}")
        chunk_ids = [f"{event_id_str}_{hashlib.md5(chunk.encode()).hexdigest()}" for chunk in chunks]
        unique_chunk_ids = list(set(chunk_ids))
        existing_chunks = self.chroma_collection.get(ids=unique_chunk_ids)
        existing_ids = set(existing_chunks['ids'])
        new_chunks_to_add, added_chunk_content = [], set()
        for i, chunk_id in enumerate(chunk_ids):
            chunk_content = chunks[i]
            if chunk_id not in existing_ids and chunk_content not in added_chunk_content:
                new_chunks_to_add.append({'id': chunk_id, 'chunk': chunk_content})
                added_chunk_content.add(chunk_content)
        if not new_chunks_to_add:
            print("    - All passages from this URL are already in the VectorDB.")
            return
        print(f"    - Found {len(new_chunks_to_add)} new unique passages to index.")
        new_ids, new_documents = [item['id'] for item in new_chunks_to_add], [item['chunk'] for item in
                                                                              new_chunks_to_add]
        new_embeddings = self.embedding_model.encode(new_documents).tolist()
        new_metadatas = [{"source_url": url, "event_id": event_id_str} for _ in new_ids]
        self.chroma_collection.add(ids=new_ids, embeddings=new_embeddings, documents=new_documents,
                                   metadatas=new_metadatas)
        self.mission_corpus.extend(new_documents)

    def _chunk_markdown_with_ast(self, markdown_text: str) -> list[str]:
        try:
            tokens = self.md_parser.parse(markdown_text)
        except Exception:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64).split_text(markdown_text)
        chunks, current_chunk = [], ""
        for token in tokens:
            if token.type.endswith('_open') and token.tag in ['h1', 'h2', 'h3']:
                if current_chunk: chunks.append(current_chunk.strip())
                current_chunk = ""
            if token.content: current_chunk += token.content + "\n"
            if token.type.startswith('table_'):
                if current_chunk and not token.type.endswith('_close'): chunks.append(current_chunk.strip())
                current_chunk = ""
        if current_chunk: chunks.append(current_chunk.strip())
        return [c for c in chunks if c]

    def _retrieve_and_fuse_evidence(self, query: str, top_k: int) -> list[dict]:
        bm25_results = []
        if self.bm25_index and self.mission_corpus:
            doc_scores = self.bm25_index.get_scores(query.lower().split())
            top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
            bm25_results = [self.corpus_map[i] for i in top_n_indices]
        collection_count = self.chroma_collection.count()
        n_results = min(top_k, collection_count)
        hnsw_results = []
        if n_results > 0:
            query_embedding = self.embedding_model.encode(query).tolist()
            chroma_results_set = self.chroma_collection.query(query_embeddings=[query_embedding], n_results=n_results)
            hnsw_results = [{"id": _id, "snippet": doc} for _id, doc in
                            zip(chroma_results_set['ids'][0], chroma_results_set['documents'][0])]
        fused_scores, k = {}, 60
        all_results = {item['id']: item for item in bm25_results + hnsw_results}
        for rank, item in enumerate(bm25_results):
            if item['id'] not in fused_scores: fused_scores[item['id']] = 0; fused_scores[item['id']] += 1 / (
                        k + rank + 1)
        for rank, item in enumerate(hnsw_results):
            if item['id'] not in fused_scores: fused_scores[item['id']] = 0; fused_scores[item['id']] += 1 / (
                        k + rank + 1)
        if not fused_scores: return []
        sorted_fused = sorted(fused_scores.items(), key=lambda i: i[1], reverse=True)
        return [all_results[doc_id] for doc_id, score in sorted_fused[:top_k]]

    def _rerank_evidence_with_cross_encoder(self, query: str, evidence: list[dict]) -> list[dict]:
        if not evidence: return []
        pairs = [(query, item['snippet']) for item in evidence]
        scores = self.cross_encoder.predict(pairs)
        for i, item in enumerate(evidence): item['rerank_score'] = scores[i]
        return sorted(evidence, key=lambda x: x['rerank_score'], reverse=True)

    def _update_knowledge_base_with_rag(self, knowledge_base: dict, event_name: str, variant_name: str):
        print(f"    - Updating knowledge for '{variant_name}'...")
        for field_name in self.schema:
            field_obj = knowledge_base[variant_name].get(field_name, Field())
            if field_name in DEFAULT_BLANK_FIELDS or field_obj.confidence > 0.95: continue
            instruction = self.field_instructions.get(field_name, f"Extract data for '{field_name}'.")
            query = f"Information about '{field_name}' for the '{event_name} - {variant_name}' race."
            candidate_evidence = self._retrieve_and_fuse_evidence(query, top_k=RAG_CANDIDATE_POOL_SIZE)
            reranked_evidence = self._rerank_evidence_with_cross_encoder(query, candidate_evidence)
            final_evidence = reranked_evidence[:RAG_FINAL_EVIDENCE_COUNT]
            if not final_evidence: continue
            json_response_admonition = "Your answer MUST be a single, concise string value."
            if field_name in ['newsCoverage', 'participationCriteria', 'refundPolicy']:
                json_response_admonition = "Your answer MUST be a concise summary in a single string."
            evidence_prompt = "\n".join([f"Evidence Snippet:\n---\n{e['snippet']}\n---" for e in final_evidence])
            prompt = f"You are a data analyst. Based ONLY on the provided evidence, answer the question. Prioritize evidence that seems most relevant.\n\n## Event Focus\nEvent: {event_name}\nRace Variant: {variant_name}\n\n## Evidence\n{evidence_prompt}\n\n## Task\n{instruction}\n{json_response_admonition}\n\nRespond in a single valid JSON object with two keys: 'answer' and 'confidence'. The 'confidence' value MUST be a numerical float between 0.0 and 1.0 (e.g., 0.85), not a word like 'high'. DO NOT add text before or after the JSON."
            response_text = self._call_llm(prompt)
            try:
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    result = dirtyjson.loads(match.group(0))
                    new_value = result.get('answer')
                    if isinstance(new_value, (dict, list)): new_value = json.dumps(new_value)
                    try:
                        new_confidence = float(result.get('confidence', 0.0))
                    except (ValueError, TypeError):
                        if DEBUG: print(
                            f"      - ⚠️  Invalid confidence value from LLM: '{result.get('confidence')}' for '{field_name}'. Defaulting to 0.")
                        new_confidence = 0.0
                    if new_value and new_confidence > field_obj.confidence:
                        sources = [{"id": e.get("id"), "snippet": e["snippet"]} for e in final_evidence]
                        knowledge_base[variant_name][field_name] = Field(value=new_value, confidence=new_confidence,
                                                                         sources=sources,
                                                                         inferred_by="rag_reranked_llm")
            except (dirtyjson.error.Error, AttributeError, TypeError, ValueError) as e:
                if DEBUG: print(f"      - ⚠️  LLM response parsing failed for '{field_name}': {e}.")
        return knowledge_base

    def _discover_and_filter_variants(self, text: str, event_name: str, requested_type: str, knowledge_base: dict):
        print("    - Classifying race variants from text...")
        valid_types = ", ".join(CHOICE_OPTIONS.get('type', []))
        prompt = f"You are a race event analyst. From the text about '{event_name}', identify all distinct race variants mentioned. For each, determine its type based on its description (e.g., a race with running and cycling is a 'Duathlon').\nValid types are: {valid_types}.\nReturn ONLY a single valid JSON object where keys are the full variant names and values are their race type.\nExample:\n{{\n  \"Half Iron - 90km Cycling, 21.1km Run\": \"Duathlon\",\n  \"Olympic Distance Triathlon\": \"Triathlon\"\n}}\n\nText to analyze:\n---\n{text[:4000]}"
        response_text = self._call_llm(prompt)
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                variants_map = dirtyjson.loads(match.group(0))
                for name, discovered_type in variants_map.items():
                    if name in knowledge_base: continue
                    llm_type_lower, req_type_lower = discovered_type.lower(), requested_type.lower()
                    type_cache_key = (llm_type_lower, req_type_lower)
                    is_type_match = self.type_validation_cache.get(type_cache_key)
                    if is_type_match is None:
                        if llm_type_lower == req_type_lower or llm_type_lower in req_type_lower or req_type_lower in llm_type_lower:
                            is_type_match = True
                        else:
                            validation_prompt = f"Is a '{discovered_type}' considered a type of '{requested_type}' event? Please answer with only 'Yes' or 'No'."
                            validation_response = self._call_llm(validation_prompt)
                            is_type_match = "yes" in validation_response.lower()
                        self.type_validation_cache[type_cache_key] = is_type_match
                    if not is_type_match:
                        if DEBUG: print(
                            f"      - Skipping variant '{name}' (type '{discovered_type}' is not '{requested_type}')")
                        continue
                    variant_cache_key = (event_name, name)
                    is_part_of_event = self.variant_validation_cache.get(variant_cache_key)
                    if is_part_of_event is None:
                        if event_name.lower() in name.lower():
                            is_part_of_event = True
                        else:
                            print(f"      - Verifying relationship of '{name}' to '{event_name}'...")
                            validation_prompt = f"You are analyzing the '{event_name}' race festival. Is the race named '{name}' part of this same event festival (e.g., a different distance like a 10K run within a Marathon event)? Answer with only 'Yes' or 'No'."
                            validation_response = self._call_llm(validation_prompt)
                            is_part_of_event = "yes" in validation_response.lower()
                        self.variant_validation_cache[variant_cache_key] = is_part_of_event
                    if not is_part_of_event:
                        if DEBUG: print(f"      - Skipping unrelated event: '{name}'")
                        continue
                    is_semantic_duplicate = False
                    for existing_variant_name in knowledge_base.keys():
                        merge_cache_key = tuple(sorted((name, existing_variant_name)))
                        is_same = self.semantic_merge_cache.get(merge_cache_key)
                        if is_same is None:
                            print(f"      - Checking for semantic similarity: '{name}' vs '{existing_variant_name}'")
                            validation_prompt = f"Are the race variants '{name}' and '{existing_variant_name}' referring to the same race concept (e.g., are they both the full marathon distance)? Answer with only 'Yes' or 'No'."
                            validation_response = self._call_llm(validation_prompt)
                            is_same = "yes" in validation_response.lower()
                            self.semantic_merge_cache[merge_cache_key] = is_same
                        if is_same:
                            is_semantic_duplicate = True
                            if DEBUG: print(
                                f"      - Skipping semantic duplicate: '{name}' is the same as '{existing_variant_name}'")
                            break
                    if not is_semantic_duplicate:
                        print(f"      - ✨ Found relevant variant: '{name}'")
                        knowledge_base[name] = {field: Field() for field in self.schema}
        except (dirtyjson.error.Error, AttributeError, TypeError) as e:
            if DEBUG: print(f"      - ⚠️  Could not parse variant discovery response: {e}.")
            if event_name not in knowledge_base:
                knowledge_base[event_name] = {field: Field() for field in self.schema}

    def _run_inferential_filling(self, knowledge_base: dict):
        print("\n[INFERENCE] Running final analysis to infer missing data...")
        for variant_name, data in knowledge_base.items():
            context = {
                "city": data.get("city").value if data.get("city") and data.get("city").confidence > 0.7 else None,
                "country": data.get("country").value if data.get("country") and data.get(
                    "country").confidence > 0.7 else None,
                "swim_type": data.get("swimType").value if data.get("swimType") and data.get(
                    "swimType").confidence > 0.7 else None}
            if not context["city"]: continue
            for field_name in INFERABLE_FIELDS:
                if field_name not in self.schema or data.get(field_name, Field()).value: continue
                print(f"  - Inferring '{field_name}' for '{variant_name}'...")
                prompt = ""
                if field_name == 'country' and context['city']:
                    prompt = f"What country is the city of '{context['city']}' in? Respond with ONLY the country name."
                elif field_name == 'waterTemperature' and context['city'] and context['swim_type']:
                    prompt = f"For an open water swim event in a '{context['swim_type']}' in '{context['city']}', what is a likely water temperature in Celsius? Provide a reasonable single number estimate (e.g., '18')."
                elif 'Elevation' in field_name and context['city']:
                    options = ", ".join(CHOICE_OPTIONS.get(field_name, []))
                    if options: prompt = f"Considering the general topography of '{context['city']}', what is the most likely course elevation profile for a race? Your answer MUST be one of: {options}."
                if not prompt: continue
                inferred_value = self._call_llm(prompt).strip().replace('"', '')
                if inferred_value:
                    print(f"    - Inferred value: {inferred_value}")
                    knowledge_base[variant_name][field_name] = Field(value=inferred_value, confidence=0.5,
                                                                     inferred_by="llm_inference")
                    if field_name == 'country': context['country'] = inferred_value
        return knowledge_base

    def run(self, race_info: dict) -> dict:
        event_name = race_info.get("Festival")
        search_results = self._step_1a_initial_search(race_info)
        if not search_results: return None
        validated_urls = self._step_1b_validate_and_select_urls(event_name, search_results, TOP_N_URLS_TO_PROCESS)
        if not validated_urls: return None
        return self._crawl_and_extract(validated_urls, race_info)

    def _is_valid_url(self, url: str) -> bool:
        if url.lower().endswith('.pdf'):
            if DEBUG: print(f"  - Skipping PDF link: {url}")
            return False
        if any(year in url for year in self.invalid_years):
            if DEBUG: print(f"  - Filtering out past year URL: {url}")
            return False
        return True

    def _crawl_and_extract(self, urls: list, race_info: dict) -> dict:
        event_name, requested_type = race_info.get("Festival"), race_info.get("Type", "Unknown").lower()
        event_id_str = self.get_caching_key(event_name)
        self.chroma_collection = self.chroma_client.get_or_create_collection(name=event_id_str)
        print(f"\n[STEP 2] Starting RAG processing for '{event_name}' (Collection: {event_id_str})")
        knowledge_base = {}
        self.mission_corpus, self.corpus_map, self.bm25_index = [], {}, None
        self.type_validation_cache, self.variant_validation_cache, self.semantic_merge_cache = {}, {}, {}

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CRAWLERS) as executor:
            futures = {executor.submit(self._get_content_from_url, url): url for url in urls}
            for future in as_completed(futures):
                if content := future.result():
                    url = futures[future]
                    print(f"  - Processing content from: {url}")
                    self._chunk_and_index_text(content, url, event_id_str)
                    self._discover_and_filter_variants(content, event_name, requested_type, knowledge_base)

        all_docs = self.chroma_collection.get()
        self.mission_corpus = all_docs['documents']
        if self.mission_corpus:
            print(f"  - Building BM25 index for {len(self.mission_corpus)} total passages...")
            self.bm25_index = BM25Okapi([doc.lower().split() for doc in self.mission_corpus])
            self.corpus_map = {i: {'id': all_docs['ids'][i], 'snippet': doc} for i, doc in
                               enumerate(self.mission_corpus)}

        if not knowledge_base: print(
            f"  - ⚠️  No variants of type '{requested_type}' were discovered. Aborting."); return {}

        for variant in list(knowledge_base.keys()):
            self._update_knowledge_base_with_rag(knowledge_base, event_name, variant)

        knowledge_base = self._run_inferential_filling(knowledge_base)
        print("\n✅ All search and analysis phases complete.")
        return knowledge_base