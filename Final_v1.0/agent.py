# agent.py
# This file contains the core logic for the Crawl4AI Mistral Analyst Agent.

import os
import re
import csv
import json
import time
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from urllib.parse import urljoin, urlparse

import requests
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage
import ftfy
from dateutil.parser import parse as date_parse

from config import (
    MISTRAL_MODEL, BLACKLISTED_DOMAINS, NEWS_DOMAINS,
    RELEVANT_SUBPAGE_KEYWORDS, MAX_SUBPAGES_PER_SITE, MAX_RETRIES, DEBUG,
    MAX_CONCURRENT_CRAWLERS, MAX_SEARCH_RESULTS, TOP_N_URLS_TO_PROCESS,
    CRAWL_CACHE_DIR, DEFAULT_BLANK_FIELDS, CHOICE_OPTIONS
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
                        print(f"⚠️ Function '{f.__name__}' failed with {e}. Retrying in {current_delay:.2f}s...")
                        time.sleep(current_delay)
                    else:
                        print(f"❌ Function '{f.__name__}' failed after {retries} retries.")
                        return None

        return wrapper

    return decorator


class MistralAnalystAgent:
    def __init__(self, mistral_key_1: str, mistral_key_2: str, search_key: str, cse_id: str, schema: list):
        if not all([mistral_key_1, mistral_key_2, search_key, cse_id]):
            raise ValueError("One or more API keys (Mistral, Google Search) are missing.")

        self.llm_clients = [
            ChatMistralAI(api_key=mistral_key_1, model=MISTRAL_MODEL, temperature=0.0),
            ChatMistralAI(api_key=mistral_key_2, model=MISTRAL_MODEL, temperature=0.0)
        ]
        self.llm_client_index = 0
        self.search_api_key = search_key
        self.cse_id = cse_id
        self.schema = schema
        self.field_instructions = self._generate_field_instructions()
        self.invalid_years = [str(y) for y in range(2015, 2025)]

    def get_caching_key(self, event_name: str) -> str:
        base_name = re.sub(r'sprint|standard|olympic|full iron|half iron|70\.3', '', event_name, flags=re.IGNORECASE)
        return re.sub(r'[^a-z0-9]+', '-', base_name.lower()).strip('-')

    def _generate_field_instructions(self) -> dict:
        instructions = {}
        for key in self.schema:
            if key in DEFAULT_BLANK_FIELDS:
                continue
            if key in CHOICE_OPTIONS:
                instructions[
                    key] = f"Extract the data for '{key}'. MUST be one of the following options: {', '.join(CHOICE_OPTIONS[key])}."
            else:
                instructions[key] = f"Extract the data for '{key}'."
        return instructions

    @retry()
    def _call_llm(self, prompt: str) -> str:
        client = self.llm_clients[self.llm_client_index]
        self.llm_client_index = (self.llm_client_index + 1) % len(self.llm_clients)
        if DEBUG: print(f"    - 🤖 Using Mistral API Key #{self.llm_client_index}")
        messages = [HumanMessage(content=prompt)]
        response = client.invoke(messages)
        return response.content

    def _google_search(self, query: str, num_results=10) -> list:
        print(f"  - Sending raw query to Google: '{query}'")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.search_api_key, "cx": self.cse_id, "q": query, "num": num_results}
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        return [{"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")} for item in
                results.get("items", [])]

    def _step_1a_initial_search(self, race_info: dict) -> list:
        event_name = race_info.get("Festival")
        print(f"\n[STEP 1A] Searching for '{event_name}' with Raw Google Search")
        print("*" * 60)
        query = f'{event_name} 2025 OR 2026'

        try:
            search_results = self._google_search(query, num_results=MAX_SEARCH_RESULTS)
            if not search_results: print("  - ❌ Google Search returned no results."); return []
        except requests.HTTPError as e:
            print(f"  - ❌ Google Search API call failed: {e}");
            return []

        clean_search_results = [r for r in search_results if r.get('link') and not any(
            domain in r['link'] for domain in BLACKLISTED_DOMAINS) and self._is_valid_url(r['link'])]
        print(f"✅ Pre-filtering complete. {len(clean_search_results)} results remain for LLM validation.")
        return clean_search_results

    def _step_1b_validate_and_select_urls(self, event_name: str, search_results: list, top_n: int) -> list:
        print(f"\n[STEP 1B] Validating search results with LLM Analyst")
        print("*" * 60)
        prompt_lines = [
            "You are an intelligence analyst. Your task is to identify the most relevant websites for a given event from a list of Google search results.",
            "Analyze the title, link, and snippet for each result.",
            "Select the single best 'primary_url' (the official event page) and up to three 'secondary_urls' (news, registration portals, etc.).",
            f"\nEvent to find: '{event_name}'",
            f"\nSearch Results:\n```json\n{json.dumps(search_results, indent=2)}\n```",
            "\nYour response MUST be a single valid JSON object with two keys: 'primary_url' (a single string) and 'secondary_urls' (a list of strings).",
        ]
        validation_prompt = "\n".join(prompt_lines)

        response_text = self._call_llm(validation_prompt)
        try:
            match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if match:
                validated_urls = json.loads(match.group(0))
                primary = validated_urls.get("primary_url")
                secondaries = validated_urls.get("secondary_urls", [])
                final_urls = [primary] + secondaries if primary else secondaries
                final_urls = list(dict.fromkeys([url for url in final_urls if url]))
                print(f"✅ LLM Analyst selected {len(final_urls)} relevant URLs.")
                return final_urls[:top_n]
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            print(f"  - ⚠️ LLM validation failed: {e}. Falling back to programmatic selection.")

        return [r['link'] for r in search_results[:top_n] if r.get('link')]

    @retry(retries=2, delay=10)
    def _crawl_url_with_jina(self, url: str) -> str | None:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(CRAWL_CACHE_DIR, f"{url_hash}.md")
        if os.path.exists(cache_path):
            print(f"  - 🧠 Found crawl cache for: {url}")
            with open(cache_path, 'r', encoding='utf-8') as f: return f.read()
        print(f"  - 🕸️  Crawling: {url}")
        try:
            response = requests.get(f"https://r.jina.ai/{url}", timeout=180)
            response.raise_for_status()
            content = response.text
            print(f"  - ✅ Successfully received {len(content)} characters from Jina for {url}")
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return content
        except requests.RequestException as e:
            print(f"  - ❌ Failed to crawl {url}: {e}");
            return None

    def _local_preprocess_markdown(self, markdown: str) -> str:
        text = markdown
        patterns = {
            "social_links": r'\[[^\]]*?\]\((https?:\/\/[^\/]*?(facebook\.com|instagram\.com|twitter\.com|x\.com|linkedin\.com|youtube\.com)[^\)]*?)\)',
            "junk_links": r'\[\s*(shop|help|sign in|faqs|pro series)\s*\]\([^)]+\)', "images": r'!\[.*?\]\(.*?\)',
            "separators": r'={3,}',
        }
        for key, pattern in patterns.items():
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        lines = [line for line in text.split('\n') if not re.search(r'about us|privacy policy', line, re.IGNORECASE)]
        text = re.sub(r'\s{2,}', " ", "\n".join(lines)).strip()
        return text

    def _is_valid_url(self, url: str) -> bool:
        if any(year in url for year in self.invalid_years):
            if DEBUG: print(f"  - Filtering out past year URL: {url}")
            return False
        return True

    def _update_knowledge_base(self, variant_memory: dict, new_text: str, event_name: str, variant_name: str,
                               is_primary_source: bool) -> dict:
        print(f"    - 🧠 Updating knowledge for '{variant_name}'...")
        source_priority_instruction = "This text is from the PRIMARY official source. Trust it highly." if is_primary_source else "This text is from a SECONDARY source (like a news article). Use it to fill in gaps, but be cautious about overwriting existing data unless the new data is clearly more specific."
        prompt_lines = [
            "You are a meticulous data analyst performing fact-checking. Your primary goal is accuracy.",
            source_priority_instruction,
            "Update the `CURRENT_KNOWLEDGE` JSON with new, more specific, or more accurate information from the `NEW_TEXT_CHUNK`.",
            "Your response must be ONLY the complete, updated JSON object.",
            f"Event: '{event_name}'", f"Race Variant to Focus On: '{variant_name}'",
            f"CURRENT_KNOWLEDGE:\n```json\n{json.dumps(variant_memory, indent=2)}\n```",
            f"Data Schema with Instructions:\n```json\n{json.dumps(self.field_instructions, indent=2)}\n```",
            f"NEW_TEXT_CHUNK:\n{new_text}"
        ]
        update_prompt = "\n".join(prompt_lines)
        response_text = self._call_llm(update_prompt)
        if not response_text:
            print("    - ⚠️ LLM call failed after retries. Skipping this knowledge update.")
            return variant_memory
        try:
            match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if match: return json.loads(match.group(0))
        except (json.JSONDecodeError, AttributeError):
            pass
        return variant_memory

    def _discover_and_update_variants(self, knowledge_base: dict, text: str, event_name: str):
        print("    - 🔍 Scanning text for new race variants...")
        variant_prompt = f"From the provided text about '{event_name}', identify all distinct race variants (e.g., 'Olympic Distance', 'Sprint Relay'). Return ONLY a valid JSON list of strings.\n\nText:\n{text[:4000]}"
        response_text = self._call_llm(variant_prompt)
        try:
            match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if match:
                discovered_variants = json.loads(match.group(0))
                for variant in discovered_variants:
                    if variant not in knowledge_base:
                        print(f"    - ✨ New variant discovered and added to knowledge base: '{variant}'")
                        knowledge_base[variant] = {}
        except (json.JSONDecodeError, AttributeError, TypeError):
            if event_name not in knowledge_base:
                knowledge_base[event_name] = {}

    def run(self, race_info: dict) -> dict:
        event_name = race_info.get("Festival")
        search_results = self._step_1a_initial_search(race_info)
        if not search_results: return None
        validated_urls = self._step_1b_validate_and_select_urls(event_name, search_results, TOP_N_URLS_TO_PROCESS)
        if not validated_urls: return None
        return self._crawl_and_extract(validated_urls, event_name)

    def _crawl_and_extract(self, urls: list, event_name: str) -> dict:
        print("\n[STEP 2] Tiered Deep Crawl & Intelligent Extraction")
        print("*" * 60)
        knowledge_base = {event_name: {}}

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CRAWLERS) as executor:
            for i, base_url in enumerate(urls):
                print(f"\nProcessing URL #{i + 1}: {base_url}")
                is_primary = (i == 0)

                main_markdown = self._crawl_url_with_jina(base_url)
                if not main_markdown: continue

                clean_main_text = self._local_preprocess_markdown(main_markdown)
                self._discover_and_update_variants(knowledge_base, clean_main_text, event_name)

                for variant in list(knowledge_base.keys()):
                    knowledge_base[variant] = self._update_knowledge_base(knowledge_base[variant], clean_main_text,
                                                                          event_name, variant,
                                                                          is_primary_source=is_primary)

                link_pattern = re.compile(r'\[.*?\]\((.*?)\)')
                found_links, subpage_urls = link_pattern.findall(main_markdown), set()
                base_netloc, url_friendly_event_name = urlparse(base_url).netloc, re.sub(r'\s+', '-',
                                                                                         event_name.lower())

                for href in found_links:
                    full_url = urljoin(base_url, href)
                    if self._is_valid_url(full_url) and urlparse(full_url).netloc == base_netloc and any(
                            keyword in full_url.lower() for keyword in
                            RELEVANT_SUBPAGE_KEYWORDS) and url_friendly_event_name in full_url.lower():
                        subpage_urls.add(full_url)

                if subpage_urls:
                    future_to_url = {executor.submit(self._crawl_url_with_jina, url): url for url in
                                     list(subpage_urls)[:MAX_SUBPAGES_PER_SITE]}
                    for future in as_completed(future_to_url):
                        if subpage_markdown := future.result():
                            clean_subpage_text = self._local_preprocess_markdown(subpage_markdown)
                            self._discover_and_update_variants(knowledge_base, clean_subpage_text, event_name)
                            for variant in list(knowledge_base.keys()):
                                # *** DEFINITIVE BUG FIX IS HERE: Corrected the typo ***
                                knowledge_base[variant] = self._update_knowledge_base(knowledge_base[variant],
                                                                                      clean_subpage_text, event_name,
                                                                                      variant,
                                                                                      is_primary_source=is_primary)

        print("\n✅ All URLs and subpages processed.")
        return knowledge_base