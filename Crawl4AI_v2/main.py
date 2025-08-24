# main.py
# This is the main entry point for running the Crawl4AI Mistral Analyst Agent.

import os
import re
import csv
import json
from datetime import datetime
from agent import MistralAnalystAgent, Field
from config import (
    MISTRAL_API_KEY, MISTRAL_API_KEY_1, SEARCH_API_KEY, CSE_ID,
    OUTPUT_DIR, APP_VERSION, RACE_INPUT_FILE, VECTOR_DB_PATH,
    CRAWL_CACHE_DIR, KNOWLEDGE_CACHE_DIR, MIN_CONFIDENCE_THRESHOLD
)
# Import schemas from the dedicated file
from schemas import (
    TRIATHLON_SCHEMA, RUNNING_SCHEMA, SWIMMING_SCHEMA, DUATHLON_SCHEMA,
    DEFAULT_BLANK_FIELDS
)


def serialize_knowledge_base(knowledge_base: dict) -> dict:
    serializable_kb = {}
    for variant, data in knowledge_base.items():
        serializable_kb[variant] = {field_name: field_obj.to_dict() for field_name, field_obj in data.items()}
    return serializable_kb


def deserialize_knowledge_base(cached_data: dict) -> dict:
    knowledge_base = {}
    for variant, data in cached_data.items():
        knowledge_base[variant] = {
            field_name: Field(
                value=field_data.get('value'),
                confidence=field_data.get('confidence', 0.0),
                sources=field_data.get('sources', []),
                inferred_by=field_data.get('inferred_by', '')
            ) for field_name, field_data in data.items()
        }
    return knowledge_base


def format_final_row(festival_name_input: str, variant_name: str, data: dict, schema: list) -> dict | None:
    from dateutil.parser import parse as date_parse
    import ftfy

    def get_value(field_name: str) -> any:
        field_obj = data.get(field_name)
        if not isinstance(field_obj, Field): return ""
        min_thresh = 0.45 if field_obj.inferred_by == "llm_inference" else MIN_CONFIDENCE_THRESHOLD
        if field_obj.confidence >= min_thresh:
            return field_obj.value
        return ""

    def finalize_value(value: any) -> str:
        if value is None: return ""
        if isinstance(value, str) and value.startswith(("{", "[")): return value.strip()
        text = ftfy.fix_text(str(value)).strip()
        if text.lower() in ["na", "n/a", "", "none", "not specified", "null"]: return ""
        return text

    date_str = finalize_value(get_value("date"))
    if date_str:
        try:
            event_date = date_parse(date_str, fuzzy=True, dayfirst=True)
            if event_date.year < 2025:
                print(
                    f"    - ⚠️  Filtering out past event: {festival_name_input} - {variant_name} dated {event_date.year}")
                return None
        except (ValueError, TypeError):
            pass

    row = {}
    for key in schema:
        if key in DEFAULT_BLANK_FIELDS:
            row[key] = ""
            continue
        raw_value = get_value(key)

        # --- NEW & UPDATED FORMATTING LOGIC ---
        if key == "startTime":
            val_str = finalize_value(raw_value)
            if not val_str:
                row[key] = ""
            else:
                try:
                    row[key] = date_parse(val_str, fuzzy=True).strftime("%I:%M %p")
                except (ValueError, TypeError):
                    row[key] = ""
        elif "Cutoff" in key:  # NEW: Specific rule for all cutoff fields
            val_str = finalize_value(raw_value)
            # Find time-like patterns (e.g., 7:00, 2 hours, 90 mins)
            match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?|\d+\s*hours?|\d+\s*hr?s?\.?|\d+\s*minutes?|\d+\s*mins?\.?)',
                              val_str, re.IGNORECASE)
            row[key] = match.group(1).strip() if match else ""
        elif key in ["date", "lastDate"]:
            val_str = finalize_value(raw_value)
            if not val_str:
                row[key] = ""
            else:
                try:
                    row[key] = date_parse(val_str, fuzzy=True).strftime("%d/%m/%Y")
                except (ValueError, TypeError):
                    row[key] = ""
        elif key == "ageLimitation":
            val_str = finalize_value(raw_value)
            match = re.search(r'(\d+)\+?', val_str)
            row[key] = f"{match.group(1)}+" if match else ""
        elif key == "registrationCost":
            val_str = finalize_value(raw_value)
            if not val_str:
                row[key] = ""
            elif val_str.lower() == "free":
                row[key] = "0"
            else:
                match = re.search(r'(\d[\d,.]*)', val_str)
                row[key] = match.group(1).replace(',', '').split('.')[0] if match else ""
        elif any(k in key for k in ["Distance", "gain", "loss", "Edition", "editionYear", "Temperature"]):
            val_str = finalize_value(raw_value)
            match = re.search(r'(\d+\.?\d*)', val_str)
            row[key] = match.group(1) if match else ""
        else:
            row[key] = finalize_value(raw_value)

    row[
        "event"] = f"{festival_name_input} - {variant_name}" if festival_name_input.lower() not in variant_name.lower() else variant_name

    if row.get("date"):
        try:
            dt = date_parse(row["date"], dayfirst=True)
            row["month"] = dt.strftime("%B")
            year_str = str(dt.year)
            row["editionYear"], row["lastEdition"] = year_str, year_str
            first_ed_str = row.get("firstEdition", "")
            if first_ed_str and first_ed_str.isdigit():
                count = int(year_str) - int(first_ed_str) + 1
                row["countEditions"] = str(count) if count > 0 else "1"
            else:
                row["countEditions"] = "1"
        except (ValueError, TypeError):
            pass

    # --- NEW: SET DEFAULT VALUES FOR SPECIFIC FIELDS ---
    if not row.get("restrictedTraffic"):
        row["restrictedTraffic"] = "Yes"
    if not row.get("aidStations"):
        row["aidStations"] = "Yes"
    if not row.get("approvalStatus"):
        row["approvalStatus"] = "Approved"

    for key in DEFAULT_BLANK_FIELDS:
        if key in row: row[key] = ""

    return row


def main(output_dir_override=None):
    """
    Main execution function for the Mistral Analyst Agent.
    :param output_dir_override: If provided, this path will be used for outputs instead of the one from config.
    """
    print("=" * 60)
    print(f"🚀 LAUNCHING Crawl4AI Agent v{APP_VERSION}...")
    print("=" * 60)

    # --- FEATURE: Use user-selected output directory ---
    effective_output_dir = output_dir_override if output_dir_override else OUTPUT_DIR
    print(f"INFO: Output directory set to: {os.path.abspath(effective_output_dir)}")

    try:
        with open(RACE_INPUT_FILE, 'r', encoding='utf-8') as f:
            races = json.load(f)
        races.sort(key=lambda x: x.get('Priority', 99))
        print(f"✅ Found {len(races)} events to process from '{RACE_INPUT_FILE}'.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ CONFIGURATION ERROR: Could not read '{RACE_INPUT_FILE}'. Error: {e}");
        return

    for dir_path in [effective_output_dir, CRAWL_CACHE_DIR, KNOWLEDGE_CACHE_DIR, VECTOR_DB_PATH]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path);
            print(f"📂 Created directory: {dir_path}")

    grouped_races, failed_missions = {}, []
    for race in races:
        race_type = race.get("Type", "Unknown").lower()
        if race_type not in grouped_races: grouped_races[race_type] = []
        grouped_races[race_type].append(race)

    csv_writers, output_files = {}, {}
    try:
        for race_type, race_list in grouped_races.items():
            schema_map = {"triathlon": TRIATHLON_SCHEMA, "running": RUNNING_SCHEMA, "trail running": RUNNING_SCHEMA,
                          "swimming": SWIMMING_SCHEMA, "duathlon": DUATHLON_SCHEMA}
            schema = schema_map.get(race_type)
            if not schema: print(f"⚠️  Skipping unknown race type '{race_type}'."); continue

            agent = MistralAnalystAgent(mistral_key_1=MISTRAL_API_KEY, mistral_key_2=MISTRAL_API_KEY_1,
                                        search_key=SEARCH_API_KEY, cse_id=CSE_ID, schema=schema)
            output_filename = f"{datetime.now():%Y-%m-%d}_{APP_VERSION}_{race_type}.csv"

            # Use the effective output directory
            output_filepath = os.path.join(effective_output_dir, output_filename)

            print(f"\n💾 Processing {len(race_list)} '{race_type}' events. Output will be saved to: {output_filepath}")
            output_files[race_type] = open(output_filepath, 'w', newline='', encoding='utf-8')
            writer = csv.DictWriter(output_files[race_type], fieldnames=schema)
            writer.writeheader()
            csv_writers[race_type] = writer

            for i, race_info in enumerate(race_list):
                event_name = race_info.get("Festival")
                if not event_name: print(f"⚠️  Skipping item #{i + 1} as it has no 'Festival' name."); continue

                print("\n" + "=" * 60)
                print(f"🏁 STARTING MISSION FOR '{race_type.upper()}': {event_name}")
                print("=" * 60)

                caching_key = agent.get_caching_key(event_name)
                cache_file_path = os.path.join(KNOWLEDGE_CACHE_DIR, f"{caching_key}.json")
                knowledge_base, is_fresh_run = None, False

                if os.path.exists(cache_file_path):
                    print(f"🧠 Found knowledge cache for '{caching_key}'. Loading data.")
                    with open(cache_file_path, 'r', encoding='utf-8') as f:
                        knowledge_base = deserialize_knowledge_base(json.load(f))
                else:
                    print(f"🧠 No knowledge cache found for '{caching_key}'. Running a full analysis.")
                    knowledge_base = agent.run(race_info)
                    is_fresh_run = True

                if knowledge_base:
                    for variant_name, data in knowledge_base.items():
                        if row := format_final_row(event_name, variant_name, data, schema):
                            csv_writers[race_type].writerow(row)
                    csv_writers[race_type].writerow({})
                    if is_fresh_run:
                        print(f"💾 Saving new knowledge to cache: {cache_file_path}")
                        with open(cache_file_path, 'w', encoding='utf-8') as f:
                            json.dump(serialize_knowledge_base(knowledge_base), f, indent=4)
                    print(f"✅ MISSION COMPLETE FOR: {event_name}")
                else:
                    print(f"❌ MISSION FAILED FOR: {event_name}. No data could be built.")
                    failed_missions.append(event_name)
    finally:
        for f in output_files.values():
            if f and not f.closed: f.close()
        print("\n✅ All output files have been closed.")

    print("\n" + "=" * 60)
    print("🎉 ALL MISSIONS COMPLETE")
    if failed_missions:
        print("\n📋 Summary of Failed Missions:")
        for event in failed_missions: print(f"  - {event}")
    else:
        print("\n✅ All missions completed successfully.")
    print("=" * 60)


if __name__ == '__main__':
    main()