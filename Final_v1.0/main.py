# main.py
# This is the main entry point for running the Crawl4AI Mistral Analyst Agent.

import os
import re
import csv
import json
from datetime import datetime
from agent import MistralAnalystAgent
from config import (
    MISTRAL_API_KEY, MISTRAL_API_KEY_1, SEARCH_API_KEY, CSE_ID,
    OUTPUT_DIR, APP_VERSION, MISTRAL_MODEL, RACE_INPUT_FILE,
    TRIATHLON_SCHEMA, RUNNING_SCHEMA, SWIMMING_SCHEMA, DEFAULT_BLANK_FIELDS,
    CRAWL_CACHE_DIR, KNOWLEDGE_CACHE_DIR
)


def format_final_row(festival_name_input: str, variant_name: str, data: dict, schema: list) -> dict | None:
    from dateutil.parser import parse as date_parse
    import ftfy

    # *** DEFINITIVE FIX: Standardize missing values to empty string "" instead of "NA" ***
    def finalize_value(value: any) -> str:
        if value is None: return ""
        text = ftfy.fix_text(str(value)).strip()
        if text.lower() in ["na", "n/a", "", "none", "not specified"]: return ""
        return text

    date_str = finalize_value(data.get("date", ""))
    if date_str:
        try:
            event_date = date_parse(date_str, fuzzy=True, dayfirst=True)
            if event_date.year < 2025:
                print(
                    f"    - ⚠️ Filtering out past event: {festival_name_input} - {variant_name} dated {event_date.year}")
                return None
        except (ValueError, TypeError):
            pass

    row = {}
    for key in schema:
        if key in DEFAULT_BLANK_FIELDS:
            row[key] = ""
            continue

        raw_value = data.get(key, "")

        # Apply specific formatting rules, all defaulting to ""
        if key == "startTime":
            val_str = finalize_value(raw_value)
            if not val_str:
                row[key] = ""
            else:
                try:
                    row[key] = date_parse(val_str, fuzzy=True).strftime("%I:%M %p")
                except (ValueError, TypeError):
                    row[key] = ""
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
        elif "Distance" in key or "gain" in key or "loss" in key or "Edition" in key or "editionYear" in key:
            val_str = finalize_value(raw_value)
            match = re.search(r'(\d+\.?\d*)', val_str)
            row[key] = match.group(1) if match else ""
        else:
            row[key] = finalize_value(raw_value)

    row["festivalName"] = row.get("festivalName") or festival_name_input
    row["event"] = f"{row['festivalName']} - {variant_name}" if row[
                                                                    'festivalName'].lower() not in variant_name.lower() else variant_name

    if row.get("date"):
        try:
            dt = date_parse(row["date"], dayfirst=True)
            row["month"] = dt.strftime("%B")
            year_str = str(dt.year)
            row["editionYear"] = year_str
            row["lastEdition"] = year_str

            first_ed_str = row.get("firstEdition", "")
            if first_ed_str and first_ed_str.isdigit():
                count = int(year_str) - int(first_ed_str) + 1
                row["countEditions"] = str(count) if count > 0 else "1"
            else:
                row["countEditions"] = "1"
        except (ValueError, TypeError):
            pass

    return row


def main():
    print("=" * 60)
    print(f"🚀 LAUNCHING Crawl4AI {APP_VERSION}: Mistral Analyst Agent (Batch CSV Mode)")
    print("=" * 60)

    try:
        with open(RACE_INPUT_FILE, 'r', encoding='utf-8') as f:
            races = json.load(f)
        races.sort(key=lambda x: x.get('Priority', 99))
        print(f"✅ Found {len(races)} events to process from '{RACE_INPUT_FILE}'.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ CONFIGURATION ERROR: Could not read or parse '{RACE_INPUT_FILE}'. Error: {e}");
        return

    for dir_path in [OUTPUT_DIR, CRAWL_CACHE_DIR, KNOWLEDGE_CACHE_DIR]:
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
            if race_type == "triathlon":
                schema = TRIATHLON_SCHEMA
            elif race_type in ["running", "trail running"]:
                schema = RUNNING_SCHEMA
            elif race_type == "swimming":
                schema = SWIMMING_SCHEMA
            else:
                print(f"⚠️ Skipping unknown race type '{race_type}'.")
                continue

            agent = MistralAnalystAgent(
                mistral_key_1=MISTRAL_API_KEY, mistral_key_2=MISTRAL_API_KEY_1,
                search_key=SEARCH_API_KEY, cse_id=CSE_ID, schema=schema
            )

            output_filename = f"{datetime.now():%Y-%m-%d}_{APP_VERSION}_{race_type}.csv"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)
            print(f"\n💾 Processing {len(race_list)} '{race_type}' events. Output will be saved to: {output_filepath}")

            output_files[race_type] = open(output_filepath, 'w', newline='', encoding='utf-8')
            writer = csv.DictWriter(output_files[race_type], fieldnames=schema)
            writer.writeheader()
            csv_writers[race_type] = writer

            for i, race_info in enumerate(race_list):
                event_name = race_info.get("Festival")
                if not event_name:
                    print(f"⚠️ Skipping item #{i + 1} as it has no 'Festival' name.")
                    continue

                print("\n" + "=" * 60)
                print(f"🏁 STARTING MISSION FOR '{race_type}': {event_name}")
                print("=" * 60)

                caching_key = agent.get_caching_key(event_name)
                cache_file_path = os.path.join(KNOWLEDGE_CACHE_DIR, f"{caching_key}.json")

                knowledge_base = None
                if os.path.exists(cache_file_path):
                    print(f"🧠 Found knowledge cache for '{caching_key}'. Loading data.")
                    with open(cache_file_path, 'r', encoding='utf-8') as cache_file:
                        knowledge_base = json.load(cache_file)
                else:
                    knowledge_base = agent.run(race_info)

                if knowledge_base:
                    for variant_name, data in knowledge_base.items():
                        row = format_final_row(event_name, variant_name, data, schema)
                        if row:
                            csv_writers[race_type].writerow(row)

                    csv_writers[race_type].writerow({})  # Blank separator row

                    with open(cache_file_path, 'w', encoding='utf-8') as cache_file:
                        json.dump(knowledge_base, cache_file, indent=4)
                    print(f"✅ MISSION COMPLETE FOR: {event_name}")
                else:
                    print(f"❌ MISSION FAILED FOR: {event_name}. No data could be built.")
                    failed_missions.append(event_name)

    finally:
        for f in output_files.values():
            f.close()
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