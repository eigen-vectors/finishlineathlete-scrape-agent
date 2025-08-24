# gemini.py (Updated to integrate with the UI)

import os
import re
import io
import json
import datetime
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from dateutil.parser import parse

# --- Configuration ---
load_dotenv()
# MODIFIED: Renamed for clarity. This is now the fallback default.
DEFAULT_INPUT_FOLDER = 'input_folder'
# MODIFIED: Changed to just a filename. The full path will be constructed dynamically.
OUTPUT_CSV_FILENAME = 'event_data.csv'
CACHE_FILE = 'processed_images.log'
ALLOWED_EXTENSIONS = ('.png', '.jpeg', '.jpg')

CSV_HEADERS = [
    'event', 'festivalName', 'imageURL', 'raceVideo', 'type', 'date', 'city', 'organiser',
    'participationType', 'firstEdition', 'lastEdition', 'countEditions', 'mode', 'raceAccredition',
    'theme', 'numberOfparticipants', 'startTime', 'scenic', 'registrationCost', 'ageLimitation',
    'eventWebsite', 'organiserWebsite', 'bookingLink', 'newsCoverage', 'lastDate',
    'participationCriteria', 'refundPolicy', 'swimDistance', 'swimType', 'swimmingLocation',
    'waterTemperature', 'swimCoursetype', 'swimCutoff', 'swimRoutemap', 'cyclingDistance',
    'cyclingElevation', 'cyclingSurface', 'cyclingElevationgain', 'cycleCoursetype', 'cycleCutoff',
    'cyclingRoutemap', 'runningDistance', 'runningElevation', 'runningSurface', 'runningElevationgain',
    'runningElevationloss', 'runningCoursetype', 'runCutoff', 'runRoutemap', 'organiserRating',
    'triathlonType', 'standardTag', 'region', 'approvalStatus', 'difficultyLevel', 'month',
    'primaryKey', 'latitude', 'longitude', 'country', 'editionYear', 'aidStations',
    'restrictedTraffic', 'user_id', 'femaleParticpation', 'jellyFishRelated',
    'registrationOpentag', 'eventConcludedtag', 'state', 'nextEdition'
]

CHOICE_FIELDS = {
    "participationType": ["Individual", "Relay", "Group"],
    "mode": ["Virtual", "On-Ground"],
    "runningSurface": ["Road", "Trail", "Track", "Road + Trail"],
    "runningCourseType": ["Single Loop", "Multiple Loop", "Out and Back", "Point to Point"],
    "region": ["West India", "Central and East India", "North India", "South India", "Nepal", "Bhutan", "Sri Lanka"],
    "runningElevation": ["Flat", "Rolling", "Hilly", "Skyrunning"],
    "type": ["Triathlon", "Aquabike", "Aquathlon", "Duathlon", "Run", "Cycling", "Swimathon"],
    "swimType": ["Lake", "Beach", "River", "Pool"],
    "swimCoursetype": ["Single Loop", "Multiple Loops", "Out and Back", "Point to Point"],
    "cyclingElevation": ["Flat", "Rolling", "Hilly"],
    "cycleCoursetype": ["Single Loop", "Multiple Loops", "Out and Back", "Point to Point"],
    "triathlonType": ["Super Sprint", "Sprint Distance", "Olympic Distance", "Half Iron(70.3)", "Iron Distance (140.6)",
                      "Ultra Distance"],
    "standardTag": ["Standard", "Non Standard"],
    "restrictedTraffic": ["Yes", "No"],
    "jellyFishRelated": ["Yes", "No"],
    "approvalStatus": ["Approved", "Pending Approval"]
}


# --- Helper Functions (Your new improved logic is preserved) ---

def clean_value(value):
    if value is None or str(value).strip().upper() in ["NA", "N/A", "NONE", "NOT SPECIFIED", ""]:
        return ""
    return str(value).encode('utf-8', 'ignore').decode('utf-8').strip()


def validate_choice(value, options):
    cleaned_value = clean_value(value)
    if not cleaned_value: return ""
    for option in options:
        if cleaned_value.lower() == option.lower(): return option
    return ""


def format_date_value(date_str):
    cleaned_str = clean_value(date_str)
    if not cleaned_str: return ""
    try:
        dt = parse(cleaned_str, fuzzy=True, dayfirst=True)
        if dt and dt.year >= 2025: return dt.strftime("%d/%m/%Y")
        return ""
    except (ValueError, TypeError):
        return ""


def format_time_value(time_str):
    cleaned_str = clean_value(time_str)
    if not cleaned_str: return ""
    try:
        match = re.search(r'(\d{1,2})[:.]?(\d{2})?\s*(am|pm)?', cleaned_str, re.IGNORECASE)
        if not match: return ""
        hour, minute, am_pm = match.groups()
        hour = int(hour)
        minute = int(minute) if minute else 0
        if am_pm:
            am_pm = am_pm.upper()
        else:
            am_pm = "AM" if 5 <= hour < 12 else "PM"
        if am_pm == "PM" and hour < 12: hour += 12
        if am_pm == "AM" and hour == 12: hour = 0
        dt = datetime.time(hour, minute)
        return dt.strftime("%I:%M %p")
    except Exception:
        return ""


def extract_numeric(value_str):
    cleaned_str = clean_value(str(value_str))
    if not cleaned_str: return ""
    match = re.search(r'(\d+\.?\d*|\.\d+)', cleaned_str)
    return match.group(1) if match else ""


def extract_registration_cost(value_str):
    cleaned_str = clean_value(value_str)
    if not cleaned_str: return ""
    if "free" in cleaned_str.lower(): return "0"
    cleaned_str = cleaned_str.replace(',', '')
    match = re.search(r'(\d+)', cleaned_str)
    return match.group(1) if match else ""


def extract_age_limit(value_str):
    cleaned_str = clean_value(value_str)
    if not cleaned_str: return ""
    match = re.search(r'(\d+\+?)', cleaned_str)
    return match.group(1) if match else ""


# --- Core API and Processing Logic (Your new improved logic is preserved) ---

def get_gemini_response(image_path: str) -> dict:
    """Sends the image to Gemini with a much smarter, more detailed prompt."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

    try:
        genai.configure(api_key=api_key)

        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            if img.mode == 'RGBA': img = img.convert('RGB')
            img.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()

        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # --- THE NEW, SMARTER PROMPT ---
        prompt = """
        You are a highly intelligent data extraction assistant specializing in analyzing event posters for athletic races (marathons, triathlons, etc.).
        Your task is to meticulously analyze the provided image and extract specific details into a structured JSON format.

        **CRITICAL INSTRUCTIONS:**
        1.  **Be thorough:** Do not give up easily. Scan the entire image for text, logos, and icons.
        2.  **Infer Logically:** If a value isn't explicitly stated, infer it from context. For example, if the event name is "Mumbai Marathon 2025", you MUST infer `city` as "Mumbai" and `date` as being in "2025".
        3.  **Default to Empty, Not 'NA':** Only use an empty string "" for a field if the information is absolutely impossible to find or infer. DO NOT use "NA" or "N/A".
        4.  **Follow the JSON Schema:** Your output MUST be a single, clean JSON object with the exact nested structure provided below. Do not add any extra text or markdown.

        **JSON SCHEMA TO FOLLOW:**

        {
          "eventDetails": {
            "festivalName": "The main title of the event or festival. Often the largest text.",
            "type": "The specific race type. Choose ONE from the list: [Triathlon, Aquabike, Aquathlon, Duathlon, Run, Cycling, Swimathon]. 'Run' is for marathons, 10ks, etc.",
            "theme": "Any specific theme mentioned (e.g., 'Monsoon Run', 'Diwali Edition').",
            "firstEdition": "The year the event first started, if mentioned (e.g., 'Estd. 2010')."
          },
          "eventLogistics": {
            "date": "The primary date of the event. Find the full date including day, month, and year.",
            "lastDate": "The last date for registration, if specified.",
            "startTime": "The start time of the race (e.g., '6:00 AM').",
            "city": "The city where the event is held. Infer from the event title if necessary.",
            "state": "The state or province of the city.",
            "region": "The region. Choose one: [West India, Central and East India, North India, South India, Nepal, Bhutan, Sri Lanka].",
            "mode": "Choose one: [On-Ground, Virtual]."
          },
          "organizerInfo": {
            "organiser": "The name of the organizing company or group.",
            "eventWebsite": "The URL for the event's website.",
            "organiserWebsite": "The URL for the organizer's main website.",
            "bookingLink": "The direct URL for registration or booking tickets."
          },
          "participationInfo": {
            "participationType": "Choose one: [Individual, Relay, Group].",
            "registrationCost": "The cost to register. Look for currency symbols (₹, $, etc.). If it says 'Free', use '0'.",
            "ageLimitation": "Any age restriction (e.g., '18+', 'For under 16s').",
            "numberOfparticipants": "The total number of slots or participants, if mentioned.",
            "participationCriteria": "Any specific criteria for participation (e.g., 'Must have completed a 10k')."
          },
          "raceSegments": {
            "swimDistance": "Distance for the swim leg in km.",
            "swimType": "Choose one: [Lake, Beach, River, Pool].",
            "cyclingDistance": "Distance for the cycling leg in km.",
            "cyclingElevationgain": "Total elevation gain for the cycle route in meters.",
            "runningDistance": "Distance for the run leg in km. (e.g., 42.2, 21.1, 10).",
            "runningSurface": "Choose one: [Road, Trail, Track, Road + Trail].",
            "runningElevationgain": "Total elevation gain for the run route in meters.",
            "runningElevationloss": "Total elevation loss for the run route in meters."
          }
        }
        """

        response = model.generate_content([prompt, image_part], request_options={"timeout": 120})
        clean_response_text = re.sub(r'^```json\s*|\s*```$', '', response.text.strip(), flags=re.MULTILINE)

        return json.loads(clean_response_text)

    except Exception as e:
        print(f"  -> Error calling Gemini API for {os.path.basename(image_path)}: {e}")
        return {}


def process_image_data(raw_data: dict) -> dict:
    """Processes the NESTED dictionary from Gemini and formats it according to all rules."""

    processed_row = {header: "" for header in CSV_HEADERS}

    # --- UPDATED TO PARSE THE NESTED JSON STRUCTURE ---
    details = raw_data.get('eventDetails', {})
    logistics = raw_data.get('eventLogistics', {})
    organizer = raw_data.get('organizerInfo', {})
    participation = raw_data.get('participationInfo', {})
    segments = raw_data.get('raceSegments', {})

    # Event Details
    processed_row['festivalName'] = clean_value(details.get('festivalName'))
    processed_row['type'] = validate_choice(details.get('type'), CHOICE_FIELDS['type'])
    processed_row['theme'] = clean_value(details.get('theme'))
    processed_row['firstEdition'] = extract_numeric(details.get('firstEdition'))

    # Event Logistics
    processed_row['date'] = format_date_value(logistics.get('date'))
    processed_row['lastDate'] = format_date_value(logistics.get('lastDate'))
    processed_row['startTime'] = format_time_value(logistics.get('startTime'))
    processed_row['city'] = clean_value(logistics.get('city'))
    processed_row['state'] = clean_value(logistics.get('state'))
    processed_row['region'] = validate_choice(logistics.get('region'), CHOICE_FIELDS['region'])
    processed_row['mode'] = validate_choice(logistics.get('mode'), CHOICE_FIELDS['mode'])

    # Organizer Info
    processed_row['organiser'] = clean_value(organizer.get('organiser'))
    processed_row['eventWebsite'] = clean_value(organizer.get('eventWebsite'))
    processed_row['organiserWebsite'] = clean_value(organizer.get('organiserWebsite'))
    processed_row['bookingLink'] = clean_value(organizer.get('bookingLink'))

    # Participation Info
    processed_row['participationType'] = validate_choice(participation.get('participationType'),
                                                         CHOICE_FIELDS['participationType'])
    processed_row['registrationCost'] = extract_registration_cost(participation.get('registrationCost'))
    processed_row['ageLimitation'] = extract_age_limit(participation.get('ageLimitation'))
    processed_row['numberOfparticipants'] = extract_numeric(participation.get('numberOfparticipants'))
    processed_row['participationCriteria'] = clean_value(participation.get('participationCriteria'))

    # Race Segments
    processed_row['swimDistance'] = extract_numeric(segments.get('swimDistance'))
    processed_row['swimType'] = validate_choice(segments.get('swimType'), CHOICE_FIELDS['swimType'])
    processed_row['cyclingDistance'] = extract_numeric(segments.get('cyclingDistance'))
    processed_row['cyclingElevationgain'] = extract_numeric(segments.get('cyclingElevationgain'))
    processed_row['runningDistance'] = extract_numeric(segments.get('runningDistance'))
    processed_row['runningSurface'] = validate_choice(segments.get('runningSurface'), CHOICE_FIELDS['runningSurface'])
    processed_row['runningElevationgain'] = extract_numeric(segments.get('runningElevationgain'))
    processed_row['runningElevationloss'] = extract_numeric(segments.get('runningElevationloss'))

    # --- DERIVED AND CALCULATED FIELDS (Logic remains the same) ---
    event_type = processed_row.get('type', '')
    if processed_row['festivalName'] and processed_row['festivalName'] not in event_type:
        processed_row['event'] = f"{processed_row['festivalName']} - {event_type}"
    else:
        processed_row['event'] = event_type

    if processed_row['date']:
        try:
            date_obj = datetime.datetime.strptime(processed_row['date'], "%d/%m/%Y")
            processed_row['month'] = date_obj.strftime("%B")
            year = date_obj.year
            processed_row['editionYear'] = str(year)
            processed_row['lastEdition'] = str(year)
        except (ValueError, TypeError):
            pass

    try:
        first_ed = int(float(processed_row.get('firstEdition') or 0))
        edition_yr = int(float(processed_row.get('editionYear') or 0))
        if first_ed > 1000 and edition_yr >= first_ed:
            processed_row['countEditions'] = str((edition_yr - first_ed) + 1)
        else:
            processed_row['countEditions'] = "1"
    except (ValueError, TypeError):
        processed_row['countEditions'] = "1"

    today = datetime.datetime.now().date()
    processed_row['registrationOpentag'] = "No"
    processed_row['eventConcludedtag'] = "No"
    if processed_row.get('lastDate'):
        try:
            if datetime.datetime.strptime(processed_row['lastDate'], "%d/%m/%Y").date() >= today:
                processed_row['registrationOpentag'] = "Yes"
        except (ValueError, TypeError):
            pass
    if processed_row.get('date'):
        try:
            if datetime.datetime.strptime(processed_row['date'], "%d/%m/%Y").date() < today:
                processed_row['eventConcludedtag'] = "Yes"
        except (ValueError, TypeError):
            pass

    return processed_row


# MODIFIED: The main function now accepts arguments from the UI
def main(output_dir_override=None, input_dir_override=None):
    """Main function to orchestrate the entire process."""
    print("--- Event Data Extraction Script (Smarter Version) ---")

    # MODIFIED: Use user-selected input directory from UI, or fall back to default
    effective_input_dir = input_dir_override if input_dir_override else DEFAULT_INPUT_FOLDER
    print(f"INFO: Reading images from: {os.path.abspath(effective_input_dir)}")

    # MODIFIED: Use user-selected output directory from UI to build the final CSV path
    if output_dir_override and not os.path.exists(output_dir_override):
        os.makedirs(output_dir_override)
        print(f"📂 Created user-specified output directory: {output_dir_override}")

    final_output_path = os.path.join(output_dir_override,
                                     OUTPUT_CSV_FILENAME) if output_dir_override else OUTPUT_CSV_FILENAME
    print(f"INFO: Output file will be saved to: {os.path.abspath(final_output_path)}")

    if not os.path.exists(effective_input_dir):
        os.makedirs(effective_input_dir)
        print(f"Created '{effective_input_dir}'. Please add images and run again.")
        return

    try:
        with open(CACHE_FILE, 'r') as f:
            processed_images = set(f.read().splitlines())
    except FileNotFoundError:
        processed_images = set()
    print(f"Found {len(processed_images)} previously processed image(s) in cache.")

    all_images = [f for f in os.listdir(effective_input_dir) if f.lower().endswith(ALLOWED_EXTENSIONS)]
    new_images = [f for f in all_images if f not in processed_images]

    if not new_images:
        print("No new images to process. Exiting.")
        return

    print(f"Found {len(new_images)} new image(s) to process.")

    new_data_rows = []
    processed_this_run = []

    for image_name in new_images:
        image_path = os.path.join(effective_input_dir, image_name)
        print(f"\nProcessing '{image_name}'...")

        raw_data = get_gemini_response(image_path)
        if not raw_data:
            print(f"  -> Skipping {image_name} due to API error or invalid response.")
            continue

        processed_row = process_image_data(raw_data)
        new_data_rows.append(processed_row)
        processed_this_run.append(image_name)
        print(f"  -> Successfully extracted data for '{image_name}'.")

    if not new_data_rows:
        print("\nNo data was successfully extracted in this run. Exiting.")
        return

    print(f"\nProcessed {len(new_data_rows)} new image(s). Appending to '{os.path.basename(final_output_path)}'...")
    df = pd.DataFrame(new_data_rows)
    df = df[CSV_HEADERS]

    if os.path.exists(final_output_path):
        df.to_csv(final_output_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df.to_csv(final_output_path, mode='w', header=True, index=False, encoding='utf-8')

    print(f"Data successfully saved to '{final_output_path}'.")

    with open(CACHE_FILE, 'a') as f:
        for image_name in processed_this_run:
            f.write(f"{image_name}\n")
    print(f"Updated cache with {len(processed_this_run)} new image name(s).")

    print("\n--- Script Finished ---")


if __name__ == '__main__':
    main()