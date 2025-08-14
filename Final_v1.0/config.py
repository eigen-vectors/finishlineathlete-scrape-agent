# config.py
# This file contains the static configuration for the Analyst Agent.

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- API Keys & Model Configuration ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_KEY_1 = os.getenv("MISTRAL_API_KEY_1")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
CSE_ID = os.getenv("CSE_ID")
MISTRAL_MODEL = "mistral-large-latest"

# --- File & Execution Configuration ---
RACE_INPUT_FILE = "races.json"
OUTPUT_DIR = "outputs"
APP_VERSION = "v62.4"
CRAWL_CACHE_DIR = "crawl_cache"
KNOWLEDGE_CACHE_DIR = "knowledge_cache"
TOP_N_URLS_TO_PROCESS = 4
MAX_SEARCH_RESULTS = 10
MAX_SUBPAGES_PER_SITE = 5
MAX_CONCURRENT_CRAWLERS = 5
MAX_RETRIES = 3
DEBUG = True

# --- NEW: Fields to default to a blank value and NOT send to the LLM ---
DEFAULT_BLANK_FIELDS = [
    'imageURL', 'raceVideo', 'scenic', 'swimRoutemap', 'cyclingRoutemap',
    'runRoutemap', 'difficultyLevel', 'country', 'user_id', 'femaleParticpation',
    'jellyFishRelated', 'primaryKey', 'latitude', 'longitude', 'organiserRating',
    'approvalStatus', 'nextEdition'
]

# --- NEW: Fields with strict choice-based options for the LLM ---
CHOICE_OPTIONS = {
    "participationType": ["Individual", "Relay", "Group"], "mode": ["Virtual", "On-Ground"],
    "runningSurface": ["Road", "Trail", "Track", "Road + Trail"],
    "runningCourseType": ["Single Loop", "Multiple Loop", "Out and Back", "Point to Point"],
    "region": ["West India", "Central and East India", "North India", "South India", "Nepal", "Bhutan", "Sri Lanka"],
    "runningElevation": ["Flat", "Rolling", "Hilly", "Skyrunning"],
    "type": ["Triathlon", "Aquabike", "Aquathlon", "Duathlon", "Run", "Cycling", "Swimathon"],
    "swimType": ["Lake", "Beach", "River", "Pool"],
    "swimCoursetype": ["Single Loop", "Multiple Loops", "Out and Back", "Point to Point"],
    "cyclingElevation": ["Flat", "Rolling", "Hilly"],
    "cycleCoursetype": ["Single Loop", "Multiple Loops", "Out and Back", "Point to Point"],
    "triathlonType": ["Super Sprint", "Sprint Distance", "Olympic Distance", "Half Iron(70.3)", "Iron Distance (140.6)", "Ultra Distance"],
    "standardTag": ["Standard", "Non Standard"], "approvalStatus": ["Approved", "Pending Approval"],
    "restrictedTraffic": ["Yes", "No"], "jellyFishRelated": ["Yes", "No"]
}

# --- Definitive Schema for TRIATHLON events ---
TRIATHLON_SCHEMA = [
    'event', 'festivalName', 'imageURL', 'raceVideo', 'type', 'date', 'city', 'organiser',
    'participationType', 'firstEdition', 'lastEdition', 'countEditions', 'mode', 'raceAccredition',
    'theme', 'numberOfparticipants', 'startTime', 'scenic', 'registrationCost', 'ageLimitation',
    'eventWebsite', 'organiserWebsite', 'bookingLink', 'newsCoverage', 'lastDate', 'participationCriteria',
    'refundPolicy', 'swimDistance', 'swimType', 'swimmingLocation', 'waterTemperature', 'swimCoursetype',
    'swimCutoff', 'swimRoutemap', 'cyclingDistance', 'cyclingElevation', 'cyclingSurface',
    'cyclingElevationgain', 'cycleCoursetype', 'cycleCutoff', 'cyclingRoutemap', 'runningDistance',
    'runningElevation', 'runningSurface', 'runningElevationgain', 'runningElevationloss',
    'runningCoursetype', 'runCutoff', 'runRoutemap', 'organiserRating', 'triathlonType',
    'standardTag', 'region', 'approvalStatus', 'difficultyLevel', 'month', 'primaryKey',
    'latitude', 'longitude', 'country', 'editionYear', 'aidStations', 'restrictedTraffic',
    'user_id', 'femaleParticpation', 'jellyFishRelated'
]

# --- Definitive Schema for RUNNING & TRAIL RUNNING events ---
RUNNING_SCHEMA = [
    'event', 'festivalName', 'imageURL', 'raceVideo', 'type', 'date', 'city', 'organiser',
    'participationType', 'firstEdition', 'lastEdition', 'countEditions', 'mode', 'raceAccredition',
    'theme', 'numberOfparticipants', 'startTime', 'scenic', 'registrationCost', 'ageLimitation',
    'eventWebsite', 'organiserWebsite', 'bookingLink', 'newsCoverage', 'lastDate', 'participationCriteria',
    'refundPolicy', 'runningDistance', 'runningElevation', 'runningSurface',
    'runningElevationgain', 'runningElevationloss', 'runningCoursetype', 'runCutoff', 'runRoutemap',
    'organiserRating', 'region',
    'approvalStatus', 'difficultyLevel', 'month', 'primaryKey', 'latitude', 'longitude', 'country',
    'editionYear', 'aidStations', 'restrictedTraffic',
    'user_id'
]

# --- Definitive Schema for SWIMMING events ---
SWIMMING_SCHEMA = [
    'event', 'festivalName', 'imageURL', 'raceVideo', 'type', 'date', 'city', 'organiser',
    'participationType', 'firstEdition', 'lastEdition', 'countEditions', 'mode', 'raceAccredition',
    'theme', 'numberOfparticipants', 'startTime', 'scenic', 'registrationCost', 'ageLimitation',
    'eventWebsite', 'organiserWebsite', 'bookingLink', 'newsCoverage', 'lastDate', 'participationCriteria',
    'refundPolicy', 'swimDistance', 'swimType', 'swimmingLocation', 'waterTemperature', 'swimCoursetype',
    'swimCutoff', 'swimRoutemap', 'organiserRating', 'standardTag', 'registrationOpentag', 'eventConcludedtag',
    'state', 'region', 'approvalStatus', 'nextEdition', 'difficultyLevel', 'month', 'editionYear',
    'aidStations', 'restrictedTraffic', 'user_id', 'femaleParticpation', 'jellyFishRelated',
    'primaryKey', 'latitude', 'longitude', 'country'
]

# --- Crawling & Preprocessing Rules ---
BLACKLISTED_DOMAINS = [
    "facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com", "pinterest.com",
    "youtube.com", "tiktok.com", "indiamart.com", "allevents.in", "wikipedia.org", "about.com",
    "worldsmarathons.com", "triathlon-database.com", "triathlon.org", "strava.com", "podcasts.apple.com", "racingtheplanetstore.com",
    "aims-worldrunning.org/calendar"
]

NEWS_DOMAINS = ["news", "times", "express", "herald", "chronicle", "tribune"]
RELEVANT_SUBPAGE_KEYWORDS = ["result", "participant", "detail", "info", "course", "race", "register", "schedule", "faq", "rules"]