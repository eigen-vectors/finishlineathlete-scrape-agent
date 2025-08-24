# schemas.py
# This file contains the static schemas, choice options, and rule-based lists for the agent.

DEFAULT_BLANK_FIELDS = [
    'imageURL', 'raceVideo', 'scenic', 'swimRoutemap', 'cyclingRoutemap',
    'runRoutemap', 'difficultyLevel', 'user_id', 'femaleParticpation',
    'jellyFishRelated', 'primaryKey', 'latitude', 'longitude', 'organiserRating',
    'approvalStatus', 'nextEdition', 'raceAccredition', 'theme',
    'festivalName'
]

INFERABLE_FIELDS = [
    'country', 'region', 'waterTemperature', 'cyclingElevation', 'cyclingSurface',
    'cycleCoursetype', 'runningElevation', 'runningSurface', 'runningCoursetype'
]

CHOICE_OPTIONS = {
    "participationType": ["Individual", "Relay", "Group"], "mode": ["Virtual", "On-Ground"],
    "runningSurface": ["Road", "Trail", "Track", "Road + Trail"],
    # UPDATED: Changed to plural "Loops"
    "runningCourseType": ["Single Loops", "Multiple Loops", "Out and Back", "Point to Point"],
    "region": ["West India", "Central and East India", "North India", "South India", "Nepal", "Bhutan", "Sri Lanka"],
    "runningElevation": ["Flat", "Rolling", "Hilly", "Skyrunning"],
    "type": ["Triathlon", "Aquabike", "Aquathlon", "Duathlon", "Run", "Cycling", "Swimathon"],
    "swimType": ["Lake", "Beach", "River", "Pool"],
    "swimCoursetype": ["Single Loop", "Multiple Loops", "Out and Back", "Point to Point"],
    "cyclingElevation": ["Flat", "Rolling", "Hilly"],
    "cycleCoursetype": ["Single Loop", "Multiple Loops", "Out and Back", "Point to Point"],
    "triathlonType": ["Super Sprint", "Sprint Distance", "Olympic Distance", "Half Iron(70.3)", "Iron Distance (140.6)", "Ultra Distance"],
    "standardTag": ["Standard", "Non Standard"],
    "approvalStatus": ["Approved", "Pending Approval"],
    "restrictedTraffic": ["Yes", "No"],
    "jellyFishRelated": ["Yes", "No"]
}

TRIATHLON_SCHEMA = ['event', 'festivalName', 'imageURL', 'raceVideo', 'type', 'date', 'city', 'organiser', 'participationType', 'firstEdition', 'lastEdition', 'countEditions', 'mode', 'raceAccredition', 'theme', 'numberOfparticipants', 'startTime', 'scenic', 'registrationCost', 'ageLimitation', 'eventWebsite', 'organiserWebsite', 'bookingLink', 'newsCoverage', 'lastDate', 'participationCriteria', 'refundPolicy', 'swimDistance', 'swimType', 'swimmingLocation', 'waterTemperature', 'swimCoursetype', 'swimCutoff', 'swimRoutemap', 'cyclingDistance', 'cyclingElevation', 'cyclingSurface', 'cyclingElevationgain', 'cycleCoursetype', 'cycleCutoff', 'cyclingRoutemap', 'runningDistance', 'runningElevation', 'runningSurface', 'runningElevationgain', 'runningElevationloss', 'runningCoursetype', 'runCutoff', 'runRoutemap', 'organiserRating', 'triathlonType', 'standardTag', 'region', 'approvalStatus', 'difficultyLevel', 'month', 'primaryKey', 'latitude', 'longitude', 'country', 'editionYear', 'aidStations', 'restrictedTraffic', 'user_id', 'femaleParticpation', 'jellyFishRelated']
RUNNING_SCHEMA = ['event', 'festivalName', 'imageURL', 'raceVideo', 'type', 'date', 'city', 'organiser', 'participationType', 'firstEdition', 'lastEdition', 'countEditions', 'mode', 'raceAccredition', 'theme', 'numberOfparticipants', 'startTime', 'scenic', 'registrationCost', 'ageLimitation', 'eventWebsite', 'organiserWebsite', 'bookingLink', 'newsCoverage', 'lastDate', 'participationCriteria', 'refundPolicy', 'runningDistance', 'runningElevation', 'runningSurface', 'runningElevationgain', 'runningElevationloss', 'runningCoursetype', 'runCutoff', 'runRoutemap', 'organiserRating', 'region', 'approvalStatus', 'difficultyLevel', 'month', 'primaryKey', 'latitude', 'longitude', 'country', 'editionYear', 'aidStations', 'restrictedTraffic', 'user_id']
SWIMMING_SCHEMA = ['event', 'festivalName', 'imageURL', 'raceVideo', 'type', 'date', 'city', 'organiser', 'participationType', 'firstEdition', 'lastEdition', 'countEditions', 'mode', 'raceAccredition', 'theme', 'numberOfparticipants', 'startTime', 'scenic', 'registrationCost', 'ageLimitation', 'eventWebsite', 'organiserWebsite', 'bookingLink', 'newsCoverage', 'lastDate', 'participationCriteria', 'refundPolicy', 'swimDistance', 'swimType', 'swimmingLocation', 'waterTemperature', 'swimCoursetype', 'swimCutoff', 'swimRoutemap', 'organiserRating', 'standardTag', 'registrationOpentag', 'eventConcludedtag', 'state', 'region', 'approvalStatus', 'nextEdition', 'difficultyLevel', 'month', 'editionYear', 'aidStations', 'restrictedTraffic', 'user_id', 'femaleParticpation', 'jellyFishRelated', 'primaryKey', 'latitude', 'longitude', 'country']
DUATHLON_SCHEMA = ['event', 'festivalName', 'imageURL', 'raceVideo', 'type', 'date', 'city', 'organiser', 'participationType', 'firstEdition', 'lastEdition', 'countEditions', 'mode', 'raceAccredition', 'theme', 'numberOfparticipants', 'startTime', 'scenic', 'registrationCost', 'ageLimitation', 'eventWebsite', 'organiserWebsite', 'bookingLink', 'newsCoverage', 'lastDate', 'participationCriteria', 'refundPolicy', 'cyclingDistance', 'cyclingElevation', 'cyclingSurface', 'cyclingElevationgain', 'cycleCoursetype', 'cycleCutoff', 'cyclingRoutemap', 'runningDistance', 'runningElevation', 'runningSurface', 'runningElevationgain', 'runningElevationloss', 'runningCoursetype', 'runCutoff', 'runRoutemap', 'organiserRating', 'standardTag', 'region', 'approvalStatus', 'difficultyLevel', 'month', 'primaryKey', 'latitude', 'longitude', 'country', 'editionYear', 'aidStations', 'restrictedTraffic', 'user_id']
BLACKLISTED_DOMAINS = ["facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com", "pinterest.com", "youtube.com", "tiktok.com", "indiamart.com", "allevents.in", "wikipedia.org", "about.com", "worldsmarathons.com", "triathlon-database.com", "triathlon.org", "strava.com", "podcasts.apple.com", "racingtheplanetstore.com", "aims-worldrunning.org/calendar"]