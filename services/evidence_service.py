import urllib.parse
import xml.etree.ElementTree as ET
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Set, Tuple
import requests
from utils.source_analyzer import extract_domain, classify_source_tier
from utils.text_cleaner import detect_language

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "also", "said", "will", "new", "one", "two", "per",
    "already", "just"
}

MONTHS_AND_TIME = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "today", "yesterday"
}

CONTRADICTION_CUES = [
    "debunk", "false", "hoax", "fact check", "fake", "denies", "refutes",
    "incorrect", "misleading", "myth", "no evidence", "unproven", "fabricated",
    "disproved", "rumor", "conspiracy", "falsely", "untrue", "no permanent",
    "never built", "no colony", "no base exists", "no human settlement",
    "has not built", "no astronauts", "unfounded"
]

AUTHORITATIVE_DOMAINS = {
    "isro.gov.in", "nasa.gov", "esa.int", "who.int", "cdc.gov",
    "nature.com", "science.org", "scientificamerican.com", "spacedaily.com",
    "space.com", "phys.org", "newscientist.com", "nationalgeographic.com",
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "thehindu.com",
    "nytimes.com", "wsj.com", "washingtonpost.com", "bloomberg.com",
    "theguardian.com", "npr.org", "afp.com", "aljazeera.com", "ndtv.com",
    "theindianexpress.com", "indiatoday.in", "csis.org"
}

ACTION_SYNONYMS = {
    "build": {
        "build", "built", "building", "builds", "construct", "constructed",
        "constructing", "constructs", "establish", "established", "establishing",
        "establishes", "found", "founded", "founding", "create", "created", "creating",
        "set up", "setup"
    },
    "colonize": {
        "colonize", "colonized", "colonizing", "colonization", "settle",
        "settled", "settlement", "settling", "colonies", "colony"
    },
    "live": {
        "live", "living", "lives", "lived", "inhabit", "inhabited",
        "inhabiting", "inhabits", "reside", "resided", "residing", "resides", "staying"
    },
    "land": {
        "land", "landed", "landing", "lands", "touchdown", "touchdowns",
        "soft-landed", "soft-landing", "soft-land", "touching", "touch", "touched",
        "descent", "descend", "descended", "descending"
    },
    "operate": {
        "operate", "operational", "operating", "operates", "function",
        "functioning", "functional", "active", "maintain", "maintains", "maintaining", "maintained"
    },
    "discover": {
        "discover", "discovered", "discovering", "discovery", "found",
        "find", "detect", "detected", "detecting", "revealed", "confirmed"
    },
    "deploy": {
        "deploy", "deployed", "deploying", "deploys", "rollout", "separated",
        "ramp", "ramped", "release", "released"
    },
    "announce": {
        "announce", "announced", "announces", "announcing", "declare",
        "declared", "declares", "report", "reported"
    },
    "approve": {
        "approve", "approved", "approving", "approves", "pass", "passed",
        "passing", "sign", "signed", "authorize", "authorized"
    },
    "die": {
        "die", "died", "dying", "dies", "dead", "killed", "kill", "killing",
        "assassinated", "passed away"
    },
    "arrest": {
        "arrest", "arrested", "arresting", "arrests", "detain", "detained",
        "custody", "captured"
    },
    "corrode": {
        "corrode", "corroded", "corroding", "corrosion", "rust", "rusted",
        "degrade", "degraded", "damage", "damaged", "wear", "flaw", "flaws"
    },
    "select": {
        "select", "selected", "selecting", "selects", "choose", "chosen",
        "choosing", "pick", "picked", "recruit", "recruited"
    },
    "confirm": {
        "confirm", "confirmed", "confirms", "confirming", "certify", "certified"
    },
    "economic_trend": {
        "inflation", "steady", "rate", "gdp", "growth", "rose", "fell", "percent"
    }
}

ACTION_TO_CANONICAL = {}
for canonical, words in ACTION_SYNONYMS.items():
    for w in words:
        ACTION_TO_CANONICAL[w] = canonical

OBJECT_SYNONYMS = {
    "colony": {
        "colony", "colonies", "settlement", "settlements", "outpost",
        "outposts", "moonbase", "base", "bases", "habitat",
        "camp", "camps", "station", "stations", "facility", "facilities"
    },
    "city": {
        "city", "cities", "metropolis", "town", "towns", "municipality"
    },
    "citizen": {
        "citizen", "citizens", "people", "humans", "residents", "inhabitants",
        "settlers", "population", "civilians"
    },
    "spacecraft": {
        "spacecraft", "lander", "rover", "satellite", "probe", "orbiter",
        "module", "modules", "capsule", "rocket", "chandrayaan", "chandrayaan-3", "vikram", "pragyan",
        "gateway", "lunar gateway"
    },
    "corrosion": {
        "corrosion", "corroded", "rust", "rusting", "damage", "wear", "degradation"
    },
    "astronaut": {
        "astronaut", "astronauts", "cosmonaut", "cosmonauts", "crew",
        "humans", "personnel", "spationaut", "taikonaut"
    },
    "water": {
        "water", "ice", "ocean", "lake", "moisture", "h2o"
    },
    "bunker": {
        "bunker", "shelter", "facility", "underground"
    },
    "alien": {
        "alien", "extraterrestrial", "ufo", "et"
    },
    "inflation": {
        "inflation", "prices", "cpi"
    }
}

OBJECT_TO_CANONICAL = {}
for canonical, words in OBJECT_SYNONYMS.items():
    for w in words:
        OBJECT_TO_CANONICAL[w] = canonical

LOCATION_SYNONYMS = {
    "mars": {"mars", "martian", "red planet"},
    "moon": {"moon", "lunar", "selene"},
    "earth": {"earth", "terrestrial", "world", "globe"},
    "space": {"space", "orbit", "cosmos", "gateway"}
}

LOCATION_TO_CANONICAL = {}
for canonical, words in LOCATION_SYNONYMS.items():
    for w in words:
        LOCATION_TO_CANONICAL[w] = canonical

STEM_MAP = {
    "lunar": "moon",
    "successful": "success",
    "successfully": "success",
    "historic": "history",
    "historical": "history",
    "polar": "pole",
    "permanent": "permanently",
    "permanently": "permanently",
    "operational": "operate",
    "astronauts": "astronaut",
    "colonies": "colony",
    "settlements": "settlement",
    "landers": "lander",
    "rovers": "rover",
    "modules": "module",
    "regions": "region",
    "descent": "descend",
    "touchdown": "touchdown",
    "soft-landed": "land",
    "soft-landing": "land",
    "soft-land": "land"
}

# Ranking and Ordinal normalization
RANKING_MAP = {
    "first": 1, "1st": 1, "initial": 1, "premier": 1, "maiden": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10
}

# Antagonistic modifier pairs (if claim has A and evidence asserts B -> contradiction)
MODIFIER_CONFLICTS = {
    "permanent": {"temporary", "temporarily", "short-term", "brief", "transient", "mission"},
    "temporary": {"permanent", "permanently", "forever", "perpetual", "colony", "settlement"},
    "only": {"multiple", "several", "one of", "fourth", "4th", "second", "2nd", "third", "3rd", "one of several", "among others"},
    "sole": {"multiple", "several", "one of", "fourth", "4th", "second", "2nd", "third", "3rd", "one of several"},
    "largest": {"smaller", "smallest", "minor"},
    "smallest": {"larger", "largest", "major", "biggest"},
    "all": {"some", "few", "partial", "partially"}
}

RANKING_SCOPE_MAP = {
    # Country / nation / state
    "country": "country", "countries": "country",
    "nation": "country", "nations": "country",
    "state": "country", "states": "country",
    "power": "country", "powers": "country",
    "government": "country", "governments": "country",

    # Probe / spacecraft / vehicle / lander / rover
    "probe": "probe", "probes": "probe",
    "spacecraft": "probe", "spacecrafts": "probe",
    "craft": "probe", "crafts": "probe",
    "lander": "probe", "landers": "probe",
    "rover": "probe", "rovers": "probe",
    "satellite": "probe", "satellites": "probe",
    "module": "probe", "modules": "probe",
    "vehicle": "probe", "vehicles": "probe",
    "vessel": "probe", "vessels": "probe",
    "orbiter": "probe", "capsule": "probe",

    # Mission / program / project
    "mission": "mission", "missions": "mission",
    "expedition": "mission", "expeditions": "mission",
    "program": "mission", "programs": "mission",
    "programme": "mission", "programmes": "mission",
    "flight": "mission", "flights": "mission",
    "attempt": "mission", "attempts": "mission",
    "project": "mission", "projects": "mission",

    # Person / human / astronaut
    "person": "person", "persons": "person",
    "people": "person", "human": "person", "humans": "person",
    "man": "person", "men": "person",
    "woman": "person", "women": "person",
    "astronaut": "person", "astronauts": "person",
    "cosmonaut": "person", "cosmonauts": "person",
    "taikonaut": "person", "taikonauts": "person",
    "citizen": "person", "citizens": "person",

    # Landing / action / event
    "landing": "landing", "landings": "landing",
    "touchdown": "landing", "touchdowns": "landing"
}

SUB_LOCATION_PATTERNS = {
    "south_pole": [
        r'\bsouth\s+pole\b',
        r'\blunar\s+south\s+pole\b',
        r'\bsouth\s+polar\b',
        r'\bpolar\s+region\b',
        r'\bsouthern\s+pole\b',
        r'\bpolar\b'
    ],
    "north_pole": [
        r'\bnorth\s+pole\b',
        r'\blunar\s+north\s+pole\b',
        r'\bnorth\s+polar\b',
        r'\bnorthern\s+pole\b'
    ],
    "far_side": [
        r'\bfar\s+side\b',
        r'\bdark\s+side\b',
        r'\bfarside\b'
    ],
    "near_side": [
        r'\bnear\s+side\b',
        r'\bnearside\b'
    ],
    "equator": [
        r'\bequator\b',
        r'\bequatorial\b'
    ],
    "orbit": [
        r'\blunar\s+orbit\b',
        r'\bin\s+orbit\b',
        r'\borbital\b'
    ]
}

COUNTRY_NAMES = {
    "india", "indian", "us", "usa", "america", "american", "russia", "russian",
    "ussr", "soviet", "china", "chinese", "japan", "japanese", "israel", "israeli",
    "france", "french", "germany", "german", "uk", "britain", "british"
}

PROBE_NAMES = {
    "chandrayaan", "chandrayaan-3", "chandrayaan-2", "chandrayaan-1",
    "vikram", "pragyan", "luna", "luna-25", "apollo", "artemis", "perseverance",
    "curiosity", "chang'e", "change"
}

def extract_sub_locations(text: str) -> Set[str]:
    """
    Extracts specific sub-locations or regional identifiers (e.g. south pole, polar region, far side).
    """
    sub_locs = set()
    text_lower = text.lower()
    for sloc, pats in SUB_LOCATION_PATTERNS.items():
        for pat in pats:
            if re.search(pat, text_lower):
                sub_locs.add(sloc)
                break
    return sub_locs

def classify_entity_type(term: str) -> str:
    """Classifies entity into broad category: country, probe, mission, person, general."""
    t = term.lower().strip()
    if t in COUNTRY_NAMES or any(c in t for c in COUNTRY_NAMES):
        return "country"
    if t in ("isro", "nasa", "esa", "jaxa", "roscosmos", "cnsa"):
        return "agency"
    if t in PROBE_NAMES or any(p in t for p in PROBE_NAMES):
        return "probe"
    if t in ("mission", "project", "program", "programme"):
        return "mission"
    if t in ("astronaut", "cosmonaut", "person", "human", "citizen"):
        return "person"
    return "general"

def extract_ranking_details(text: str) -> List[Dict[str, Any]]:
    """
    Extracts structured ranking information including rank value, scope (country/probe/mission/person/landing),
    target noun, phrase, and associated sub-location.
    """
    text_lower = text.lower()
    details = []
    ord_pattern = r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)\b'
    for m in re.finditer(ord_pattern, text_lower):
        ord_word = m.group(1)
        val = RANKING_MAP.get(ord_word)
        if not val:
            continue
        start_idx = m.start()
        after_text = text_lower[start_idx:start_idx + 80]

        scope = "general"
        target_noun = ""
        sub_loc = None

        tokens = re.findall(r'[a-z\-]+', after_text)
        for w in tokens[1:6]:
            if w in RANKING_SCOPE_MAP:
                scope = RANKING_SCOPE_MAP[w]
                target_noun = w
                break

        if scope == "general":
            to_m = re.match(r'(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)\s+to\s+([a-z]+)', after_text)
            if to_m:
                act = to_m.group(1)
                if any(c in text_lower for c in COUNTRY_NAMES):
                    scope = "country"
                    target_noun = f"country to {act}"
                elif any(p in text_lower for p in PROBE_NAMES):
                    scope = "probe"
                    target_noun = f"probe to {act}"
                else:
                    scope = "action"
                    target_noun = f"to {act}"

        for sloc, pats in SUB_LOCATION_PATTERNS.items():
            for pat in pats:
                if re.search(pat, after_text):
                    sub_loc = sloc
                    break
            if sub_loc:
                break

        phrase_m = re.match(r'^(?:[a-z0-9\-]+\s+){1,6}', after_text)
        phrase = phrase_m.group(0).strip() if phrase_m else ord_word

        details.append({
            "val": val,
            "word": ord_word,
            "scope": scope,
            "target_noun": target_noun,
            "sub_location": sub_loc,
            "phrase": phrase
        })
    return details

def extract_rankings(text: str) -> Dict[int, str]:
    """
    Extracts ordinal rankings and ranking phrases from text.
    e.g. 'first country' -> {1: 'first country'}
         '4th country' -> {4: '4th country'}
         'fourth nation' -> {4: 'fourth nation'}
    """
    rankings = {}
    text_lower = text.lower()

    phrase_pattern = r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)\s+(country|nation|state|entity|probe|spacecraft|rover|lander|satellite|module|vehicle|mission|landing|touchdown|to\s+\w+)\b'
    for match in re.finditer(phrase_pattern, text_lower):
        ord_word = match.group(1)
        full_phrase = match.group(0)
        rank_val = RANKING_MAP.get(ord_word)
        if rank_val:
            rankings[rank_val] = full_phrase

    standalone_pattern = r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)\b'
    for match in re.finditer(standalone_pattern, text_lower):
        ord_word = match.group(1)
        rank_val = RANKING_MAP.get(ord_word)
        if rank_val and rank_val not in rankings:
            rankings[rank_val] = ord_word

    return rankings

PLANNED_YEAR_KEYWORDS = {
    "plan", "plans", "planned", "planning", "target", "targets", "targeted", "targeting",
    "schedule", "scheduled", "schedules", "scheduling", "aim", "aims", "aimed", "aiming",
    "slated", "expected", "projected", "forecast", "forecasted", "future", "upcoming",
    "next", "roadmap", "subsequent", "propose", "proposed", "proposes", "proposing",
    "envisage", "envisaged", "timeline"
}

PUBLICATION_YEAR_KEYWORDS = {
    "published", "publish", "publishes", "reported", "reporting", "posted", "updated",
    "copyright", "accessed", "retrieved", "dateline", "byline"
}

HISTORICAL_YEAR_KEYWORDS = {
    "prior", "predecessor", "former", "earlier", "previous", "previously"
}

def extract_temporal_metadata(text: str) -> Dict[str, Any]:
    """
    Extracts and classifies 4-digit years into distinct temporal categories:
    - occurrence_years: when the actual asserted event occurred
    - planned_years: when a future or scheduled mission/event is planned
    - publication_years: article publication or filing dates
    - historical_years: references to prior missions or historical context
    Also determines event_phase: 'completed', 'planned', 'historical', or 'unspecified'.
    """
    text_lower = text.lower()
    details = []
    occurrence_years = set()
    planned_years = set()
    publication_years = set()
    historical_years = set()
    all_years = set()

    for m in re.finditer(r'\b(19\d\d|20\d\d)\b', text):
        y_str = m.group(1)
        y_val = int(y_str)
        all_years.add(y_str)

        start_idx = m.start()
        win_start = max(0, start_idx - 60)
        win_end = min(len(text_lower), start_idx + 60)
        window = text_lower[win_start:win_end]
        tokens = set(re.findall(r'[a-z\-]+', window))

        is_planned = bool(tokens.intersection(PLANNED_YEAR_KEYWORDS)) or bool(
            re.search(r'\b(?:will|would|to be)\s+(?:launch|land|operate|deploy|send|conduct)\b', window)
        )
        is_publication = bool(tokens.intersection(PUBLICATION_YEAR_KEYWORDS)) or bool(
            re.search(rf'\b(?:copyright|\(c\)|\(reuters|\(ap|\(bbc)\b[^\.\,\;]{{0,30}}\b{y_str}\b', window)
        )
        is_historical = (y_val < 2020) and (
            bool(tokens.intersection(HISTORICAL_YEAR_KEYWORDS)) or
            bool(re.search(rf'\b(?:since|first since|back in)\s+{y_str}\b', window))
        )

        if is_planned:
            y_type = "planned"
            phase = "planned"
            planned_years.add(y_str)
        elif is_publication:
            y_type = "publication"
            phase = "publication"
            publication_years.add(y_str)
        elif is_historical:
            y_type = "historical"
            phase = "historical"
            historical_years.add(y_str)
        else:
            y_type = "occurrence"
            phase = "completed"
            occurrence_years.add(y_str)

        details.append({
            "year": y_str,
            "year_int": y_val,
            "type": y_type,
            "phase": phase,
            "window": window
        })

    if planned_years and not occurrence_years:
        overall_phase = "planned"
    elif occurrence_years:
        overall_phase = "completed"
    else:
        overall_phase = "unspecified"

    return {
        "all_years": all_years,
        "details": details,
        "occurrence_years": occurrence_years,
        "planned_years": planned_years,
        "publication_years": publication_years,
        "historical_years": historical_years,
        "event_phase": overall_phase
    }

def extract_quantities(text: str, excluded_years: Set[str] = None) -> Dict[int, str]:
    """
    Extracts numerical quantities, population/headcounts, and counts.
    Excludes:
    - 4-digit years (e.g. 1900-2099)
    - Date days associated with months (e.g. 23 in 'August 23')
    - Model/mission designations (e.g. 3 in 'Chandrayaan-3', 19 in 'COVID-19', 'Ch-3')
    - Ordinal ranking numbers (1st, 2nd, 3rd, 4th)
    """
    if excluded_years is None:
        excluded_years = set()
    quantities = {}
    text_lower = text.lower()
    months = "january|february|march|april|may|june|july|august|september|october|november|december"

    # Identify date days in text so they are not treated as headcounts/quantities
    date_days = set()
    for m in re.finditer(rf'\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b', text_lower):
        date_days.add(int(m.group(2)))
    for m in re.finditer(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})\b', text_lower):
        date_days.add(int(m.group(1)))

    for m in re.finditer(r'\b(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)\b', text):
        start_pos = m.start()
        end_pos = m.end()

        # Skip if preceded by hyphen or underscore or slash (e.g. Chandrayaan-3, COVID-19, Ch-3)
        if start_pos > 0 and text[start_pos - 1] in ('-', '_', '/'):
            continue

        raw_str = m.group(0)
        clean_str = raw_str.replace(',', '')
        try:
            val_num = float(clean_str) if '.' in clean_str else int(clean_str)
            val = int(val_num) if isinstance(val_num, float) and val_num.is_integer() else val_num
        except ValueError:
            continue

        # Skip 4-digit years
        if raw_str in excluded_years or (isinstance(val, int) and 1900 <= val <= 2099 and len(clean_str) == 4):
            continue
        # Skip date days (e.g. 23 in August 23)
        if val in date_days:
            continue
        # Skip ordinal suffix (1st, 2nd, 3rd, 4th, etc.)
        if end_pos < len(text) and text[end_pos:end_pos+2].lower() in ("st", "nd", "rd", "th"):
            continue
        # Skip hyphenated duration units (e.g. 13-month, 2-year, 5-day)
        if end_pos < len(text) and text[end_pos] == '-' and re.match(r'-(?:month|year|day|week|hour|min)', text[end_pos:].lower()):
            continue
        quantities[val] = raw_str

    word_quantities = [
        (r'\bten\s+thousand\b', 10000, '10,000'),
        (r'\bone\s+hundred\b', 100, '100'),
        (r'\bmillion\b', 1000000, 'million'),
        (r'\bbillion\b', 1000000000, 'billion')
    ]
    for pat, val, repr_str in word_quantities:
        if re.search(pat, text_lower):
            quantities[val] = repr_str

    return quantities

def extract_dates(text: str) -> Set[str]:
    """
    Extracts month-day dates, e.g. 'August 23', '23 August', 'September 15'.
    """
    dates = set()
    text_lower = text.lower()
    months = "january|february|march|april|may|june|july|august|september|october|november|december"
    for m in re.finditer(rf'\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b', text_lower):
        month = m.group(1)
        day = int(m.group(2))
        dates.add(f"{month} {day}")
    for m in re.finditer(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})\b', text_lower):
        day = int(m.group(1))
        month = m.group(2)
        dates.add(f"{month} {day}")
    return dates

MONTH_NAMES = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}

def extract_months(text: str) -> Set[str]:
    """
    Extracts standalone month names from text.
    """
    text_lower = text.lower()
    return {m for m in MONTH_NAMES if re.search(rf'\b{m}\b', text_lower)}

def extract_modifiers(text: str) -> Set[str]:
    """
    Extracts qualitative and exclusivity modifiers from text.
    """
    modifiers = set()
    text_lower = text.lower()
    if re.search(r'\b(permanent|permanently|forever|perpetual|colony|settlement)\b', text_lower):
        modifiers.add("permanent")
    if re.search(r'\b(temporary|temporarily|short-term|brief|transient|14-day|single mission)\b', text_lower):
        modifiers.add("temporary")
    if re.search(r'\b(only|sole|solely|exclusive|exclusively)\b', text_lower):
        modifiers.add("only")
    if re.search(r'\b(one of several|among others|multiple|several|various|fourth|second|third)\b', text_lower):
        modifiers.add("multiple")
    if re.search(r'\b(largest|biggest|maximum)\b', text_lower):
        modifiers.add("largest")
    if re.search(r'\b(smaller|smallest|minor)\b', text_lower):
        modifiers.add("smaller")
    if re.search(r'\b(all|every|entire|complete)\b', text_lower):
        modifiers.add("all")
    if re.search(r'\b(some|partial|partially|few)\b', text_lower):
        modifiers.add("some")
    return modifiers

def normalize_word(w: str) -> str:
    w = w.lower().strip()
    if w in STEM_MAP:
        return STEM_MAP[w]
    if w.endswith("ly") and len(w) > 4:
        base = w[:-2]
        return STEM_MAP.get(base, base)
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        return STEM_MAP.get(base, base)
    if w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        return STEM_MAP.get(base, base)
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        base = w[:-1]
        return STEM_MAP.get(base, base)
    return w

def is_reference_source(domain: str, source_name: str) -> bool:
    """Check if a source is an encyclopedic reference source (e.g. Wikipedia)."""
    d = (domain or "").lower()
    n = (source_name or "").lower()
    return "wikipedia" in d or "wikipedia" in n

def is_authoritative_source(domain: str, source_name: str) -> bool:
    """
    Check if a source belongs to an authoritative scientific, governmental, or major established news publisher.
    Wikipedia is classified as a Reference Source, not an Authoritative Source.
    """
    d = (domain or "").lower()
    n = (source_name or "").lower()
    if is_reference_source(d, n):
        return False
    if any(d.endswith(suffix) for suffix in (".gov", ".gov.in", ".gov.uk", ".gov.au", ".edu", ".ac.uk", ".ac.in", ".mil")):
        return True
    if d in AUTHORITATIVE_DOMAINS:
        return True
    if any(auth in n for auth in ("isro", "nasa", "esa", "reuters", "associated press", "bbc", "nature", "space daily", "csis", "who")):
        return True
    return False

def decompose_claim(claim_text: str, title: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Decompose input claim into discrete factual propositions.
    Extracts canonical actions, target objects, named entities, and temporal qualifiers.
    """
    raw_text = claim_text.strip()
    if title and title.strip() and title.strip().lower() not in raw_text.lower():
        full_text = f"{title.strip()}. {raw_text}"
    else:
        full_text = raw_text

    overall_months = extract_months(full_text)
    overall_years = set(re.findall(r'\b(19\d\d|20\d\d)\b', full_text))

    clauses = re.split(r'(?<!\d),(?!\d)|;|\band\b|(?<!\d)\.(?:\s+|$)', full_text)
    propositions = []
    seen_texts = set()

    for c in clauses:
        c_clean = c.strip()
        if len(c_clean) > 8 and c_clean.lower() not in seen_texts:
            seen_texts.add(c_clean.lower())
            words = [w.lower() for w in re.findall(r'[a-zA-Z0-9\-]+', c_clean)]
            norm_words = [normalize_word(w) for w in words]
            actions = set()
            for w in words:
                if w in ACTION_TO_CANONICAL:
                    actions.add(ACTION_TO_CANONICAL[w])
                elif normalize_word(w) in ACTION_TO_CANONICAL:
                    actions.add(ACTION_TO_CANONICAL[normalize_word(w)])
            objects = set()
            for w in words:
                if w in OBJECT_TO_CANONICAL:
                    objects.add(OBJECT_TO_CANONICAL[w])
                elif normalize_word(w) in OBJECT_TO_CANONICAL:
                    objects.add(OBJECT_TO_CANONICAL[normalize_word(w)])
            locations = set()
            for w in words:
                if w in LOCATION_TO_CANONICAL:
                    locations.add(LOCATION_TO_CANONICAL[w])
                elif normalize_word(w) in LOCATION_TO_CANONICAL:
                    locations.add(LOCATION_TO_CANONICAL[normalize_word(w)])
            quantities = set(re.findall(r'\b\d{1,3}(?:,\d{3})+\b|\b\d+\b', c_clean))
            entities = {norm_words[i] for i, w in enumerate(words) if len(norm_words[i]) > 2 and w not in STOPWORDS and not w.isdigit()}
            years = set(re.findall(r'\b(19\d\d|20\d\d)\b', c_clean)) or overall_years
            rankings = extract_rankings(c_clean)
            ranking_details = extract_ranking_details(c_clean)
            sub_locations = extract_sub_locations(c_clean)
            temporal_meta = extract_temporal_metadata(c_clean)
            dates = extract_dates(c_clean)
            months = extract_months(c_clean) or overall_months
            parsed_quantities = extract_quantities(c_clean, years)
            modifiers = extract_modifiers(c_clean)

            # Detect subject candidate
            subject = ""
            for raw_w in re.findall(r'[A-Za-z0-9\-]+', c_clean):
                if raw_w[0].isupper() and raw_w.lower() not in STOPWORDS:
                    subject = raw_w
                    break
            if not subject and entities:
                subject = list(entities)[0]
            subject_type = classify_entity_type(subject) if subject else "general"

            propositions.append({
                "text": c_clean,
                "subject": subject,
                "subject_type": subject_type,
                "actions": actions,
                "objects": objects,
                "locations": locations,
                "sub_locations": sub_locations,
                "quantities": quantities,
                "parsed_quantities": parsed_quantities,
                "rankings": rankings,
                "ranking_details": ranking_details,
                "years": years,
                "occurrence_years": temporal_meta["occurrence_years"],
                "planned_years": temporal_meta["planned_years"],
                "publication_years": temporal_meta["publication_years"],
                "historical_years": temporal_meta["historical_years"],
                "event_phase": temporal_meta["event_phase"],
                "temporal_details": temporal_meta["details"],
                "dates": dates,
                "months": months,
                "modifiers": modifiers,
                "entities": entities
            })

    if not propositions and raw_text:
        words = [w.lower() for w in re.findall(r'[a-zA-Z0-9\-]+', raw_text)]
        norm_words = [normalize_word(w) for w in words]
        actions = set()
        for w in words:
            if w in ACTION_TO_CANONICAL:
                actions.add(ACTION_TO_CANONICAL[w])
            elif normalize_word(w) in ACTION_TO_CANONICAL:
                actions.add(ACTION_TO_CANONICAL[normalize_word(w)])
        objects = set()
        for w in words:
            if w in OBJECT_TO_CANONICAL:
                objects.add(OBJECT_TO_CANONICAL[w])
            elif normalize_word(w) in OBJECT_TO_CANONICAL:
                objects.add(OBJECT_TO_CANONICAL[normalize_word(w)])
        locations = set()
        for w in words:
            if w in LOCATION_TO_CANONICAL:
                locations.add(LOCATION_TO_CANONICAL[w])
            elif normalize_word(w) in LOCATION_TO_CANONICAL:
                locations.add(LOCATION_TO_CANONICAL[normalize_word(w)])
        quantities = set(re.findall(r'\b\d{1,3}(?:,\d{3})+\b|\b\d+\b', raw_text))
        entities = {norm_words[i] for i, w in enumerate(words) if len(norm_words[i]) > 2 and w not in STOPWORDS and not w.isdigit()}
        years = set(re.findall(r'\b(19\d\d|20\d\d)\b', raw_text))
        rankings = extract_rankings(raw_text)
        ranking_details = extract_ranking_details(raw_text)
        sub_locations = extract_sub_locations(raw_text)
        temporal_meta = extract_temporal_metadata(raw_text)
        dates = extract_dates(raw_text)
        months = extract_months(raw_text)
        parsed_quantities = extract_quantities(raw_text, years)
        modifiers = extract_modifiers(raw_text)

        subject = ""
        for raw_w in re.findall(r'[A-Za-z0-9\-]+', raw_text):
            if raw_w[0].isupper() and raw_w.lower() not in STOPWORDS:
                subject = raw_w
                break
        if not subject and entities:
            subject = list(entities)[0]
        subject_type = classify_entity_type(subject) if subject else "general"

        propositions.append({
            "text": raw_text,
            "subject": subject,
            "subject_type": subject_type,
            "actions": actions,
            "objects": objects,
            "locations": locations,
            "sub_locations": sub_locations,
            "quantities": quantities,
            "parsed_quantities": parsed_quantities,
            "rankings": rankings,
            "ranking_details": ranking_details,
            "years": years,
            "occurrence_years": temporal_meta["occurrence_years"],
            "planned_years": temporal_meta["planned_years"],
            "publication_years": temporal_meta["publication_years"],
            "historical_years": temporal_meta["historical_years"],
            "event_phase": temporal_meta["event_phase"],
            "temporal_details": temporal_meta["details"],
            "dates": dates,
            "months": months,
            "modifiers": modifiers,
            "entities": entities
        })

    return propositions

def extract_search_query(text: str, title: Optional[str] = None) -> str:
    """
    Extract 3-5 salient search tokens targeting the core propositions.
    Prioritizes distinctive proper nouns, object nouns, and key action terms.
    """
    tokens = []

    # 1. Primary anchor from title if available and informative
    if title and title.strip():
        clean_title = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', title).strip()
        for tok in clean_title.split():
            ct = tok.strip('-').lower()
            if len(ct) >= 3 and ct not in STOPWORDS and ct not in MONTHS_AND_TIME:
                if tok not in tokens:
                    tokens.append(tok)

    # 2. Extract salient entities and high-information words from text
    if len(tokens) < 5 and text:
        words = re.findall(r'[a-zA-Z0-9\-]+', text)
        filtered = []
        for w in words:
            wl = w.strip('-').lower()
            if len(wl) >= 3 and wl not in STOPWORDS and wl not in MONTHS_AND_TIME and not wl.isdigit():
                filtered.append(w)

        def word_priority(word):
            wl = word.lower()
            if wl in OBJECT_TO_CANONICAL:
                return 3  # Distinctive core object (colony, astronaut, spacecraft)
            if word[0].isupper():
                return 2  # Named entity / Proper noun
            if wl in ACTION_TO_CANONICAL:
                return 1  # Core action verb
            return 0

        sorted_words = sorted(filtered, key=word_priority, reverse=True)
        for w in sorted_words:
            if w.lower() not in [t.lower() for t in tokens]:
                tokens.append(w)
                if len(tokens) >= 5:
                    break

    # 3. Fallback to leading content words if needed
    if not tokens and text:
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        tokens = [w for w in clean_text.split() if len(w) > 2 and w not in STOPWORDS][:5]

    return " ".join(tokens[:5])

def search_google_news(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    """
    Query Google News RSS for live public news reports without fabricating data.
    Parses clean headlines, direct source domain, and description snippet.
    """
    if not query.strip():
        return []
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.text)
        items = root.findall("./channel/item")
        results = []
        for item in items[:max_results]:
            raw_title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            source_elem = item.find("source")
            source_name = source_elem.text.strip() if (source_elem is not None and source_elem.text) else ""
            source_url = source_elem.attrib.get("url", "") if source_elem is not None else ""

            # Extract source name from title format "Headline - Source"
            clean_title = raw_title
            if " - " in raw_title and not source_name:
                parts = raw_title.rsplit(" - ", 1)
                clean_title = parts[0].strip()
                source_name = parts[1].strip()

            # Clean snippet from description HTML
            raw_desc = item.findtext("description", default="").strip()
            clean_snippet = re.sub(r'<[^>]+>', ' ', raw_desc)
            clean_snippet = re.sub(r'\s+', ' ', clean_snippet).strip()
            if not clean_snippet or len(clean_snippet) < 15:
                clean_snippet = f"Reported by {source_name or 'verified news outlet'} on {pub_date}" if pub_date else f"Reported by {source_name or 'verified news outlet'}"

            # Domain extraction
            domain = ""
            if source_url:
                domain = extract_domain(source_url)
            if not domain and link:
                domain = extract_domain(link)

            is_auth = is_authoritative_source(domain, source_name)
            is_ref = is_reference_source(domain, source_name)

            if clean_title:
                results.append({
                    "title": clean_title,
                    "url": link,
                    "source": source_name or "Public News Source",
                    "domain": domain or (source_name.lower().replace(" ", "") + ".com" if source_name else "news.google.com"),
                    "pub_date": pub_date,
                    "snippet": clean_snippet,
                    "is_authoritative": is_auth,
                    "is_reference_source": is_ref
                })
        return results
    except Exception:
        return []

def search_wikipedia(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Query Wikipedia public API for encyclopedic verification.
    Wikipedia is classified as a Reference Source, not an Authoritative Source.
    """
    if not query.strip():
        return []
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json"
    headers = {"User-Agent": "FakeNewsDetectorSystem/2.0 (student-academic-project)"}
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return []
        data = resp.json()
        search_items = data.get("query", {}).get("search", [])
        results = []
        for item in search_items[:max_results]:
            title = item.get("title", "")
            raw_snippet = item.get("snippet", "")
            clean_snippet = re.sub(r'<[^>]+>', '', raw_snippet).strip()
            page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            results.append({
                "title": f"Wikipedia: {title}",
                "url": page_url,
                "source": "Wikipedia",
                "domain": "wikipedia.org",
                "pub_date": "Live Reference",
                "snippet": clean_snippet,
                "is_authoritative": False,
                "is_reference_source": True
            })
        return results
    except Exception:
        return []

def evaluate_evidence_relevance(
    claim_text: str,
    propositions: List[Dict[str, Any]],
    item: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Computes fine-grained semantic entailment between an evidence report and claim propositions.
    Evaluates:
    1. Entity agreement
    2. Action/event agreement
    3. Object agreement
    4. Location agreement (prevents Mars vs Moon confusion)
    5. Ranking and superlative agreement (first vs fourth, only vs multiple)
    6. Quantity and headcount agreement (10,000 vs 100)
    7. Temporal and date agreement (2025 vs 2027, exact date mismatch)
    8. Qualitative modifier agreement (permanent vs temporary)
    9. Refutation/contradiction detection
    """
    title_ev = item.get("title", "")
    snippet_ev = item.get("snippet", "")
    combined_ev = (title_ev + " " + snippet_ev).lower()
    ev_raw_words = [w.lower() for w in re.findall(r'[a-zA-Z0-9\-]+', combined_ev)]
    ev_words = {normalize_word(w) for w in ev_raw_words}
    ev_actions = set()
    for w in ev_raw_words:
        if w in ACTION_TO_CANONICAL:
            ev_actions.add(ACTION_TO_CANONICAL[w])
        elif normalize_word(w) in ACTION_TO_CANONICAL:
            ev_actions.add(ACTION_TO_CANONICAL[normalize_word(w)])
    ev_objects = set()
    for w in ev_raw_words:
        if w in OBJECT_TO_CANONICAL:
            ev_objects.add(OBJECT_TO_CANONICAL[w])
        elif normalize_word(w) in OBJECT_TO_CANONICAL:
            ev_objects.add(OBJECT_TO_CANONICAL[normalize_word(w)])
    ev_locations = set()
    for w in ev_raw_words:
        if w in LOCATION_TO_CANONICAL:
            ev_locations.add(LOCATION_TO_CANONICAL[w])
        elif normalize_word(w) in LOCATION_TO_CANONICAL:
            ev_locations.add(LOCATION_TO_CANONICAL[normalize_word(w)])
    ev_temporal = extract_temporal_metadata(combined_ev)
    ev_years = ev_temporal["all_years"]
    ev_occ_years = ev_temporal["occurrence_years"]
    ev_planned_years = ev_temporal["planned_years"]
    ev_pub_years = ev_temporal["publication_years"]
    ev_hist_years = ev_temporal["historical_years"]
    ev_event_phase = ev_temporal["event_phase"]
    ev_dates = extract_dates(combined_ev)
    ev_months = extract_months(combined_ev)
    ev_quantities = set(re.findall(r'\b\d{1,3}(?:,\d{3})+\b|\b\d+\b', combined_ev))
    ev_parsed_quantities = extract_quantities(combined_ev, ev_years)
    ev_rankings = extract_rankings(combined_ev)
    ev_sub_locations = extract_sub_locations(combined_ev)
    ev_ranking_details = extract_ranking_details(combined_ev)
    ev_modifiers = extract_modifiers(combined_ev)

    # Landing on celestial bodies (moon, mars) inherently involves a spacecraft
    if "land" in ev_actions and ("moon" in ev_locations or "mars" in ev_locations):
        ev_objects.add("spacecraft")

    # 1. Direct Refutation / Debunking Cue Check
    has_refutation = any(cue in combined_ev for cue in CONTRADICTION_CUES)
    claim_objects = set().union(*[p.get("objects", set()) for p in propositions if p.get("objects")])
    claim_entities = set().union(*[p.get("entities", set()) for p in propositions])
    claim_locations = set().union(*[p.get("locations", set()) for p in propositions if p.get("locations")])

    if has_refutation and (any(obj in combined_ev for obj in claim_objects) or len(claim_entities.intersection(ev_words)) >= 2):
        if not (claim_locations and ev_locations and not claim_locations.intersection(ev_locations)):
            return {
                "stance": "CONTRADICTING",
                "semantic_relevance": 0.95,
                "matched_proposition": "Direct Refutation",
                "conflict_type": "Debunking report",
                "claim_attribute": "Claim assertion",
                "evidence_attribute": "Refutation / Debunking",
                "reason": "Evidence explicitly contains refutation or debunking language directly refuting the asserted claim or entity."
            }

    # 2. Check Entailment & Attribute Contradictions against each Proposition
    best_stance = "IRRELEVANT"
    best_score = 0.0
    best_prop = None
    best_reason = "Evidence is not relevant to the factual claim."
    best_conflict_type = None
    best_claim_attr = None
    best_ev_attr = None

    for idx, prop in enumerate(propositions):
        prop_label = f"P{idx+1}: {prop['text']}"
        action_match = bool(prop.get("actions", set()).intersection(ev_actions)) if prop.get("actions") else True
        object_match = bool(prop.get("objects", set()).intersection(ev_objects)) if prop.get("objects") else True
        overlap = prop.get("entities", set()).intersection(ev_words)
        overlap_ratio = len(overlap) / max(1, len(prop.get("entities", set())))

        # Location check: celestial bodies or distinct sites
        location_conflict = False
        if prop.get("locations") and ev_locations:
            if not prop["locations"].intersection(ev_locations):
                location_conflict = True

        # Sub-location conflict check: e.g. north pole vs south pole
        prop_sublocs = prop.get("sub_locations", set())
        if prop_sublocs and ev_sub_locations:
            if not prop_sublocs.intersection(ev_sub_locations):
                location_conflict = True

        # Month check: different calendar months indicate different time periods
        month_conflict = bool(prop.get("months") and ev_months and not prop["months"].intersection(ev_months))

        # Relevance scoring components
        action_score = 0.30 if action_match else 0.0
        object_score = 0.30 if object_match else 0.0
        location_score = -0.30 if location_conflict else (0.15 if (prop.get("locations") and not location_conflict) else 0.05)
        entity_score = 0.20 * overlap_ratio
        total_score = max(0.0, action_score + object_score + location_score + entity_score)

        # ── Step A: Check Direct Attribute Contradictions (When discussing same core event) ──
        is_same_event = (not location_conflict) and (action_match or object_match) and (overlap_ratio >= 0.20 or len(overlap) >= 2)
        if is_same_event:
            # A1: Ranking Contradiction (e.g. first vs fourth / 4th)
            prop_ranks = prop.get("rankings", {})
            prop_rank_details = prop.get("ranking_details", [])
            if prop_ranks and ev_rankings:
                # 1. Compare structured ranking details by scope
                for c_rank in prop_rank_details:
                    c_val = c_rank["val"]
                    c_scope = c_rank["scope"]
                    for e_rank in ev_ranking_details:
                        e_val = e_rank["val"]
                        e_scope = e_rank["scope"]

                        scopes_comparable = False
                        if c_scope == e_scope and c_scope != "general":
                            scopes_comparable = True
                        elif c_scope in ("country", "general") and e_scope == "country":
                            scopes_comparable = True
                        elif c_scope == "country" and e_scope in ("general", "action") and (prop.get("subject_type") == "country" or any(cn in ev_words for cn in COUNTRY_NAMES)):
                            scopes_comparable = True

                        if scopes_comparable and c_val != e_val:
                            claim_desc = c_rank["phrase"]
                            ev_desc = e_rank["phrase"]
                            return {
                                "stance": "CONTRADICTING",
                                "semantic_relevance": 0.96,
                                "matched_proposition": prop_label,
                                "conflict_type": "Ranking contradiction",
                                "claim_attribute": claim_desc,
                                "evidence_attribute": ev_desc,
                                "reason": f"Evidence directly contradicts claim ranking: claim asserts '{claim_desc}' (rank {c_val}) but reliable evidence confirms '{ev_desc}' (rank {e_val})."
                            }

                # 2. Fallback check for ranking contradiction
                claim_rank_val = list(prop_ranks.keys())[0]
                ev_rank_val = list(ev_rankings.keys())[0]
                if claim_rank_val != ev_rank_val:
                    c_phrase = prop_ranks[claim_rank_val].lower()
                    e_phrase = ev_rankings[ev_rank_val].lower()
                    c_is_probe_or_mission = any(w in c_phrase for w in ("probe", "spacecraft", "mission", "lander", "rover"))
                    e_is_probe_or_mission = any(w in e_phrase for w in ("probe", "spacecraft", "mission", "lander", "rover"))
                    c_is_country = any(w in c_phrase for w in ("country", "nation", "state")) or prop.get("subject_type") == "country"
                    e_is_country = any(w in e_phrase for w in ("country", "nation", "state"))

                    if (c_is_country and e_is_country) or (c_is_probe_or_mission == e_is_probe_or_mission and not (c_is_country ^ e_is_country)):
                        claim_desc = prop_ranks[claim_rank_val]
                        ev_desc = ev_rankings[ev_rank_val]
                        return {
                            "stance": "CONTRADICTING",
                            "semantic_relevance": 0.96,
                            "matched_proposition": prop_label,
                            "conflict_type": "Ranking contradiction",
                            "claim_attribute": claim_desc,
                            "evidence_attribute": ev_desc,
                            "reason": f"Evidence directly contradicts claim ranking: claim asserts '{claim_desc}' (rank {claim_rank_val}) but reliable evidence confirms '{ev_desc}' (rank {ev_rank_val})."
                        }

            # A2: Exclusivity / Multi-country Contradiction (e.g. first/only vs multiple / one of several)
            if (prop_ranks.get(1) or "only" in prop.get("modifiers", set())) and ("multiple" in ev_modifiers or re.search(r'\b(one of several|among others|fourth|second|third|multiple countries)\b', combined_ev)):
                claim_desc = prop_ranks.get(1) or "only / sole"
                ev_desc = "one of several / multiple"
                return {
                    "stance": "CONTRADICTING",
                    "semantic_relevance": 0.95,
                    "matched_proposition": prop_label,
                    "conflict_type": "Ranking contradiction",
                    "claim_attribute": claim_desc,
                    "evidence_attribute": ev_desc,
                    "reason": f"Evidence directly contradicts claim exclusivity/ranking: claim asserts '{claim_desc}' but evidence confirms '{ev_desc}'."
                }

            # A3: Quantity Contradiction (e.g. 10,000 vs 100)
            same_period = True
            if prop.get("months"):
                if not ev_months or not prop["months"].intersection(ev_months):
                    same_period = False

            if same_period and not month_conflict:
                prop_qtys = prop.get("parsed_quantities", {})
                if prop_qtys and ev_parsed_quantities:
                    claim_q_val = list(prop_qtys.keys())[0]
                    ev_q_val = list(ev_parsed_quantities.keys())[0]
                    if claim_q_val != ev_q_val:
                        claim_q_desc = prop_qtys[claim_q_val]
                        ev_q_desc = ev_parsed_quantities[ev_q_val]
                        return {
                            "stance": "CONTRADICTING",
                            "semantic_relevance": 0.95,
                            "matched_proposition": prop_label,
                            "conflict_type": "Quantity contradiction",
                            "claim_attribute": str(claim_q_desc),
                            "evidence_attribute": str(ev_q_desc),
                            "reason": f"Evidence directly contradicts the stated quantity: claim asserts '{claim_q_desc}' but evidence reports '{ev_q_desc}'."
                        }

            # A4: Temporal Date Contradiction (e.g. September 15 vs August 23)
            if prop.get("dates") and ev_dates:
                if not prop["dates"].intersection(ev_dates):
                    claim_d = list(prop["dates"])[0]
                    ev_d = list(ev_dates)[0]
                    return {
                        "stance": "CONTRADICTING",
                        "semantic_relevance": 0.95,
                        "matched_proposition": prop_label,
                        "conflict_type": "Temporal contradiction",
                        "claim_attribute": claim_d,
                        "evidence_attribute": ev_d,
                        "reason": f"Evidence directly contradicts the event date: claim asserts '{claim_d}' but evidence reports '{ev_d}'."
                    }

            # A5: Temporal Year Contradiction (Strict Same-Event Occurrence Date Conflict)
            prop_occ_years = prop.get("occurrence_years", set()) or prop.get("years", set())
            if prop_occ_years and ev_occ_years:
                conflicting_occ = ev_occ_years - prop_occ_years
                if not prop_occ_years.intersection(ev_occ_years) and conflicting_occ:
                    same_event_action = action_match and object_match and not location_conflict
                    same_phase = (prop.get("event_phase", "completed") == ev_event_phase) or (ev_event_phase != "planned")
                    if same_event_action and same_phase:
                        claim_y = list(prop_occ_years)[0]
                        ev_y = list(conflicting_occ)[0]
                        return {
                            "stance": "CONTRADICTING",
                            "semantic_relevance": 0.95,
                            "matched_proposition": prop_label,
                            "conflict_type": "Temporal contradiction",
                            "claim_attribute": claim_y,
                            "evidence_attribute": ev_y,
                            "reason": f"Evidence directly contradicts the event occurrence year: claim asserts '{claim_y}' but reliable evidence confirms '{ev_y}' for the same event."
                        }

            # A6: Modifier Contradiction (e.g. permanent vs temporary)
            for mod in prop.get("modifiers", set()):
                conflicts = MODIFIER_CONFLICTS.get(mod, set())
                if conflicts.intersection(ev_modifiers) or any(re.search(rf'\b{re.escape(c)}\b', combined_ev) for c in conflicts):
                    matched_conflict = list(conflicts.intersection(ev_modifiers))[0] if conflicts.intersection(ev_modifiers) else list(conflicts)[0]
                    return {
                        "stance": "CONTRADICTING",
                        "semantic_relevance": 0.95,
                        "matched_proposition": prop_label,
                        "conflict_type": "Modifier contradiction",
                        "claim_attribute": mod,
                        "evidence_attribute": matched_conflict,
                        "reason": f"Evidence directly contradicts claim modifier: claim asserts '{mod}' but evidence confirms '{matched_conflict}'."
                    }

        # ── Step B: Strict Entailment for SUPPORTING ──
        # All required specific attributes MUST be satisfied; otherwise demoted to CONTEXTUAL
        prop_ranks = prop.get("rankings", {})
        prop_rank_details = prop.get("ranking_details", [])
        ranking_satisfied = True
        ranking_demotion_reason = ""

        if prop_ranks:
            if not bool(set(prop_ranks.keys()).intersection(ev_rankings.keys())):
                ranking_satisfied = False
                ranking_demotion_reason = f"Corroborates the general event ({prop.get('subject') or 'topic'}), but lacks factual confirmation of claim ranking ('{list(prop_ranks.values())[0]}')."
            else:
                # Ranking number matched. Now verify SCOPE, SUBJECT TYPE, and SUB-LOCATION entailment!
                has_entailing_rank = False
                for c_rank in prop_rank_details:
                    c_val = c_rank["val"]
                    c_scope = c_rank["scope"]
                    c_subloc = c_rank.get("sub_location")

                    for e_rank in ev_ranking_details:
                        if e_rank["val"] == c_val:
                            e_scope = e_rank["scope"]
                            e_subloc = e_rank.get("sub_location")

                            # 1. Scope check:
                            # 'country' cannot be entailed by 'probe', 'mission', or 'landing'
                            # 'probe' cannot be entailed by 'mission'
                            scope_matches = False
                            if c_scope == e_scope:
                                scope_matches = True
                            elif c_scope in ("general", ""):
                                scope_matches = True
                            elif c_scope == "probe" and e_scope in ("probe", "spacecraft", "lander", "rover"):
                                scope_matches = True
                            elif c_scope == "country" and e_scope in ("country", "nation"):
                                scope_matches = True

                            # 2. Geographic / Sub-location check:
                            # If claim is a global ranking (c_subloc is None) but evidence is restricted
                            # to a sub-location (e.g. e_subloc == "south_pole"), the regional first
                            # does NOT entail the global first!
                            loc_scope_matches = True
                            if c_subloc is None and e_subloc is not None:
                                loc_scope_matches = False
                            elif c_subloc is not None and e_subloc != c_subloc:
                                loc_scope_matches = False

                            if scope_matches and loc_scope_matches:
                                has_entailing_rank = True
                                break
                            else:
                                if not scope_matches:
                                    ranking_demotion_reason = (
                                        f"Corroborates the general event ({prop.get('subject') or 'topic'}), "
                                        f"but claim asserts {c_scope}-level ranking ('{c_rank['phrase']}'), "
                                        f"whereas evidence refers to {e_scope}-level ranking ('{e_rank['phrase']}')."
                                    )
                                elif not loc_scope_matches:
                                    loc_name = list(prop.get('locations', ['the Moon']))[0].capitalize()
                                    ranking_demotion_reason = (
                                        f"Corroborates the general event ({prop.get('subject') or 'topic'}), "
                                        f"but claim asserts global ranking on {loc_name}, "
                                        f"whereas evidence specifies a location-restricted achievement ('{e_rank['phrase']}')."
                                    )
                    if has_entailing_rank:
                        break

                if not has_entailing_rank:
                    ranking_satisfied = False

        prop_qtys = prop.get("parsed_quantities", {})
        quantity_satisfied = True
        if prop_qtys:
            if not bool(set(prop_qtys.keys()).intersection(ev_parsed_quantities.keys())):
                quantity_satisfied = False

        date_satisfied = True
        if prop.get("dates") and ev_dates:
            if not bool(prop["dates"].intersection(ev_dates)):
                date_satisfied = False

        modifier_satisfied = True
        if "permanent" in prop.get("modifiers", set()):
            if not ("permanent" in ev_modifiers or "colony" in ev_objects):
                modifier_satisfied = False

        # Temporal year conflict only exists when evidence reports a conflicting OCCURRENCE year
        # for the same event action (never for planned years, publication years, or historical references)
        year_conflict = False
        prop_occ_years = prop.get("occurrence_years", set()) or prop.get("years", set())
        if prop_occ_years and ev_occ_years:
            if not prop_occ_years.intersection(ev_occ_years):
                if action_match and object_match and not location_conflict and ev_event_phase != "planned":
                    year_conflict = True

        non_year_prop_qty = prop.get("quantities", set()) - prop.get("years", set())
        non_year_ev_qty = ev_quantities - ev_years
        raw_qty_conflict = False
        if non_year_prop_qty and non_year_ev_qty:
            if not non_year_prop_qty.intersection(non_year_ev_qty):
                raw_qty_conflict = True

        if location_conflict or month_conflict or not ranking_satisfied or not quantity_satisfied or not date_satisfied or not modifier_satisfied or year_conflict or raw_qty_conflict:
            entails = False
        elif prop.get("actions") and prop.get("objects"):
            entails = action_match and object_match and (overlap_ratio >= 0.25 or len(overlap) >= 2)
        elif prop.get("actions"):
            entails = action_match and (overlap_ratio >= 0.35 or len(overlap) >= 2)
        elif prop.get("objects"):
            entails = object_match and (overlap_ratio >= 0.35 or len(overlap) >= 2)
        else:
            entails = (overlap_ratio >= 0.50 or len(overlap) >= 3)

        has_matching_year = bool(prop_occ_years and prop_occ_years.intersection(ev_occ_years or ev_years))
        detail_score = 0.10 if (not year_conflict and has_matching_year) else (0.05 if not year_conflict else 0.0)
        total_score = max(0.0, total_score + detail_score)

        if entails and total_score > best_score:
            best_score = total_score
            best_stance = "SUPPORTING"
            best_prop = prop_label
            best_reason = f"Directly entails proposition P{idx+1} (Action: {prop.get('actions')}, Object: {prop.get('objects')}, Entities: {len(overlap)} matched)."
        elif not entails and total_score > best_score:
            best_score = total_score
            if total_score >= 0.15:
                best_stance = "CONTEXTUAL"
                if location_conflict:
                    best_reason = f"Mentions related entities, but refers to a different location ({list(ev_locations)} vs {list(prop.get('locations', set()))})."
                elif year_conflict and prop_occ_years and ev_occ_years:
                    best_reason = f"Corroborates the general topic, but refers to a conflicting event occurrence year ({list(ev_occ_years)[0]} vs {list(prop_occ_years)[0]})."
                elif not ranking_satisfied and prop_ranks:
                    best_reason = ranking_demotion_reason or f"Corroborates the general event ({prop.get('subject') or 'topic'}), but lacks factual confirmation of claim ranking ('{list(prop_ranks.values())[0]}')."
                elif not quantity_satisfied and prop_qtys:
                    best_reason = f"Corroborates the general event, but lacks factual confirmation of stated quantity ('{list(prop_qtys.values())[0]}')."
                elif not date_satisfied and prop.get("dates"):
                    best_reason = f"Corroborates the general event, but lacks factual confirmation of specific date ('{list(prop['dates'])[0]}')."
                elif not action_match or not object_match:
                    best_reason = f"Discusses related topic/entity, but action ({ev_actions or 'none'}) or object ({ev_objects or 'none'}) does not entail proposition P{idx+1}."
                else:
                    best_reason = f"Contextual reference relating to the topic, but lacks direct factual confirmation of proposition P{idx+1}."
            else:
                best_stance = "IRRELEVANT"
                best_reason = "Evidence is not relevant to the factual claim."

    return {
        "stance": best_stance,
        "semantic_relevance": round(best_score, 2),
        "matched_proposition": best_prop or "None",
        "conflict_type": best_conflict_type,
        "claim_attribute": best_claim_attr,
        "evidence_attribute": best_ev_attr,
        "reason": best_reason
    }

def classify_evidence_stance(claim_text: str, evidence_item: Dict[str, Any], title: Optional[str] = None) -> str:
    """
    Classify whether an evidence report supports, contradicts, or provides context for the claim.
    Requires semantic and factual agreement rather than merely sharing a single superficial keyword.
    """
    propositions = decompose_claim(claim_text, title)
    res = evaluate_evidence_relevance(claim_text, propositions, evidence_item)
    evidence_item["semantic_relevance"] = res["semantic_relevance"]
    evidence_item["matched_proposition"] = res["matched_proposition"]
    evidence_item["reason"] = res["reason"]
    return res["stance"]

def verify_claim_evidence(claim_text: str, title: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for real evidence verification.
    Decomposes claims into factual propositions and performs semantic entailment scoring.
    Deduplicates sources and counts independent publishers.
    """
    propositions = decompose_claim(claim_text, title)
    lang_info = detect_language(claim_text)
    query = extract_search_query(claim_text, title)
    if not query:
        return {
            "is_available": True,
            "status": "INSUFFICIENT",
            "evidence_strength": "INSUFFICIENT",
            "message": "Input was too short or uninformative to form a valid search query.",
            "query": "",
            "detected_language": lang_info,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "contextual_evidence": [],
            "irrelevant_evidence": [],
            "total_sources_found": 0,
            "propositions": [p["text"] for p in propositions],
            "debug_trace": {
                "input_claim": claim_text,
                "detected_language": lang_info,
                "claim_propositions": [f"P{i+1}: {p['text']}" for i, p in enumerate(propositions)],
                "search_query": "",
                "evaluations": []
            }
        }

    news_results = search_google_news(query, max_results=6)
    wiki_results = search_wikipedia(query, max_results=3)
    all_results = news_results + wiki_results

    if not all_results:
        short_query = " ".join(query.split()[:3])
        if short_query and short_query != query:
            all_results = search_google_news(short_query, max_results=4)

    if not all_results:
        return {
            "is_available": False,
            "status": "INSUFFICIENT",
            "evidence_strength": "INSUFFICIENT",
            "message": "External evidence verification unavailable. No matching independent public reports found.",
            "query": query,
            "detected_language": lang_info,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "contextual_evidence": [],
            "irrelevant_evidence": [],
            "total_sources_found": 0,
            "propositions": [p["text"] for p in propositions],
            "debug_trace": {
                "input_claim": claim_text,
                "detected_language": lang_info,
                "claim_propositions": [f"P{i+1}: {p['text']}" for i, p in enumerate(propositions)],
                "search_query": query,
                "evaluations": []
            }
        }

    # Stance classification and semantic relevance evaluation
    supporting_raw = []
    contradicting_raw = []
    contextual_raw = []
    irrelevant_raw = []
    all_evaluations = []

    for item in all_results:
        tier_info = classify_source_tier(item.get("domain", ""), item.get("source", ""))
        item["tier"] = tier_info["tier"]
        item["tier_label"] = tier_info["tier_label"]
        item["tier_name"] = tier_info["tier_name"]
        item["tier_badge"] = tier_info["badge"]
        item["is_authoritative"] = tier_info["is_authoritative"]
        item["is_reference_source"] = tier_info["is_reference"]

        eval_res = evaluate_evidence_relevance(claim_text, propositions, item)
        stance = eval_res["stance"]
        item["stance"] = stance
        item["semantic_relevance"] = eval_res["semantic_relevance"]
        item["matched_proposition"] = eval_res["matched_proposition"]
        item["reason"] = eval_res["reason"]
        item["conflict_type"] = eval_res.get("conflict_type")
        item["claim_attribute"] = eval_res.get("claim_attribute")
        item["evidence_attribute"] = eval_res.get("evidence_attribute")

        all_evaluations.append({
            "source": item.get("source"),
            "domain": item.get("domain"),
            "source_claim": item.get("title"),
            "source_summary": item.get("snippet"),
            "semantic_relevance": eval_res["semantic_relevance"],
            "proposition_match": eval_res["matched_proposition"],
            "entailment_result": "Direct Entailment" if stance == "SUPPORTING" else ("Direct Refutation" if stance == "CONTRADICTING" else "No Entailment"),
            "source_reliability": tier_info["tier_name"],
            "classification": stance,
            "reason": eval_res["reason"],
            "conflict_type": eval_res.get("conflict_type"),
            "claim_attribute": eval_res.get("claim_attribute"),
            "evidence_attribute": eval_res.get("evidence_attribute")
        })

        if stance == "CONTRADICTING":
            contradicting_raw.append(item)
        elif stance == "SUPPORTING":
            supporting_raw.append(item)
        elif stance == "CONTEXTUAL":
            contextual_raw.append(item)
        else:
            irrelevant_raw.append(item)

    # Deduplicate sources by publisher domain / normalized source name
    seen_supp_publishers = set()
    supporting = []
    for item in supporting_raw:
        pub_key = (item.get("domain") or item.get("source", "")).lower()
        if pub_key not in seen_supp_publishers:
            seen_supp_publishers.add(pub_key)
            supporting.append(item)
        else:
            contextual_raw.append(item)

    seen_cont_publishers = set()
    contradicting = []
    for item in contradicting_raw:
        pub_key = (item.get("domain") or item.get("source", "")).lower()
        if pub_key not in seen_cont_publishers:
            seen_cont_publishers.add(pub_key)
            contradicting.append(item)
        else:
            contextual_raw.append(item)

    # Deduplicate contextual references
    seen_ctx = set()
    contextual = []
    for item in contextual_raw:
        item_key = (item.get("url") or item.get("title", "")).lower()
        if item_key not in seen_ctx:
            seen_ctx.add(item_key)
            contextual.append(item)

    # Prioritize authoritative sources at top of each list
    supporting.sort(key=lambda x: not x.get("is_authoritative", False))
    contradicting.sort(key=lambda x: not x.get("is_authoritative", False))

    supp_count = len(supporting)
    cont_count = len(contradicting)

    # Proposition coverage check for compound claims
    props_supported = list({item["matched_proposition"] for item in supporting if item.get("matched_proposition") and item["matched_proposition"] != "None"})
    central_props = [p for p in propositions if p.get("actions") and p.get("objects")]
    unique_central_tuples = {(frozenset(p["actions"]), frozenset(p["objects"])) for p in central_props}
    supported_tuples = set()
    for idx, p in enumerate(propositions):
        prop_label = f"P{idx+1}: {p['text']}"
        if any(item.get("matched_proposition") == prop_label for item in supporting):
            if p.get("actions") and p.get("objects"):
                supported_tuples.add((frozenset(p["actions"]), frozenset(p["objects"])))

    is_partially_supported = (len(unique_central_tuples) >= 2) and (len(supported_tuples) > 0) and (len(supported_tuples) < len(unique_central_tuples))

    # Evidence Strength considering tier quality and independent domains
    reliable_supp = [s for s in supporting if s.get("tier") in (1, 2) or s.get("is_authoritative")]
    reliable_supp_count = len(reliable_supp)

    if cont_count > 0 and supp_count > 0:
        status = "Conflicting"
    elif cont_count > 0:
        status = "Contradicting"
    elif reliable_supp_count >= 2:
        status = "STRONG"
    elif reliable_supp_count == 1 or supp_count >= 1:
        status = "AVAILABLE"
    else:
        status = "INSUFFICIENT"

    debug_trace = {
        "input_claim": claim_text,
        "detected_language": lang_info,
        "claim_propositions": [f"P{i+1}: {p['text']} [Subject: {p.get('subject')}, Actions: {list(p.get('actions'))}, Objects: {list(p.get('objects'))}, Locations: {list(p.get('locations'))}]" for i, p in enumerate(propositions)],
        "search_query": query,
        "evaluations": all_evaluations
    }

    return {
        "is_available": True,
        "status": status,
        "evidence_strength": status,
        "message": f"Retrieved {len(all_results)} public sources ({supp_count} supporting public sources, {cont_count} contradicting).",
        "query": query,
        "detected_language": lang_info,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "contextual_evidence": contextual,
        "irrelevant_evidence": irrelevant_raw,
        "total_sources_found": len(all_results),
        "propositions": [p["text"] for p in propositions],
        "props_supported": list(props_supported),
        "is_partially_supported": is_partially_supported,
        "debug_trace": debug_trace
    }
