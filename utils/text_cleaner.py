import re
from typing import Dict, Any, List

def clean_text(text: str) -> str:
    """
    Clean and preprocess input text exactly as done during model training.
    Removes URLs and non-alphabetic characters, then converts to lowercase.
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def detect_category(text: str) -> str:
    """
    Detect news category based on keyword frequency in the text.
    """
    text_lower = str(text).lower()
    categories = {
        "Politics": ["government", "minister", "election", "parliament", "president", "policy", "senate", "congress", "democrat", "republican"],
        "Sports": ["cricket", "football", "match", "player", "tournament", "olympics", "championship", "fifa", "nba", "score"],
        "Technology": ["ai", "artificial intelligence", "software", "technology", "computer", "google", "apple", "microsoft", "cyber", "hardware", "algorithm"],
        "Business": ["market", "stock", "economy", "bank", "investment", "finance", "revenue", "inflation", "trade", "shares"],
        "Entertainment": ["movie", "actor", "actress", "film", "music", "celebrity", "hollywood", "bollywood", "cinema", "award"],
    }

    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for word in keywords if re.search(rf'\b{re.escape(word)}\b', text_lower))
        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=scores.get)
    return "General News"

def get_text_stats(text: str) -> Dict[str, int]:
    """
    Compute basic text statistics: word count, sentence count, reading time in minutes.
    """
    text_str = str(text).strip()
    words = text_str.split()
    word_count = len(words)
    sentences = [s for s in re.split(r'[.!?]+', text_str) if s.strip()]
    sentence_count = len(sentences)
    reading_time = max(1, round(word_count / 200)) if word_count > 0 else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "reading_time": reading_time
    }

SENSATIONAL_KEYWORDS = [
    "shocking", "unbelievable", "mind-blowing", "bombshell", "exposed",
    "secret revealed", "they don't want you to know", "miracle cure",
    "conspiracy", "covert", "panic", "horrifying", "apocalyptic",
    "total hoax", "hidden truth", "insane revelation"
]

def analyze_sensationalism(text: str) -> Dict[str, Any]:
    """
    Analyzes actual textual signals for sensationalism without hardcoded claims.
    Calculates uppercase ratio, exclamation mark frequency, and sensational phrase presence.
    """
    text_str = str(text)
    if not text_str.strip():
        return {
            "is_sensational": False,
            "sensational_words": [],
            "exclamation_count": 0,
            "uppercase_ratio": 0.0,
            "warning_signals": []
        }

    words = text_str.split()
    total_words = len(words)
    uppercase_words = [w for w in words if len(w) > 1 and w.isupper()]
    uppercase_ratio = (len(uppercase_words) / total_words) if total_words > 0 else 0.0

    exclamation_count = text_str.count("!")

    text_lower = text_str.lower()
    found_keywords = [kw for kw in SENSATIONAL_KEYWORDS if kw in text_lower]

    sentences = [s for s in re.split(r'[.!?]+', text_str) if s.strip()]
    sentence_count = len(sentences)

    warning_signals = []
    if uppercase_ratio > 0.18:
        warning_signals.append(f"High uppercase word frequency ({uppercase_ratio:.0%} of words capitalized)")
    if ("!!" in text_str) or (exclamation_count >= 3 and (sentence_count <= 4 or (exclamation_count / max(1, sentence_count)) >= 0.25)):
        warning_signals.append(f"Excessive exclamation marks ({exclamation_count} detected)")
    if found_keywords:
        warning_signals.append(f"Sensationalist buzzwords detected: {', '.join(found_keywords[:3])}")

    is_sensational = bool(found_keywords) or (uppercase_ratio > 0.18 and exclamation_count >= 3) or (exclamation_count >= 5 and total_words < 60)

    return {
        "is_sensational": is_sensational,
        "sensational_words": found_keywords,
        "exclamation_count": exclamation_count,
        "uppercase_ratio": uppercase_ratio,
        "warning_signals": warning_signals
    }

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ar": "Arabic"
}

def detect_language(text: str) -> Dict[str, Any]:
    """
    Detect the language of input text supporting English, Hindi, Spanish, French, German, and Arabic.
    Uses script recognition for Hindi (Devanagari) and Arabic, and lexical frequency scoring for European languages.
    """
    text_str = str(text).strip()
    if not text_str:
        return {"code": "en", "name": "English", "confidence": 1.0}

    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text_str))
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text_str))
    total_letters = len(re.findall(r'[^\s\d\W]', text_str)) or 1

    if devanagari_chars / total_letters > 0.20:
        conf = round(min(1.0, devanagari_chars / total_letters + 0.3), 2)
        return {"code": "hi", "name": "Hindi", "lang_code": "HI", "language": "Hindi", "confidence": conf}

    if arabic_chars / total_letters > 0.20:
        conf = round(min(1.0, arabic_chars / total_letters + 0.3), 2)
        return {"code": "ar", "name": "Arabic", "lang_code": "AR", "language": "Arabic", "confidence": conf}

    words = [w.lower() for w in re.findall(r'[a-zA-Z\u00C0-\u00FF]+', text_str)]
    if not words:
        return {"code": "en", "name": "English", "lang_code": "EN", "language": "English", "confidence": 0.8}

    scores = {
        "es": 0,
        "fr": 0,
        "de": 0,
        "en": 0
    }

    spanish_tokens = {"el", "la", "de", "en", "los", "las", "por", "con", "para", "una", "un", "que", "noticias", "gobierno", "este", "esta", "segun", "sobre", "del", "al"}
    french_tokens = {"le", "la", "les", "des", "en", "du", "un", "une", "pour", "dans", "qui", "sur", "est", "avec", "selon", "au", "par", "ont", "ce", "eau", "mars", "traces"}
    german_tokens = {"der", "die", "das", "und", "in", "den", "von", "zu", "mit", "sich", "des", "auf", "fuer", "ist", "im", "nicht", "eine", "einer", "dem"}
    english_tokens = {"the", "and", "is", "in", "to", "of", "that", "with", "for", "on", "was", "by", "as", "at", "from", "has", "have", "said"}

    for w in words:
        if w in spanish_tokens:
            scores["es"] += 1
        if w in french_tokens:
            scores["fr"] += 1
        if w in german_tokens:
            scores["de"] += 1
        if w in english_tokens:
            scores["en"] += 1

    # Spanish distinctive markers (inverted marks, tilde-n, distinct accents)
    if any(c in text_str for c in "áíóúñ¿¡"):
        scores["es"] += 2
    # French distinctive markers (grave/circumflex/trema accents, cedilla, and apostrophe elisions)
    if any(c in text_str for c in "èêëàâçîïôùûœ"):
        scores["fr"] += 2
    if any(elision in text_str.lower() for elision in ["d'", "l'", "c'", "qu'", "n'", "s'", "j'"]):
        scores["fr"] += 2
    # German distinctive markers
    if any(c in text_str for c in "äöüß"):
        scores["de"] += 2

    best_lang = max(scores, key=scores.get)
    if scores[best_lang] > 0:
        conf = min(0.98, max(0.65, scores[best_lang] / len(words) + 0.4))
        return {
            "code": best_lang,
            "name": SUPPORTED_LANGUAGES[best_lang],
            "lang_code": best_lang.upper(),
            "language": SUPPORTED_LANGUAGES[best_lang],
            "confidence": round(conf, 2)
        }

    return {"code": "en", "name": "English", "lang_code": "EN", "language": "English", "confidence": 0.85}
