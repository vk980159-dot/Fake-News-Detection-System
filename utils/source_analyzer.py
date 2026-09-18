import urllib.parse
from typing import Dict, Any, Optional

TIER_1_DOMAINS = {
    "isro.gov.in", "nasa.gov", "esa.int", "who.int", "cdc.gov", "nih.gov",
    "nature.com", "science.org", "scientificamerican.com", "nationalgeographic.com",
    "phys.org", "csis.org"
}

TIER_2_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "thehindu.com",
    "nytimes.com", "wsj.com", "washingtonpost.com", "bloomberg.com",
    "theguardian.com", "npr.org", "afp.com", "aljazeera.com", "ndtv.com",
    "theindianexpress.com", "indiatoday.in", "space.com", "spacedaily.com",
    "newscientist.com", "cnn.com", "dw.com", "france24.com", "timesofindia.indiatimes.com"
}

WELL_KNOWN_REPUTABLE_DOMAINS = TIER_1_DOMAINS.union(TIER_2_DOMAINS)

NON_DOMAIN_TYPES = {
    "direct input", "direct claim", "direct image upload", "image ocr",
    "unspecified", "not specified", "unspecified / direct input", ""
}

INVALID_META_STRINGS = {
    "none", "unknown", "n/a", "not specified", "not available",
    "image ocr", "user claim", "unknown author", "unknown date"
}

BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "0", "169.254.169.254"}

def is_safe_url(url: str) -> bool:
    """
    Validate URL to protect against SSRF and private IP access.
    Only allows http/https and blocks localhost/private/loopback ranges.
    """
    if not url or not isinstance(url, str):
        return False
    url_clean = url.strip()
    if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
        return False
    try:
        parsed = urllib.parse.urlparse(url_clean)
        host = (parsed.hostname or "").lower()
        if not host or host in BLOCKED_HOSTS:
            return False
        if host.startswith("127.") or host.startswith("10.") or host.startswith("192.168."):
            return False
        if host.startswith("172."):
            parts = host.split(".")
            if len(parts) >= 2 and parts[1].isdigit() and 16 <= int(parts[1]) <= 31:
                return False
        if "." not in host:
            return False
        return True
    except Exception:
        return False

def classify_source_tier(domain: str, source_name: str = "") -> Dict[str, Any]:
    """
    Classify a source into 4 standardized reliability tiers:
    - Tier 1: Government, official agencies, scientific institutions, universities
    - Tier 2: Established international news organizations and major reputable newspapers
    - Tier 3: Other identifiable general public publishers
    - Tier 4: Unknown / unverified / low-quality sources
    - Reference: Encyclopedic references (Wikipedia is a Reference Source, not Authoritative)
    """
    d = (domain or "").lower()
    n = (source_name or "").lower()

    if "wikipedia" in d or "wikipedia" in n:
        return {
            "tier": 0,
            "tier_label": "Reference Source",
            "tier_name": "Reference Source (Wikipedia)",
            "badge": "📚 Reference Source",
            "reliability_level": "Reference",
            "is_authoritative": False,
            "is_reference": True,
            "is_reference_source": True
        }

    # Tier 1 checks
    if any(d.endswith(suffix) for suffix in (".gov", ".gov.in", ".gov.uk", ".gov.au", ".edu", ".ac.uk", ".ac.in", ".mil", ".int")):
        return {
            "tier": 1,
            "tier_label": "Tier 1",
            "tier_name": "Tier 1: Government / Scientific Authority",
            "badge": "⭐ Tier 1",
            "reliability_level": "High",
            "is_authoritative": True,
            "is_reference": False
        }
    if d in TIER_1_DOMAINS or any(auth in n for auth in ("isro", "nasa", "esa", "who", "cdc", "nature", "science journal")):
        return {
            "tier": 1,
            "tier_label": "Tier 1",
            "tier_name": "Tier 1: Government / Scientific Authority",
            "badge": "⭐ Tier 1",
            "reliability_level": "High",
            "is_authoritative": True,
            "is_reference": False
        }

    # Tier 2 checks
    if d in TIER_2_DOMAINS or any(org in n for org in ("reuters", "associated press", "bbc", "afp", "al jazeera", "the hindu", "new york times")):
        return {
            "tier": 2,
            "tier_label": "Tier 2",
            "tier_name": "Tier 2: Established News Organization",
            "badge": "📰 Tier 2",
            "reliability_level": "High",
            "is_authoritative": False,
            "is_reference": False
        }

    # Tier 3 checks
    if d and "." in d and d not in NON_DOMAIN_TYPES:
        return {
            "tier": 3,
            "tier_label": "Tier 3",
            "tier_name": "Tier 3: General Public Publisher",
            "badge": "🌐 Tier 3",
            "reliability_level": "Medium",
            "is_authoritative": False,
            "is_reference": False
        }

    # Tier 4 checks
    return {
        "tier": 4,
        "tier_label": "Tier 4",
        "tier_name": "Tier 4: Unverified / Direct Input",
        "badge": "⚠️ Tier 4",
        "reliability_level": "Low",
        "is_authoritative": False,
        "is_reference": False
    }

def extract_domain(url_or_domain: str) -> str:
    """Extract clean domain name from URL or raw domain string."""
    if not url_or_domain:
        return ""
    url_str = url_or_domain.strip()
    if url_str.lower() in NON_DOMAIN_TYPES:
        return ""
    if not (url_str.startswith("http://") or url_str.startswith("https://")):
        url_str = "http://" + url_str
    try:
        parsed = urllib.parse.urlparse(url_str)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        # Must contain at least one dot to be a valid domain name
        if "." not in domain:
            return ""
        return domain
    except Exception:
        return ""

def analyze_source(
    url: Optional[str] = None,
    domain: Optional[str] = None,
    author: Optional[str] = None,
    publish_date: Optional[str] = None,
    has_evidence: bool = False
) -> Dict[str, Any]:
    """
    Analyze source reliability and metadata completeness using neutral, non-prejudicial standards.
    Distinguishes direct user submissions from published articles with verifiable metadata.
    """
    input_domain_raw = (domain or "").strip()
    is_direct_submission = (
        not url and (input_domain_raw.lower() in NON_DOMAIN_TYPES or "." not in input_domain_raw)
    )

    if is_direct_submission:
        display_domain = "Direct Input"
        if input_domain_raw.lower() == "direct claim":
            display_domain = "Direct Claim"
        elif "image" in input_domain_raw.lower() or "ocr" in input_domain_raw.lower():
            display_domain = "Direct Image Upload"

        return {
            "domain": display_domain,
            "has_domain": False,
            "author": "Not Specified",
            "publish_date": "Not Specified",
            "metadata_status": "Not Available",
            "reliability_level": "Low",
            "reliability_notes": "Direct submission; source metadata could not be independently verified.",
            "tier": 4,
            "tier_label": "Tier 4",
            "tier_name": "Tier 4: Unverified / Direct Input",
            "tier_badge": "⚠️ Tier 4",
            "is_authoritative": False,
            "is_reference": False,
            "evidence_available": has_evidence
        }

    # URL / Published Article Domain extraction
    detected_domain = extract_domain(url) if url else extract_domain(input_domain_raw)
    has_domain = bool(detected_domain)
    tier_info = classify_source_tier(detected_domain)

    # Validate Author
    author_clean = str(author).strip() if author else ""
    has_author = bool(author_clean and author_clean.lower() not in INVALID_META_STRINGS)
    display_author = author_clean if has_author else "Not Available"

    # Validate Publication Date
    date_clean = str(publish_date).strip() if publish_date else ""
    has_date = bool(date_clean and date_clean.lower() not in INVALID_META_STRINGS)
    display_date = date_clean if has_date else "Not Available"

    # Metadata completeness status:
    # "Complete" ONLY if actual author AND publish_date are present alongside domain; otherwise "Partial"
    if has_domain and has_author and has_date:
        metadata_status = "Complete"
    elif has_domain:
        metadata_status = "Partial"
    else:
        metadata_status = "Not Available"

    # Reliability assessment - neutral wording without inventing scores
    if has_domain:
        if tier_info["tier"] == 1:
            reliability_level = "High"
            reliability_notes = f"Official governmental/scientific domain ({detected_domain})."
        elif tier_info["tier"] == 2:
            reliability_level = "High"
            reliability_notes = f"Recognized established news reporting domain ({detected_domain})."
        elif tier_info["tier"] == 0:
            reliability_level = "Medium"
            reliability_notes = "Encyclopedic reference source (Wikipedia)."
        else:
            reliability_level = "Medium" if (has_author or has_date) else "Available"
            reliability_notes = f"Source domain identified ({detected_domain})."
    else:
        reliability_level = "Low"
        reliability_notes = "Source information could not be independently verified."

    return {
        "domain": detected_domain if detected_domain else "Direct Input",
        "has_domain": has_domain,
        "author": display_author,
        "publish_date": display_date,
        "metadata_status": metadata_status,
        "reliability_level": reliability_level,
        "reliability_notes": reliability_notes,
        "tier": tier_info["tier"],
        "tier_label": tier_info["tier_label"],
        "tier_name": tier_info["tier_name"],
        "tier_badge": tier_info["badge"],
        "is_authoritative": tier_info["is_authoritative"],
        "is_reference": tier_info["is_reference"],
        "evidence_available": has_evidence
    }
