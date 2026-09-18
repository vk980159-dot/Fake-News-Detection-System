import urllib.parse
import re
from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from utils.source_analyzer import extract_domain, is_safe_url

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

def is_valid_url(url: str) -> bool:
    """Check if the provided string is a syntactically valid and safe HTTP/HTTPS URL."""
    if not url or not isinstance(url, str):
        return False
    if not is_safe_url(url):
        return False
    try:
        parsed = urllib.parse.urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False

def extract_from_url(url: str) -> Dict[str, Any]:
    """
    Extract article content, title, authors, publication date, and domain from a URL.
    Attempts extraction via newspaper3k, falling back to BeautifulSoup if needed.
    """
    url = url.strip() if url else ""
    if not is_valid_url(url):
        return {
            "success": False,
            "error": "The provided URL is invalid or blocked for security reasons (only public HTTP/HTTPS URLs permitted).",
            "url": url,
            "domain": "",
            "title": "",
            "text": "",
            "author": "",
            "publish_date": ""
        }

    domain = extract_domain(url)

    # Strategy 1: Attempt using newspaper3k
    try:
        from newspaper import Article
        article = Article(url, headers={"User-Agent": USER_AGENT})
        article.download()
        article.parse()

        title = article.title.strip() if article.title else ""
        text = article.text.strip() if article.text else ""
        authors = ", ".join(article.authors) if article.authors else ""
        publish_date = str(article.publish_date) if article.publish_date else ""

        if text and len(text.split()) >= 15:
            return {
                "success": True,
                "error": None,
                "url": url,
                "domain": domain,
                "title": title or "Untitled Article",
                "text": text,
                "author": authors if authors else "Not Available",
                "publish_date": publish_date if publish_date else "Not Available"
            }
    except Exception:
        pass # Fall through to BeautifulSoup fallback

    # Strategy 2: Fallback with requests + BeautifulSoup
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if resp.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to retrieve webpage (HTTP {resp.status_code}). Website may be blocking automated access.",
                "url": url,
                "domain": domain,
                "title": "",
                "text": "",
                "author": "",
                "publish_date": ""
            }

        soup = BeautifulSoup(resp.text, "html.parser")

        # Title extraction
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Author extraction
        author = ""
        meta_author = soup.find("meta", attrs={"name": re.compile(r"author", re.I)}) or \
                      soup.find("meta", attrs={"property": re.compile(r"author", re.I)})
        if meta_author and meta_author.get("content"):
            author = meta_author["content"].strip()

        # Publish date extraction
        publish_date = ""
        meta_date = soup.find("meta", attrs={"property": re.compile(r"published_time|pubdate", re.I)}) or \
                    soup.find("meta", attrs={"name": re.compile(r"date|pubdate", re.I)})
        if meta_date and meta_date.get("content"):
            publish_date = meta_date["content"].strip()

        # Text extraction from paragraphs
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 30]
        text = "\n\n".join(paragraphs)

        if not text or len(text.split()) < 15:
            return {
                "success": False,
                "error": "Extracted text was too brief or empty. The page may require JavaScript or have anti-scraping protections.",
                "url": url,
                "domain": domain,
                "title": title,
                "text": text,
                "author": author,
                "publish_date": publish_date
            }

        return {
            "success": True,
            "error": None,
            "url": url,
            "domain": domain,
            "title": title or "Untitled Article",
            "text": text,
            "author": author if author else "Not Available",
            "publish_date": publish_date if publish_date else "Not Available"
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Network error connecting to URL: {str(e)}",
            "url": url,
            "domain": domain,
            "title": "",
            "text": "",
            "author": "",
            "publish_date": ""
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error during article parsing: {str(e)}",
            "url": url,
            "domain": domain,
            "title": "",
            "text": "",
            "author": "",
            "publish_date": ""
        }

def extract_from_text(text: str) -> Dict[str, Any]:
    """Process direct news text input."""
    clean_val = str(text).strip()
    lines = [l.strip() for l in clean_val.split("\n") if l.strip()]
    title = lines[0] if lines else "Direct Text Submission"
    if len(title) > 120:
        title = title[:120] + "..."

    return {
        "success": bool(clean_val),
        "error": None if clean_val else "Input text is empty.",
        "url": "",
        "domain": "Direct Input",
        "title": title,
        "text": clean_val,
        "author": "Not Specified",
        "publish_date": "Not Specified"
    }

def extract_from_claim(claim: str) -> Dict[str, Any]:
    """Process plain claim or fact-checking assertion."""
    clean_val = str(claim).strip()
    return {
        "success": bool(clean_val),
        "error": None if clean_val else "Claim statement is empty.",
        "url": "",
        "domain": "Direct Claim",
        "title": clean_val if len(clean_val) < 140 else clean_val[:140] + "...",
        "text": clean_val,
        "author": "User Claim",
        "publish_date": "Not Specified"
    }
