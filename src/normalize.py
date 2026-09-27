"""
Text normalization and preprocessing routines for business entity resolution.
"""

import re
import unicodedata
from typing import Dict, Any, List, Set

# Pre-compiled regular expressions for speed
RE_PUNCT = re.compile(r"[^a-z0-9\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_ALNUM_ONLY = re.compile(r"[^a-z0-9]")

# Common legal business suffixes (longest first for proper regex matching)
LEGAL_SUFFIX_PATTERNS = [
    r"\bprivate\s+limited\b",
    r"\bpvt\s+ltd\b",
    r"\bpublic\s+limited\s+company\b",
    r"\bplc\b",
    r"\bllc\b",
    r"\bincorporated\b",
    r"\binc\b",
    r"\bcorporation\b",
    r"\bcorp\b",
    r"\blimited\b",
    r"\bltd\b",
    r"\bgmbh\b",
    r"\bs\.?a\.?s\.?\b",
    r"\bs\.?a\.?\b",
    r"\bsarl\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\benterprises?\b",
    r"\bservices?\b",
    r"\bsolutions?\b",
    r"\bgroup\b",
    r"\bholdings?\b",
    r"\btrust\b",
    r"\bassociates?\b",
    r"\bpartners?\b",
]

COMPILED_LEGAL_SUFFIXES = [re.compile(p, re.IGNORECASE) for p in LEGAL_SUFFIX_PATTERNS]

# Address token standardizations
ADDRESS_REPLACEMENTS = {
    " street ": " st ",
    " road ": " rd ",
    " avenue ": " ave ",
    " boulevard ": " blvd ",
    " drive ": " dr ",
    " lane ": " ln ",
    " court ": " ct ",
    " parkway ": " pkwy ",
    " highway ": " hwy ",
    " circle ": " cir ",
    " apartment ": " apt ",
    " suite ": " ste ",
    " floor ": " fl ",
    " building ": " bldg ",
    " sector ": " sec ",
    " near ": " nr ",
    " opposite ": " opp ",
}


def is_missing(val: Any) -> bool:
    """Check if value is null, NaN, None, or empty string."""
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in {"", "nan", "none", "null", "<na>"}


def normalize_text(text: Any) -> str:
    """
    Standard text normalization:
    - Lowercase
    - Expand ampersand to 'and'
    - Replace non-alphanumeric with spaces
    - Collapse extra whitespace
    """
    if is_missing(text):
        return ""
    
    text = str(text)
    # Fast ASCII lowercase conversion
    text = text.lower()
    
    # Handle symbols
    if "&" in text:
        text = text.replace("&", " and ")
    if "@" in text:
        text = text.replace("@", " at ")
    if "#" in text:
        text = text.replace("#", " ")
    
    # Remove punctuation / non-alphanumeric
    text = RE_PUNCT.sub(" ", text)
    
    # Normalize spaces
    return RE_WHITESPACE.sub(" ", text).strip()


def strip_legal_suffixes(name: str) -> str:
    """Remove legal suffixes from normalized business name."""
    if not name:
        return ""
    cleaned = name
    for pattern in COMPILED_LEGAL_SUFFIXES:
        cleaned = pattern.sub(" ", cleaned)
    return RE_WHITESPACE.sub(" ", cleaned).strip()


def extract_numbers(text: str) -> List[str]:
    """Extract all numeric tokens (house numbers, pin codes, unit numbers)."""
    if not text:
        return []
    return RE_NUMBERS.findall(text)


def normalize_address(address_str: Any) -> Dict[str, Any]:
    """
    Normalize address field and extract structured representations.
    """
    norm = normalize_text(address_str)
    if not norm:
        return {
            "address_norm": "",
            "address_tokens": [],
            "address_numbers": [],
            "address_alnum": "",
        }
    
    # Standardize common street / road terms
    padded = f" {norm} "
    for old, new in ADDRESS_REPLACEMENTS.items():
        if old in padded:
            padded = padded.replace(old, new)
    norm = RE_WHITESPACE.sub(" ", padded).strip()
    
    tokens = [t for t in norm.split() if t]
    numbers = extract_numbers(norm)
    alnum = RE_ALNUM_ONLY.sub("", norm)
    
    return {
        "address_norm": norm,
        "address_tokens": tokens,
        "address_numbers": numbers,
        "address_alnum": alnum,
    }


def normalize_name(name_str: Any) -> Dict[str, Any]:
    """
    Normalize business name and extract structured representations.
    """
    norm = normalize_text(name_str)
    if not norm:
        return {
            "name_norm": "",
            "name_no_suffix": "",
            "name_tokens": [],
            "name_alnum": "",
        }
    
    no_suffix = strip_legal_suffixes(norm)
    if not no_suffix:
        no_suffix = norm
        
    tokens = [t for t in norm.split() if t]
    alnum = RE_ALNUM_ONLY.sub("", norm)
    
    return {
        "name_norm": norm,
        "name_no_suffix": no_suffix,
        "name_tokens": tokens,
        "name_alnum": alnum,
    }


def normalize_country(country_str: Any) -> str:
    """Normalize country field while preserving open-set labels."""
    if is_missing(country_str):
        return ""
    return str(country_str).strip().lower()