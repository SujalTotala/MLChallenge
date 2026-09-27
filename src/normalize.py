"""
Comprehensive text normalization and structured representation extraction engine.
Supports open-set country representations and multi-channel entity features.
"""

import re
import unicodedata
from typing import Dict, Any, List, Set, Tuple, Optional

# Regular expressions
RE_PUNCT = re.compile(r"[^a-z0-9\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_ALNUM_ONLY = re.compile(r"[^a-z0-9]")

# Geo pattern matchers
RE_PINCODE_IN = re.compile(r"\b[1-9][0-9]{5}\b")  # India 6-digit PIN
RE_ZIPCODE_US_FR = re.compile(r"\b\d{5}\b")        # US 5-digit ZIP / France 5-digit Code Postal

US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy", "dc", "pr"
}

# Legal business suffix regex patterns
LEGAL_SUFFIX_PATTERNS = [
    r"\bprivate\s+limited\b",
    r"\bpvt\s+ltd\b",
    r"\bpublic\s+limited\s+company\b",
    r"\bpublic\s+limited\b",
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
    r"\bindustries\b",
    r"\btech(?:nologies)?\b",
    r"\binternational\b",
    r"\bglobal\b",
]

COMPILED_LEGAL_SUFFIXES = [re.compile(p, re.IGNORECASE) for p in LEGAL_SUFFIX_PATTERNS]

# Address token standardizations & alias mapping
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
    " calcutta ": " kolkata ",
    " bombay ": " mumbai ",
    " madras ": " chennai ",
    " bangalore ": " bengaluru ",
    " gurgaon ": " gurugram ",
    " cochin ": " kochi ",
    " trivandrum ": " thiruvananthapuram ",
    " baroda ": " vadodara ",
    " poona ": " pune ",
    " pondicherry ": " puducherry ",
    " orissa ": " odisha ",
    " calicut ": " kozhikode ",
    " benares ": " varanasi ",
    " banaras ": " varanasi ",
    " allahabad ": " prayagraj ",
    " mysore ": " mysuru ",
    " mangalore ": " mangaluru ",
    " belgaum ": " belagavi ",
    " hubli ": " hubballi ",
    " secunderabad ": " hyderabad ",
    " rue ": " r ",
    " avenue ": " ave ",
    " boulevard ": " bd ",
    " allée ": " all ",
    " place ": " pl ",
}


def is_missing(val: Any) -> bool:
    """Check if value is null, NaN, None, or empty string."""
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in {"", "nan", "none", "null", "<na>"}


def normalize_country(country_str: Any) -> str:
    """
    Normalize country field as an open-set string.
    Works naturally for US, India, France, and any unseen future country.
    """
    if is_missing(country_str):
        return "unknown"
    return str(country_str).strip().lower()


def clean_text_basic(text: Any) -> str:
    """
    Basic text cleaning:
    - Lowercase
    - Unicode NFKD normalization
    - Expand & -> and, @ -> at
    - Replace punctuation with spaces
    - Collapse extra whitespace
    """
    if is_missing(text):
        return ""
    
    # 1. Convert to string and lowercase
    text = str(text).lower()
    
    # 2. Unicode normalization (convert accents/diacritics e.g., French 'Café' -> 'cafe')
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    
    # 3. Expand common symbol words
    if "&" in text:
        text = text.replace("&", " and ")
    if "@" in text:
        text = text.replace("@", " at ")
    if "#" in text:
        text = text.replace("#", " ")
        
    # 4. Remove punctuation
    text = RE_PUNCT.sub(" ", text)
    
    # 5. Collapse spaces
    return RE_WHITESPACE.sub(" ", text).strip()


def strip_legal_suffixes(name: str) -> str:
    """Remove common legal suffixes from normalized business name."""
    if not name:
        return ""
    cleaned = name
    for pattern in COMPILED_LEGAL_SUFFIXES:
        cleaned = pattern.sub(" ", cleaned)
    return RE_WHITESPACE.sub(" ", cleaned).strip()


def normalize_name(name_str: Any) -> Dict[str, Any]:
    """
    Generate multiple representations for business_name:
    - name_raw
    - name_norm
    - name_core
    - name_compact
    - name_tokens
    - name_sorted
    - name_first_tokens
    """
    raw_name = str(name_str) if not is_missing(name_str) else ""
    norm = clean_text_basic(raw_name)
    
    if not norm:
        return {
            "name_raw": raw_name,
            "name_norm": "",
            "name_core": "",
            "name_compact": "",
            "name_tokens": [],
            "name_sorted": "",
            "name_first_tokens": "",
        }
    
    # Strip legal suffixes for core name
    core = strip_legal_suffixes(norm)
    if not core:
        core = norm
        
    tokens = [t for t in core.split() if t]
    sorted_tokens = sorted(tokens)
    name_sorted = " ".join(sorted_tokens)
    
    compact = RE_ALNUM_ONLY.sub("", core)
    if not compact:
        compact = RE_ALNUM_ONLY.sub("", norm)

    if len(tokens) >= 2:
        first_tokens = f"{tokens[0]}_{tokens[1]}"
    elif len(tokens) == 1:
        first_tokens = tokens[0]
    else:
        first_tokens = ""

    return {
        "name_raw": raw_name,
        "name_norm": norm,
        "name_core": core,
        "name_compact": compact,
        "name_tokens": tokens,
        "name_sorted": name_sorted,
        "name_first_tokens": first_tokens,
    }


def normalize_address(address_str: Any, country_norm: str = "unknown") -> Dict[str, Any]:
    """
    Generate multiple representations for business_address:
    - address_norm
    - address_tokens
    - house_number
    - postal_code
    - city
    - state
    - street_tokens
    - address_compact
    """
    raw_addr = str(address_str) if not is_missing(address_str) else ""
    norm = clean_text_basic(raw_addr)
    
    if not norm:
        return {
            "address_norm": "",
            "address_tokens": [],
            "house_number": [],
            "postal_code": "",
            "city": "",
            "state": "",
            "street_tokens": [],
            "address_compact": "",
        }
    
    # Expand address abbreviations
    padded = f" {norm} "
    for old, new in ADDRESS_REPLACEMENTS.items():
        if old in padded:
            padded = padded.replace(old, new)
    norm = RE_WHITESPACE.sub(" ", padded).strip()
    
    tokens = [t for t in norm.split() if t]
    
    # Extract numbers (house numbers, suite numbers)
    numbers = RE_NUMBERS.findall(norm)
    
    # Extract postal code (India: 6-digit, US/France: 5-digit)
    postal_code = ""
    if country_norm == "india":
        pins = RE_PINCODE_IN.findall(norm)
        if pins:
            postal_code = pins[0]
    elif country_norm in {"us", "france"}:
        zips = RE_ZIPCODE_US_FR.findall(norm)
        if zips:
            postal_code = zips[0]
    else:
        # Fallback geo check
        pins = RE_PINCODE_IN.findall(norm)
        if pins:
            postal_code = pins[0]
        else:
            zips = RE_ZIPCODE_US_FR.findall(norm)
            if zips:
                postal_code = zips[0]

    # Extract state if US
    state = ""
    if country_norm == "us":
        for t in tokens:
            if t in US_STATES:
                state = t
                break

    # Non-numeric street tokens
    street_tokens = [t for t in tokens if not t.isdigit() and len(t) >= 3]
    compact = RE_ALNUM_ONLY.sub("", norm)

    return {
        "address_norm": norm,
        "address_tokens": tokens,
        "house_number": numbers,
        "postal_code": postal_code,
        "city": "",  # Extracted if available in data
        "state": state,
        "street_tokens": street_tokens,
        "address_compact": compact,
    }