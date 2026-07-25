import re

import pandas as pd


SESSION_THEATRE_PATTERN = re.compile(r"^\s*([A-Z]{1,3}\s*\d+)", re.IGNORECASE)
SESSION_LIST_TYPE_PATTERN = re.compile(r"\((2SD|3SD|AM|PM|EVE|2SD/EVE|PM/EVE)\)", re.IGNORECASE)
SESSION_TIME_BAND_PATTERN = re.compile(r"\b(AM|PM|EVE|MORNING|AFTERNOON)\b", re.IGNORECASE)
SESSION_SPECIALTY_PATTERN = re.compile(r"\(([^()]*)\)\s*(?:Emergency)?\s*$", re.IGNORECASE)


def _clean_text(value: str | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def extract_session_theatre_code(session_desc: str | None) -> str | None:
    """Extract the leading theatre/session code from SessionIDdesc."""
    text = _clean_text(session_desc)
    match = SESSION_THEATRE_PATTERN.search(text)
    if not match:
        return None
    return re.sub(r"\s+", "", match.group(1)).upper()


def extract_session_list_type(session_desc: str | None) -> str | None:
    """Extract list/session marker such as 2SD, 3SD, AM, PM, or EVE."""
    text = _clean_text(session_desc)
    match = SESSION_LIST_TYPE_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    match = SESSION_TIME_BAND_PATTERN.search(text)
    if match:
        value = match.group(1).upper()
        if value == "MORNING":
            return "AM"
        if value == "AFTERNOON":
            return "PM"
        return value
    return None


def extract_session_time_band(session_desc: str | None) -> str | None:
    """Extract broad timing information from the session description."""
    text = _clean_text(session_desc)
    match = SESSION_TIME_BAND_PATTERN.search(text)
    if not match:
        return None
    value = match.group(1).upper()
    if value == "MORNING":
        return "AM"
    if value == "AFTERNOON":
        return "PM"
    return value


def extract_session_consultant(session_desc: str | None) -> str | None:
    """Extract the consultant/list holder text from SessionIDdesc."""
    text = _clean_text(session_desc)
    if not text:
        return None

    text = SESSION_THEATRE_PATTERN.sub("", text, count=1).strip()
    text = SESSION_LIST_TYPE_PATTERN.sub("", text).strip()
    text = SESSION_TIME_BAND_PATTERN.sub("", text, count=1).strip()
    text = re.sub(r"\bEmergency\b", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\([^()]*\)", "", text).strip()
    text = re.sub(r"\bTrauma\b", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" /-")
    return text or None


def add_session_description_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regex-based features from the SessionIDdesc column."""
    df = df.copy()
    if "SessionIDdesc" not in df:
        return df

    session_desc = df["SessionIDdesc"]
    df["session_theatre_code"] = session_desc.apply(extract_session_theatre_code)
    df["session_list_type"] = session_desc.apply(extract_session_list_type)
    df["session_time_band"] = session_desc.apply(extract_session_time_band)
    df["session_consultant"] = session_desc.apply(extract_session_consultant)
    return df
