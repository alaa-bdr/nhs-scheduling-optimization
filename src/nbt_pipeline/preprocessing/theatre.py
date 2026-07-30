import re

import pandas as pd


ROOM_NUMBER_PATTERN = re.compile(r"(\d+)\s*$")
THEATRE_ROOM_PATTERN = re.compile(r"\bTH\s*([A-Z]|\d+)\s*$")
THEATRE_PREFIX_PATTERN = re.compile(r"\bTH\s*([A-Z]|\d+)?\s*$")


def _clean_room(value: str | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def extract_theatre_area(theatre_room: str | None) -> str | None:
    """Extract the broad theatre area/unit from TheatreRoom."""
    room = _clean_room(theatre_room)
    if not room:
        return None

    room_without_number = THEATRE_ROOM_PATTERN.sub("", room).strip()
    room_without_number = ROOM_NUMBER_PATTERN.sub("", room_without_number).strip()
    room_without_number = re.sub(r"\bTH\s*$", "", room_without_number).strip()
    return room_without_number or None


def extract_theatre_room_prefix(theatre_room: str | None) -> str | None:
    """Extract the room type/prefix from TheatreRoom."""
    room = _clean_room(theatre_room)
    if not room:
        return None

    if THEATRE_PREFIX_PATTERN.search(room):
        return "TH"
    if room.startswith("IR LAB"):
        return "IR LAB"
    if room.startswith("CATH LAB"):
        return "CATH LAB"
    if room.startswith("CT ROOM"):
        return "CT ROOM"
    if "FLUORO" in room:
        return "FLUORO"
    if room.startswith("PLASTIC MINOR"):
        return "MINOR"
    if room == "HYBRID THEATRE":
        return "HYBRID THEATRE"
    if room == "MOBILE IR":
        return "MOBILE IR"
    if room == "PACING ROOM":
        return "PACING ROOM"
    if room.startswith("THEATRE"):
        return "THEATRE"

    return "OTHER"


def extract_theatre_room_number(theatre_room: str | None) -> str | None:
    """Extract the final room number from TheatreRoom."""
    room = _clean_room(theatre_room)
    match = ROOM_NUMBER_PATTERN.search(room)
    theatre_match = THEATRE_ROOM_PATTERN.search(room)
    if theatre_match:
        room_value = theatre_match.group(1)
        return room_value.zfill(2) if room_value.isdigit() else room_value

    match = ROOM_NUMBER_PATTERN.search(room)
    if not match:
        return None
    return match.group(1).zfill(2)


def add_theatre_room_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple features extracted from TheatreRoom."""
    df = df.copy()
    if "TheatreRoom" not in df:
        return df

    theatre_room = df["TheatreRoom"]
    df["theatre_area"] = theatre_room.apply(extract_theatre_area)
    df["theatre_room_prefix"] = theatre_room.apply(extract_theatre_room_prefix)
    df["theatre_room_number"] = theatre_room.apply(extract_theatre_room_number)
    df["theatre_is_ir"] = theatre_room.astype("string").str.contains(
        "IR LAB",
        case=False,
        regex=False,
        na=False,
    )
    return df
