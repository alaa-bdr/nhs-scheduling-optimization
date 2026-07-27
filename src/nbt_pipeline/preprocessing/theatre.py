import re

import pandas as pd


ROOM_NUMBER_PATTERN = re.compile(r"(\d+)\s*$")


def _clean_room(value: str | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def extract_theatre_area(theatre_room: str | None) -> str | None:
    """Extract the broad theatre area/unit from TheatreRoom."""
    room = _clean_room(theatre_room)
    if not room:
        return None

    room_without_number = ROOM_NUMBER_PATTERN.sub("", room).strip()
    room_without_number = re.sub(r"\bTH\s*$", "", room_without_number).strip()
    return room_without_number or None


def extract_theatre_room_number(theatre_room: str | None) -> str | None:
    """Extract the final room number from TheatreRoom."""
    room = _clean_room(theatre_room)
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
    df["theatre_room_number"] = theatre_room.apply(extract_theatre_room_number)
    df["theatre_is_ir"] = theatre_room.astype("string").str.contains(
        "IR LAB",
        case=False,
        regex=False,
        na=False,
    )
    return df
