import re

import pandas as pd


SPECIALTY_ALIASES = {
    "trauma and orthopaedics": "T&O",
    "trauma & orthopaedics": "T&O",
    "trauma & ortho": "T&O",
    "trauma and ortho": "T&O",
    "orthopaedics": "T&O",
    "plastic surgery": "Plastic surgery",
    "urology": "Urology",
    "neurosurgery": "Neuro",
    "neurology": "Neuro",
    "obstetrics and gynaecology": "Obs&Gynae",
    "obstetrics": "Obs&Gynae",
    "gynaecology": "Obs&Gynae",
    "obs & gynae": "Obs&Gynae",
    "general surgery": "General",
    "breast surgery": "Breast",
    "colorectal surgery": "Colo-Rectal",
    "vascular surgery": "Vascular Surgery",
    "radiology": "Radiology",
}

PROCEDURE_SPECIALTY_FALLBACKS = {
    "Elective lower uterine segment caesarean delivery": "Obs&Gynae",
    "Evacuation of products of conception from uterus NEC": "Obs&Gynae",
    "Drainage of kidney NEC": "Urology",
    "Unspecified diagnostic endoscopic examination of bladder": "Urology",
    "Percutaneous transluminal angioplasty of artery": "Vascular Surgery",
}

NON_SPECIALTY_PARENTHESES = {
    "2sd",
    "3sd",
    "am",
    "pm",
    "eve",
    "2sd/eve",
    "pm/eve",
}


def normalise_specialty(value: str | None) -> str | None:
    """Map specialty text variants to a smaller set of labels."""
    if value is None or pd.isna(value):
        return None
    value_clean = re.sub(r"\s+", " ", str(value).strip()).lower()
    if not value_clean:
        return None

    if value_clean in SPECIALTY_ALIASES:
        return SPECIALTY_ALIASES[value_clean]

    for alias, label in SPECIALTY_ALIASES.items():
        if alias in value_clean:
            return label

    return str(value).strip()


def parenthesised_parts(text: str | None) -> list[str]:
    """Return all text fragments inside parentheses."""
    if text is None or pd.isna(text):
        return []
    return re.findall(r"\(([^()]*)\)", str(text))


def extract_specialty(session_desc: str | None, procedure_description: str | None = None) -> str | None:
    """Infer a specialty from session text, with procedure-based fallbacks."""
    session_text = "" if session_desc is None or pd.isna(session_desc) else str(session_desc)
    direct_match = normalise_specialty(session_text)
    if direct_match:
        return direct_match

    for part in reversed(parenthesised_parts(session_text)):
        part_clean = part.strip().lower()
        if part_clean and part_clean not in NON_SPECIALTY_PARENTHESES:
            return normalise_specialty(part)

    if procedure_description in PROCEDURE_SPECIALTY_FALLBACKS:
        return PROCEDURE_SPECIALTY_FALLBACKS[procedure_description]

    return None


def add_specialty_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a normalised specialty column from SessionIDdesc and ProcedureDescription."""
    df = df.copy()
    df["session_specialty"] = df.apply(
        lambda row: extract_specialty(row.get("SessionIDdesc"), row.get("ProcedureDescription")),
        axis=1,
    )
    return df
