import re

import pandas as pd

OPCS_CHAPTER_MAP = {
    "A": "Nervous System",
    "B": "Endocrine System and Breast",
    "C": "Eye",
    "D": "Ear",
    "E": "Respiratory Tract",
    "F": "Mouth",
    "G": "Upper Digestive System",
    "H": "Lower Digestive System",
    "J": "Other Abdominal Organs, Principally Digestive",
    "K": "Heart",
    "L": "Arteries and Veins",
    "M": "Urinary",
    "N": "Male Genital Organs",
    "O": "Overflow OPCS code - chapter requires official OPCS lookup",
    "P": "Lower Female Genital Tract",
    "Q": "Upper Female Genital Tract",
    "R": "Female Genital Tract Associated with Pregnancy, Childbirth and the Puerperium",
    "S": "Skin",
    "T": "Soft Tissue",
    "U": "Diagnostic Imaging, Testing and Rehabilitation",
    "V": "Bones and Joints of Skull and Spine",
    "W": "Other Bones and Joints",
    "X": "Miscellaneous Operations",
    "Y": "Subsidiary Classification of Methods of Operation",
    "Z": "Subsidiary Classification of Sites of Operation",
}


def decode_opcs_code(code) -> pd.Series:
    if pd.isna(code):
        return pd.Series({
            "opcs_code_clean": pd.NA,
            "opcs_code_dotted": pd.NA,
            "opcs_chapter_letter": pd.NA,
            "opcs_chapter_name": pd.NA,
            "opcs_category_code": pd.NA,
            "opcs_subcategory_digit": pd.NA,
            "opcs_code_format_valid": False,
        })

    code_clean = re.sub(r"[^A-Z0-9]", "", str(code).upper().strip())
    format_valid = bool(re.fullmatch(r"[A-Z][0-9]{3}", code_clean))

    chapter_letter = code_clean[0] if len(code_clean) >= 1 else pd.NA
    category_code = code_clean[:3] if len(code_clean) >= 3 else pd.NA
    subcategory_digit = code_clean[3] if len(code_clean) >= 4 else pd.NA
    code_dotted = f"{code_clean[:3]}.{code_clean[3]}" if format_valid else pd.NA

    return pd.Series({
        "opcs_code_clean": code_clean if code_clean else pd.NA,
        "opcs_code_dotted": code_dotted,
        "opcs_chapter_letter": chapter_letter,
        "opcs_chapter_name": OPCS_CHAPTER_MAP.get(chapter_letter, pd.NA),
        "opcs_category_code": category_code,
        "opcs_subcategory_digit": subcategory_digit,
        "opcs_code_format_valid": format_valid,
    })


def decode_opcs_column(df: pd.DataFrame, column: str = "actual_proc_1_procedure_code") -> pd.DataFrame:
    opcs_info_df = df[column].apply(decode_opcs_code)
    return pd.concat([df, opcs_info_df], axis=1)


def lookup_opcs_code(code: str) -> dict:
    """Single-code lookup returning plain JSON-friendly types, for use as a tool."""
    decoded = decode_opcs_code(code)
    return {key: (None if pd.isna(value) else value) for key, value in decoded.items()}
