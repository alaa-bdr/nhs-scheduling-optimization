import re

from crewai.tasks.task_output import TaskOutput

from nbt_pipeline.extraction.schemas import TheatreNoteExtraction


OPTIONAL_TEXT_FIELDS = [
    "procedure_summary",
    "priority_code",
    "urgency_text",
    "biopsy_or_specimen_plan",
    "closure_or_reconstruction_plan",
    "anticoagulation_status",
    "mobility_or_access_needs",
]

NON_PROCEDURE_CODE_PATTERN = re.compile(r"^(P\d+|2WW|GA|LA|DC|IP|OP)$", re.IGNORECASE)
OPCS_CODE_PATTERN = re.compile(r"^[A-Z]\d{2}(?:\.?\d)?$", re.IGNORECASE)
NEGATIVE_IMAGING_PATTERN = re.compile(r"\b(x-?ray|screening|image intensifier|ii)\b.*\b(no|not required|none)\b", re.IGNORECASE)
ANAESTHESIA_ONLY_PATTERN = re.compile(r"^(GA|LA|regional|sedation|spinal|epidural|other)$", re.IGNORECASE)
CASE_SETTING_PATTERN = re.compile(r"\b(day\s*case|inpatient|outpatient)\b", re.IGNORECASE)


def _normalise_case_setting(value: str) -> str | None:
    value_clean = value.lower().replace("-", " ")
    if "day case" in value_clean or "daycase" in value_clean:
        return "day_case"
    if "inpatient" in value_clean:
        return "inpatient"
    if "outpatient" in value_clean:
        return "outpatient"
    return None


def validate_theatre_note_extraction(output: TaskOutput) -> tuple[bool, TaskOutput | str]:
    extraction = output.pydantic
    if not isinstance(extraction, TheatreNoteExtraction):
        return False, "Output must match the TheatreNoteExtraction schema."

    if extraction.expected_duration_minutes == 0:
        extraction.expected_duration_minutes = None

    extraction.procedure_codes_mentioned = [
        code
        for code in extraction.procedure_codes_mentioned
        if OPCS_CODE_PATTERN.fullmatch(code.strip())
        and not NON_PROCEDURE_CODE_PATTERN.fullmatch(code.strip())
    ]

    extraction.procedure_components = [
        component
        for component in extraction.procedure_components
        if not ANAESTHESIA_ONLY_PATTERN.fullmatch(component.strip())
    ]

    extraction.imaging_required = [
        item for item in extraction.imaging_required if not NEGATIVE_IMAGING_PATTERN.search(item)
    ]

    special_instructions = []
    for instruction in extraction.special_instructions:
        if CASE_SETTING_PATTERN.search(instruction):
            extraction.case_setting = _normalise_case_setting(instruction) or extraction.case_setting
            continue
        special_instructions.append(instruction)
    extraction.special_instructions = special_instructions

    for field in OPTIONAL_TEXT_FIELDS:
        value = getattr(extraction, field)
        if value is not None and str(value).strip() in {"", "not_stated"}:
            setattr(extraction, field, None)

    output.pydantic = extraction

    return True, output
