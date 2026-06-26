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
NEGATIVE_IMAGING_PATTERN = re.compile(r"\b(x-?ray|screening|image intensifier|ii)\b.*\b(no|not required|none)\b", re.IGNORECASE)


def validate_theatre_note_extraction(output: TaskOutput) -> tuple[bool, TaskOutput | str]:
    extraction = output.pydantic
    if not isinstance(extraction, TheatreNoteExtraction):
        return False, "Output must match the TheatreNoteExtraction schema."

    if extraction.expected_duration_minutes == 0:
        extraction.expected_duration_minutes = None

    extraction.procedure_codes_mentioned = [
        code
        for code in extraction.procedure_codes_mentioned
        if not NON_PROCEDURE_CODE_PATTERN.fullmatch(code.strip())
    ]

    extraction.imaging_required = [
        item for item in extraction.imaging_required if not NEGATIVE_IMAGING_PATTERN.search(item)
    ]

    for field in OPTIONAL_TEXT_FIELDS:
        value = getattr(extraction, field)
        if value is not None and str(value).strip() in {"", "not_stated"}:
            setattr(extraction, field, None)

    output.pydantic = extraction

    return True, output
