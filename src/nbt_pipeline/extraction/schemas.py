from typing import Literal, Optional

from pydantic import BaseModel, Field


class TheatreNoteExtraction(BaseModel):
    """Structured fields extracted from a single `theatre_notes` entry."""

    has_extractable_note: bool = Field(
        default=False,
        description="True when the note contains meaningful clinical, booking, or procedural information.",
    )
    procedure_summary: Optional[str] = Field(
        default=None,
        description="Concise plain-English summary of the main planned/performed procedure.",
    )
    procedure_components: list[str] = Field(
        default_factory=list,
        description="Specific procedure actions mentioned, e.g. excision, closure, biopsy, cystoscopy, repair.",
    )
    procedure_codes_mentioned: list[str] = Field(
        default_factory=list,
        description=(
            "OPCS-like procedure/classification codes explicitly written in the note, if any. "
            "Do not include priority codes such as P2, P3, P4 or urgency codes such as 2WW."
        ),
    )
    diagnoses_or_indications: list[str] = Field(
        default_factory=list,
        description="Diagnoses, suspected diagnoses, reasons for surgery, symptoms, or clinical indications.",
    )
    anatomical_sites: list[str] = Field(
        default_factory=list,
        description="Body sites or anatomical targets explicitly stated in the note.",
    )
    laterality: Literal["left", "right", "bilateral", "midline", "not_stated"] = Field(
        default="not_stated",
        description="Laterality of the procedure when stated or clearly implied.",
    )
    anaesthesia_types: list[Literal["GA", "LA", "regional", "sedation", "spinal", "epidural", "other"]] = Field(
        default_factory=list,
        description="Normalised anaesthesia types explicitly mentioned in the note.",
    )
    expected_duration_minutes: Optional[int] = Field(
        default=None,
        description="Expected or listed duration in minutes, only when stated in the note. Use null, never 0, when absent.",
    )
    case_setting: Literal["day_case", "inpatient", "outpatient", "not_stated"] = Field(
        default="not_stated",
        description="Admission or setting hint stated in the note, such as day case, inpatient, or outpatient.",
    )
    priority_code: Optional[str] = Field(
        default=None,
        description="Priority code exactly as stated, e.g. P2, P3, P4, D2, RCS Level 2. Use null when absent.",
    )
    urgency_text: Optional[str] = Field(
        default=None,
        description="Free-text urgency or waiting-time statement, e.g. urgent non cancer, 2WW, maximum wait 6 weeks. Use null when absent.",
    )
    requires_histology: Optional[bool] = Field(
        default=None,
        description="True if histology/pathology/specimen analysis is explicitly requested; false if explicitly not needed.",
    )
    biopsy_or_specimen_plan: Optional[str] = Field(
        default=None,
        description="Biopsy, specimen, pathology, or histology plan if mentioned. Use null when absent.",
    )
    closure_or_reconstruction_plan: Optional[str] = Field(
        default=None,
        description="Closure, graft, flap, repair, reconstruction, or direct-closure plan if mentioned. Use null when absent.",
    )
    equipment_required: list[str] = Field(
        default_factory=list,
        description="Named equipment, sets, scopes, saws, microscopes, stacks, or theatre resources required.",
    )
    imaging_required: list[str] = Field(
        default_factory=list,
        description=(
            "Imaging or guidance explicitly requested, e.g. X-ray, image intensifier, ultrasound, screening. "
            "Do not include negative statements such as 'Xray no' or 'X-ray No'."
        ),
    )
    implants_or_materials: list[str] = Field(
        default_factory=list,
        description="Implants, plates, screws, anchors, prostheses, grafts, Botox, clips, or other materials mentioned.",
    )
    medications_or_injections: list[str] = Field(
        default_factory=list,
        description="Drugs or injections explicitly mentioned as part of the procedure or peri-operative plan.",
    )
    comorbidities_or_risk_factors: list[str] = Field(
        default_factory=list,
        description="Comorbidities, risk factors, ASA-like details, pregnancy/BMI, anticoagulation, or relevant history.",
    )
    anticoagulation_status: Optional[str] = Field(
        default=None,
        description="Anticoagulant status or named anticoagulant if explicitly mentioned. Use null when absent.",
    )
    mobility_or_access_needs: Optional[str] = Field(
        default=None,
        description="Mobility, access, transfer, or patient-support needs if mentioned. Use null when absent.",
    )
    special_instructions: list[str] = Field(
        default_factory=list,
        description="Operational instructions, scheduling constraints, ICU need, named clinician requests, or comments.",
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Uncertain alternatives or question-marked items stated in the note, preserving uncertainty.",
    )
    abbreviations_expanded: Optional[dict[str, str]] = Field(
        default=None,
        description="Abbreviation to expansion pairs found in the note. Only expand when the meaning is clear.",
    )
