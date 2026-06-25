import pandas as pd

from nbt_pipeline.extraction.crews.extractor_crew.extractor_crew import ExtractorCrew
from nbt_pipeline.extraction.schemas import TheatreNoteExtraction


def extract_theatre_notes(df: pd.DataFrame, column: str = "theatre_notes") -> pd.DataFrame:
    crew = ExtractorCrew().crew()
    records = []

    for note in df[column]:
        if pd.isna(note):
            records.append(TheatreNoteExtraction().model_dump())
            continue
        result = crew.kickoff(inputs={"theatre_note": note})
        records.append(result.pydantic.model_dump())

    extracted_df = pd.DataFrame(records, index=df.index).add_prefix("theatre_notes_")
    return pd.concat([df, extracted_df], axis=1)
