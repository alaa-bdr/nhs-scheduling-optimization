from nbt_pipeline.config import PROCESSED_DATA_DIR, RESULT_DIR
from nbt_pipeline.modeling import save_final_model_artifact
from nbt_pipeline.outputs import save_dataframe
from nbt_pipeline.preprocessing import build_analysis_dataset


def main() -> None:
    df = build_analysis_dataset()
    output_path = save_dataframe(df, PROCESSED_DATA_DIR / "nbt_smallset_analysis_room_only.xlsx")
    print(f"Saved cleaned analysis dataset to: {output_path}")
    model_output_dir = RESULT_DIR / "modeling" / "final_models"
    for target in ("duration_error_mins", "operation_length_mins", "meaningful_overrun_flag"):
        metadata = save_final_model_artifact(df, target, model_output_dir)
        print(f"Saved {target} model pipeline to: {metadata['artifact_path']}")


if __name__ == "__main__":
    main()
