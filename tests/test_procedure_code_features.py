import pandas as pd

from nbt_pipeline.preprocessing.codes import add_code_labels


def test_add_code_labels_extracts_procedure_hierarchy():
    source = pd.DataFrame(
        {
            "actual_proc_1_procedure_code": ["s092", " M451 ", pd.NA, "invalid"],
        }
    )

    result = add_code_labels(source)

    assert result["procedure_code_chapter"].tolist()[:2] == ["S", "M"]
    assert result["procedure_code_category"].tolist()[:2] == ["S09", "M45"]
    assert result["procedure_code_group"].tolist()[:2] == ["S0", "M4"]
    assert result.loc[2:, "procedure_code_category"].isna().all()


def test_add_code_labels_does_not_mutate_source_dataframe():
    source = pd.DataFrame({"actual_proc_1_procedure_code": ["S092"]})

    result = add_code_labels(source)

    assert "procedure_code_category" not in source.columns
    assert result.loc[0, "procedure_code_category"] == "S09"
