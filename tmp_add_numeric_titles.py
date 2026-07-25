import json
from pathlib import Path

path = Path("notebooks/nbt_smallset_preprocessing.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))

if not any(cell.get("id") == "numeric-summary-title" for cell in notebook["cells"]):
    inserts = [
        (
            "numeric-summary",
            "numeric-summary-title",
            [
                "### 5.1 Summary statistics\n",
                "\n",
                "This table summarises the numeric columns using count, mean, standard deviation, minimum, quartiles, and maximum. It helps us quickly understand the range and spread of the values. `intended_management` and `sex_national_code` are coded categories, so their averages are less meaningful than their counts and ranges.\n",
            ],
            "numeric-summary-note",
            [
                "**Result interpretation:** Use this output to spot unusual values. For example, very high operation durations, very low ages, or ASA scores outside the expected range may need further checking.\n",
            ],
        ),
        (
            "numeric-outliers",
            "numeric-outliers-title",
            [
                "### 5.2 Suspicious numeric values\n",
                "\n",
                "This check counts values that may be impossible or suspicious, such as zero operation length, very old patient age, or ASA scores outside the usual scale.\n",
            ],
            "numeric-outliers-note",
            [
                "**Result interpretation:** A non-zero count does not automatically mean the row is wrong. It means those rows should be reviewed before cleaning, removing, or using them in modelling.\n",
            ],
        ),
        (
            "duration-diff",
            "duration-diff-title",
            [
                "### 5.3 Planned versus actual duration\n",
                "\n",
                "This creates `duration_error_mins`, which compares the recorded operation length with the expected duration. Positive values mean the case took longer than planned; negative values mean it finished faster than planned.\n",
            ],
            "duration-diff-note",
            [
                "**Result interpretation:** This is useful because the project may later focus on theatre overruns or duration prediction. Large positive errors show cases that overran the expected time.\n",
            ],
        ),
        (
            "duration-largest-errors",
            "duration-largest-errors-title",
            [
                "### 5.4 Largest duration overruns\n",
                "\n",
                "This table shows the rows with the largest positive duration errors, including theatre room and procedure description.\n",
            ],
            "duration-largest-errors-note",
            [
                "**Result interpretation:** These rows are important to inspect because they may represent true long operations, data entry issues, or cases where the planned duration was too short.\n",
            ],
        ),
    ]

    for target_id, title_id, title_source, note_id, note_source in reversed(inserts):
        index = next(
            i for i, cell in enumerate(notebook["cells"]) if cell.get("id") == target_id
        )
        notebook["cells"].insert(
            index,
            {
                "cell_type": "markdown",
                "id": title_id,
                "metadata": {},
                "source": title_source,
            },
        )
        notebook["cells"].insert(
            index + 2,
            {
                "cell_type": "markdown",
                "id": note_id,
                "metadata": {},
                "source": note_source,
            },
        )

path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
