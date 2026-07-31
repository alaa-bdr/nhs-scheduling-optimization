# Next Steps

## Current focus

The current branch is for creating a cleaned analysis dataset before deeper EDA. The raw dataset should stay unchanged. Cleaning should happen on `analysis_df`, not directly on `nbt_smallset_df`.

## Data cleaning approach

- Keep `nbt_smallset_df` as the raw source dataframe.
- Create `analysis_df` as the cleaned dataframe for later EDA and modelling preparation.
- Convert only clearly invalid or illogical values to `NaN`.
- Do not delete rows at this stage.
- Do not fill missing values with mean, median, mode, or `Unknown` yet.
- Keep a cleaning summary showing which columns changed and how many values were converted to `NaN`.
- Visualise missing values after cleaning to decide which columns need attention later.

## Values currently converted to `NaN`

- `ExpectedDurationMins <= 0`
- `operation_length_mins <= 0`
- `age_at_operation <= 0`
- `age_at_operation > 125`
- `ASAScore` outside the valid range `1` to `6`
- invalid values in known coded columns, including `admission_type`, `intended_management`, `sex_national_code`, `ASAScore`, and `PriorityLevelCode`
- empty text strings

## Ambiguous values to review before changing

These values should not be automatically cleaned to `NaN` yet. They may be valid local operational values, optional fields, or values that need domain confirmation.

- Generic theatre labels such as `THEATRE 01`, because the area is unclear.
- Named theatre locations such as `MOBILE IR`, `PACING ROOM`, and `HYBRID THEATRE`, because they are valid named locations and may not have room numbers.
- `GYNAE TH B`, because it uses a letter room rather than a numeric room. It is likely valid, but should be confirmed.
- `TTH` in `SessionIDdesc`, because it looks like a typo for `TH`, but should be confirmed before correcting.
- Missing `recovery_time`, because it may be optional or not captured consistently.
- Missing `PriorityLevelCode`, because it may not be required for every case.
- Missing `theat_anae_1_national_code`, because it may mean no anaesthetist code was recorded, a local coding gap, or not applicable.
- Consultant/staff-code mismatches, because `session_consultant`, `listing_cons_code`, `theat_surg_1_national_code`, and `theat_anae_1_national_code` may refer to different roles.
- Rare theatre rooms, because rare does not automatically mean invalid.
- Rare procedure codes, because rare does not automatically mean invalid.
- Suspicious time rows, because they are better handled with validation flags first.
- `theatre_notes = "."`, because it probably means no note, but the project should decide before converting it.

Current rule: impossible values are cleaned to `NaN`; ambiguous values are kept, flagged, or reviewed.

## Time columns

- Keep the raw time columns unchanged.
- Use Section 8 for inspection, explanation, and validation.
- Treat `out_of_theatre` as the likely full clock-time anchor.
- Treat columns such as `into_theatre`, `incision`, `closure`, and `operation_end_time` carefully because they appear to store minute-of-hour values.
- Use `operation_length_rule_valid` and `time_sequence_valid` as quality flags.
- Do not replace suspicious time rows with `NaN` yet.
- Later, decide whether suspicious time rows should be excluded from duration modelling.

## Relationship checks

- Continue relationship and visual EDA after Section 9 cleaning.
- Check the relationship between `ExpectedDurationMins` and `operation_length_mins`.
- Compare operation length by `TheatreRoom`.
- Important: inspect all rooms within each theatre area, for example Brunel rooms, Cotswold rooms, Plastic Minor rooms, and IR rooms. This helps check whether rooms from the same area behave similarly or whether individual room numbers have different duration patterns.
- Compare operation length by `ProcedureDescription`.
- Compare operation length by `anaesthetic_desc`.
- Compare operation length by `admission_type`.
- Compare operation length by `PriorityLevelCode`.
- Compare operation length by `ASAScore`.
- Use median as well as mean because operation durations contain outliers.

## Modelling preparation

- Decide the target variable before modelling.
- Possible target: `operation_length_mins`.
- Decide which columns are useful features.
- Be careful with high-cardinality identifiers, such as consultant and staff-code columns.
- Use the time validation flags to avoid training on suspicious timing rows.
- Keep raw columns and engineered columns separate.

## Important project decision

The time reconstruction rule looks useful, but it is inferred from the dataset rather than confirmed by official documentation. Use it for validation and quality flags first. Only add inferred time columns to the main pipeline after the rule has been reviewed and accepted.

## Related notes

- Project target: `notes/project_target.md`
- Supervisor questions: `notes/supervisor_questions.md`
