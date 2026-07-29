# Next Steps

## Current focus

The current branch is for time column inspection and validation. The raw dataset should stay unchanged. Any corrected or inferred time values should be treated as derived analysis fields, not replacements for the original columns.

## Time columns

- Keep the raw time columns unchanged.
- Use Section 8 for inspection, explanation, and validation.
- Treat `out_of_theatre` as the likely full clock-time anchor.
- Treat columns such as `into_theatre`, `incision`, `closure`, and `operation_end_time` carefully because they appear to store minute-of-hour values.
- Use `operation_length_rule_valid` and `time_sequence_valid` as quality flags.
- Do not replace suspicious time rows with `NaN` yet.
- Later, decide whether suspicious time rows should be excluded from duration modelling.

## Relationship checks

- Continue Section 9 after the time validation.
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
