# Project Target

## Main direction

The current project direction is theatre scheduling and bottleneck analysis.

The main goal is to find what predicts longer operations, overruns, under-used planned time, and theatre bottlenecks. The next stage should focus on:

```text
time use + theatre location + procedure type + specialty + admission/priority
```

This is the professional direction because theatre scheduling depends on understanding both planned time and real theatre use.

## Scheduling questions to answer

### 1. Where are delays happening?

Use these fields:

```text
theatre_area
TheatreRoom
session_time_band
session_list_type
operation_length_mins
overrun_minutes
underrun_minutes
```

This helps identify whether delays are linked to a specific theatre area, exact room, time band, or type of session.

### 2. Which theatres are busiest?

Use these fields:

```text
TheatreRoom
theatre_area
cases
```

This helps show where the largest amount of theatre activity is happening. Busy rooms or areas may have more pressure on scheduling.

### 3. Which procedure groups take longer or overrun more?

Use these fields:

```text
procedure_code_group
ProcedureDescription
ExpectedDurationMins
operation_length_mins
duration_error_mins
```

This helps compare planned and actual operation time by procedure type. The full procedure description gives detail, while `procedure_code_group` gives a broader and more stable grouping.

### 4. Is the planned time accurate?

Compare these fields:

```text
ExpectedDurationMins
operation_length_mins
duration_error_mins
```

This checks whether planned durations are close to actual durations. Large positive values suggest overruns; large negative values suggest planned time was unused.

### 5. Which specialties or consultants have different patterns?

Use these fields carefully:

```text
session_specialty
session_consultant
listing_cons_code
theat_surg_1_national_code
```

This may show differences in operation duration or scheduling accuracy by specialty or staff group. These columns should be interpreted carefully because consultant names and staff codes may represent different roles.

### 6. Are some session types more likely to overrun?

Use these fields:

```text
admission_type_label
intended_management_label
priority_level_label
session_list_type
anaesthetic_desc
ASAScore
```

This helps check whether emergency/elective status, priority, anaesthetic type, or patient risk level is linked to longer operations or overruns.

## Main modelling target

The main target variable should be:

```text
operation_length_mins
```

This is the actual operation duration recorded in minutes. It is the best starting target because better duration prediction can support better theatre scheduling.

## Secondary derived targets

These targets can be created from planned vs actual duration:

```text
duration_error_mins = operation_length_mins - ExpectedDurationMins
```

```text
is_overrun = operation_length_mins > ExpectedDurationMins
```

```text
overrun_minutes = positive difference when actual duration is longer than planned
```

```text
underrun_minutes = positive difference when planned duration is longer than actual duration
```

## Useful feature groups

Useful predictors for scheduling analysis may include:

- procedure group and procedure description
- theatre area and exact theatre room
- session list type and session time band
- clinical specialty
- admission type
- intended management
- priority level
- anaesthetic type
- ASA score
- age at operation
- sex

## Important modelling note

Before modelling, suspicious timing rows should be handled carefully. Use `operation_length_rule_valid` and `time_sequence_valid` to identify rows that may not be reliable for duration modelling.
