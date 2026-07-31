# Supervisor Questions

These questions should be discussed before final cleaning, feature engineering, or modelling decisions are locked in.

## Cleaning rules

- Are the current cleaning rules acceptable?
- Should invalid duration values `<= 0` be converted to `NaN` in `analysis_df`?
- Should impossible age values, invalid ASA values, and invalid coded-category values be converted to `NaN`?
- Should missing values be left as `NaN` for now, or should any specific columns be filled using a local operational rule?

## Timing data

- Should suspicious time rows be excluded from duration modelling later, or kept with validation flags?
- Are `operation_length_rule_valid` and `time_sequence_valid` acceptable flags for identifying unreliable timing rows?
- Is the current assumption correct that `out_of_theatre` is the likely full clock-time anchor?
- Are fields such as `into_theatre`, `incision`, `closure`, and `operation_end_time` stored as minute-of-hour values rather than full clock times?
- Should the inferred time reconstruction rule be used only for validation, or can it later be added to the main pipeline?

## Theatre room and location values

- What is the meaning of generic theatre values such as `THEATRE 01`?
- Are generic theatre values unclear because they do not include a specific area such as Brunel, Parkview, Cotswold, or Gynae?
- Are room numbers such as `TH 01`, `TH 02`, and `TH 11` local to each theatre area, or do they refer to the same physical rooms across the hospital?
- Current assumption: `BRUNEL TH 01`, `PARKVIEW TH01`, `COTSWOLD TH01`, and `GYNAE TH 01` should be treated as different locations unless confirmed otherwise.
- Should the original `TheatreRoom` column remain the safest exact room label?
- Is `GYNAE TH B` a valid room label, and should it be kept as room number `B`?
- Are `MOBILE IR`, `PACING ROOM`, and `HYBRID THEATRE` valid named locations with no room number?

## Session description values

- Is `TTH` in `SessionIDdesc` a typo for `TH`?
- If `TTH` is a typo, can it be standardised to `TH`?
- Are extracted fields such as `session_theatre_code`, `session_list_type`, `session_time_band`, and `session_consultant` meaningful for local operational analysis?

## Missing or ambiguous fields

- Should `theatre_notes = "."` be treated as a missing note?
- Are missing `PriorityLevelCode` values expected in normal data capture?
- Is missing `recovery_time` expected, optional, or a data-quality issue?
- Is missing `theat_anae_1_national_code` expected for some cases, or does it indicate a coding gap?

## Staff and consultant fields

- Do `session_consultant`, `listing_cons_code`, `theat_surg_1_national_code`, and `theat_anae_1_national_code` represent different roles?
- Should any of these staff/consultant fields match in specific cases?
- Should high-cardinality consultant/staff-code columns be used in modelling, grouped, or excluded?

## Modelling direction

- Is `operation_length_mins` the best first modelling target for theatre scheduling?
- Should secondary targets such as `is_overrun`, `overrun_minutes`, and `underrun_minutes` also be modelled?
- Are there local operational rules that define what counts as an important overrun or bottleneck?

## Overrun and bottleneck definition

- What threshold should define a meaningful overrun?
- Should very small overruns, such as 1-5 minutes, be treated as normal variation rather than real delay?
- Is it reasonable to classify overruns as:
  - close to planned: -5 to +5 minutes
  - minor overrun: 6 to 15 minutes
  - moderate overrun: 16 to 30 minutes
  - major overrun: more than 30 minutes
- Should bottlenecks be defined using both overrun frequency and overrun size?
- Is the following rule acceptable for EDA: a stronger bottleneck should have high case count, high overrun rate, and high median or total overrun minutes?
- Should we use `median_overrun_minutes`, `total_overrun_minutes`, and `case_count` together instead of relying only on overrun rate?
- Should under-used planned time also be analysed using `underrun_minutes`, because finishing much earlier than planned may indicate inefficient scheduling?
