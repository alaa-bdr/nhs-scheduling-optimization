# Supervisor Presentation Notes

## 1. Project Aim

The aim of this project was to understand why theatre operations finish earlier or later than planned, and to test whether machine-learning models can improve scheduling predictions.

The analysis followed the notebooks in this order:

1. `nbt_smallset_preprocessing.ipynb`
2. `nbt_smallset_data_cleaning.ipynb`
3. `nbt_smallset_statistical_analysis.ipynb`
4. `nbt_duration_error_regression.ipynb`
5. `nbt_operation_length_regression.ipynb`
6. `nbt_meaningful_overrun_classification.ipynb`
7. `nbt_modeling_summary.ipynb`

The three final prediction questions were:

- Will the operation meaningfully overrun?
- How long will the operation take?
- How many minutes should be added to or subtracted from the planned duration?

## 2. Preprocessing Notebook

Notebook: `notebooks/nbt_smallset_preprocessing.ipynb`

### Question

What is inside the raw NBT theatre dataset, and what data-quality issues must be handled before analysis?

### Method

The notebook first inspected the raw dataset structure, missing values, duplicate rows, numeric variables, categorical variables, procedure codes, staff codes, theatre-room features, session descriptions, and time columns.

Important sections:

- Section 3: Basic structure
- Section 4: Data quality checks
- Section 5: Numeric variable inspection
- Section 6: Categorical and coded variable inspection
- Section 7: Feature extraction
- Section 8: Time column inspection and provisional duration validation
- Section 10: EDA on cleaned data

### Main Results

The raw/preprocessed dataset had 14,918 rows and 66 columns before final cleaning.

Section 4 found seven exact duplicate rows. These were removed later in the cleaning pipeline.

Section 5 showed that planned duration and recorded operation duration were usable, but some duration errors were very large. These extreme records were not deleted automatically. Instead, they were flagged for review.

Section 6 showed that procedure codes were mostly complete and internally consistent, but there were 1,038 unique procedure codes. Because this was too detailed for modelling, procedure information was grouped into more stable features:

- `procedure_code_category`
- `procedure_code_group`

Section 7 extracted useful features from existing fields, including:

- procedure hierarchy
- session specialty
- theatre-room features
- provisional start-hour information

Section 8 showed that reconstructed time features were useful for exploratory checks, but not fully validated. Therefore, raw and reconstructed timestamps were not used as primary predictors.

### EDA Bottleneck Findings

The bottleneck analysis is mainly in Section 10 of `nbt_smallset_preprocessing.ipynb`. The notebook does not label the charts as "Figure 1", "Figure 2", etc. Therefore, in the presentation I should refer to the notebook section and chart title/topic.

The main bottleneck question was:

Which parts of the scheduling process show repeated evidence of pressure, underestimation, or unstable planned duration?

The notebook did not define a bottleneck using one number only. It used several pieces of evidence together:

- case volume
- meaningful-overrun rate
- meaningful-underrun rate
- median duration error
- recorded operation length
- theatre occupancy, only as provisional evidence
- whether the pattern repeats across procedure, room, time, or pathway

This is important because a high overrun percentage alone can be misleading if the category has very few cases. A useful bottleneck candidate should usually have both enough cases and a meaningful duration problem.

#### Evidence 1: Planned Versus Actual Duration

Reference: Section 10.4, "Planned versus actual duration".

This chart compares `ExpectedDurationMins` with `operation_length_mins`.

What the graph proves:

- There is a positive relationship: longer planned operations are usually longer recorded operations.
- However, the spread around the diagonal line is wide.
- Points above the diagonal are overruns.
- Points below the diagonal are underruns.
- Planned durations often appear in repeated booking increments, while actual duration varies more continuously.

How to explain it:

"The planned duration is useful, but it is not accurate enough for every individual case. Some operations are booked too short and some too long. This is why we later model both operation length and duration error."

Key result from the notebook:

- median planning error was about -8 minutes
- mean planning error was about -0.5 minutes
- this means the average looks balanced, but individual cases still vary widely

So the first bottleneck is not one theatre room. It is the uncertainty in planned-versus-recorded duration.

#### Evidence 2: Operation Length By Category

Reference: Section 10.5, especially:

- Section 10.5.1, "Operation-length distribution by ordered clinical category"
- Section 10.5.4, "Comparable categorical outcome dashboards"

These charts compare operation length, overrun, underrun, and median error across categories such as admission type, intended management, anaesthetic type, theatre area, specialty, ASA, and priority.

What the graphs prove:

- Operation duration is strongly linked to case mix.
- Emergency and inpatient pathways behave differently from elective waiting-list and day-case activity.
- General anaesthetic and combined anaesthetic cases tend to have longer recorded durations.
- Some categories show high overrun rates, while others show high underrun rates.

Important examples from Section 10.5.4:

- Elective waiting-list cases were the largest group and often finished under plan.
- Emergency A&E cases had longer duration and positive median error.
- Inpatient operations were longer and had more meaningful overruns than day cases.
- Day cases had many meaningful underruns, suggesting conservative booking or different case mix.
- Gynae and Hybrid Theatre showed stronger pressure patterns than some lower-duration areas.
- Plastic Minor had short operations and high meaningful-underrun proportions.

How to explain it:

"The bottleneck is not just time lost inside theatre. It is also booking calibration. Some pathways appear underestimated, while others appear conservatively booked."

This supports the modelling decision to include:

- admission type
- intended management
- anaesthetic type
- procedure group/category
- expected duration

#### Evidence 3: Theatre-Flow Bottleneck EDA

Reference: Section 10.6, "Bottleneck EDA setup".

Important chart sections:

- Section 10.6.1, "Recorded operation length and provisional theatre-flow profile"
- Section 10.6.2, "Provisional flow patterns by operational category"
- Section 10.6.3, "Assumption-sensitivity checks for stage contribution and delay signals"

These charts looked at provisional theatre-flow features such as:

- theatre occupancy
- theatre-to-anaesthetic start
- anaesthetic-to-incision
- incision-to-closure
- closure-to-operation end

What the graphs suggest:

- Different areas and rooms have different recorded operation lengths.
- Hybrid Theatre had a longer provisional profile.
- Some reconstructed stage medians varied by room, procedure, anaesthetic type, and duration status.

But the notebook is careful:

- these stage durations come from reconstructed timestamps
- the reconstruction uses assumptions
- passing the reconstruction rule does not prove the inferred times are correct
- these figures are sensitivity evidence, not confirmed bottleneck proof

How to explain it:

"The theatre-flow figures are useful for discussion with NBT, but I do not use them as primary evidence. They tell us where to ask operational questions, not where to declare a confirmed bottleneck."

This is why raw timing and stage-duration columns were dropped from the final modelling dataset.

#### Evidence 4: Start-Time And Time-of-Day Pressure

Reference: Section 10.7, "Start-time and delay patterns", and Section 10.11.8, "Duration-status dashboard by time and case type".

Important chart topic:

- "Hourly status-rate and median-error figures"

What the graphs show:

- High proportional overrun appears overnight and after 20:00, but these periods have low case volume.
- In the busy daytime period, 09:00 and 14:00 show comparatively greater overrun pressure.
- Around 15:00-16:59, the charts show more underrun.
- Median duration error is shown alongside rates, so small differences are not exaggerated.

How to explain it:

"Time of day appears to contain some scheduling signal. However, because operation start hour was reconstructed, I kept it as a sensitivity feature rather than a primary predictor."

This was later confirmed in the modelling notebooks:

- for `duration_error_mins`, start hour improved MAE from 31.11 to 30.68
- for `operation_length_mins`, start hour improved MAE from 30.62 to 30.38
- for `meaningful_overrun_flag`, start hour improved PR-AUC from 0.797 to 0.801

So the graphs and models agree that start hour may matter, but it needs validation.

#### Evidence 5: Theatre-Room Pressure

Reference: Section 10.8, "Theatre and start-time bottleneck checks", and Section 10.8.1, "Theatre-room pressure profile".

These charts examine exact theatre rooms using case volume, overrun rate, underrun rate, and median error.

What the graphs show:

- Some rooms have higher apparent overrun pressure.
- Some rooms have more underrun, suggesting conservative booking or different case mix.
- High-volume rooms give more stable evidence than low-volume rooms.

But the interpretation is cautious:

- rooms are not randomly assigned
- different rooms handle different specialties and procedures
- a high room overrun rate may reflect case mix, not room inefficiency

How to explain it:

"Theatre room is useful for EDA, but it should not be interpreted as room performance. It identifies where further case-mix-adjusted analysis is needed."

This also explains why `TheatreRoom` was tested in modelling but not selected in the final primary model. Once procedure and other preoperative variables were included, `TheatreRoom` did not improve prediction enough.

#### Evidence 6: Procedure, Consultant, And Room Interactions

Reference: Section 10.11, "Deeper visual relationship EDA".

Important chart sections:

- Section 10.11.4, "Three-variable overrun and underrun relationship plots"
- Section 10.11.5, "Operational drilldowns: room, consultant, priority, and theatre-flow features"
- Section 10.11.6, "Consultant working-time and duration-status patterns"
- Section 10.11.7, "Procedure-group timing and bottleneck analysis"
- Section 10.11.7.2.1, "Official OPCS category: duration, occupancy, status, and operational context"

These figures are stronger than simple one-variable charts because they combine multiple dimensions:

- procedure with theatre area
- procedure with room
- consultant with procedure
- consultant with theatre room
- procedure group with start hour
- expected duration, actual duration, and error together

What the figures prove:

- procedure group is one of the strongest and most repeated sources of variation
- some procedure-room combinations show repeated pressure
- some consultant-room or consultant-procedure combinations show high overrun patterns
- these patterns need case-mix adjustment before interpretation

How to explain it:

"The deeper EDA shows that bottlenecks are interaction problems. A room, consultant, or time period cannot be judged alone, because the procedure type and case mix change the interpretation."

This supports using procedure hierarchy in the final model:

- `procedure_code_group`
- `procedure_code_category`

It also supports excluding raw consultant identifiers from the final model because they are governance-sensitive, high-cardinality, and not safe to interpret as performance measures.

#### Evidence 7: Executive Bottleneck Evidence Dashboard

Reference: Section 10.14.1, "Executive bottleneck evidence dashboard".

This is the strongest summary figure for the bottleneck story.

The dashboard combines:

- case volume
- meaningful-overrun rate
- total overrun minutes
- median duration error
- median theatre occupancy

What the dashboard proves:

- bottleneck candidates should be prioritised when they combine volume, overrun rate, positive median error, and long duration or occupancy
- the dashboard avoids creating one misleading performance score
- it separates frequency, severity, and resource use

How to explain it:

"The dashboard is not a ranking of staff or theatres. It is a prioritisation tool. It tells us which categories deserve deeper review because they combine enough activity with evidence of planning pressure."

This dashboard supports the final modelling direction because the same important signals reappear:

- expected duration
- procedure type
- anaesthetic type
- intended management
- admission pathway
- timing sensitivity

#### Final Bottleneck Conclusion

The EDA does not identify one single confirmed bottleneck. Instead, it shows several candidate bottleneck mechanisms:

1. Planning uncertainty: expected duration is useful but not accurate enough for all individual cases.
2. Procedure complexity: procedure category and group repeatedly explain duration and overrun differences.
3. Pathway differences: emergency and inpatient cases appear less predictable than elective/day-case work.
4. Anaesthetic context: general or combined anaesthetic cases are linked with longer operation times.
5. Location pressure: some theatre areas/rooms show pressure, but this overlaps with procedure mix.
6. Time-of-day patterns: start hour may matter, especially around some daytime periods, but it is provisional.
7. Data-quality pressure: missing priority and missing anaesthetic type are associated with different overrun rates.

The safest presentation statement is:

"The bottleneck evidence points to scheduling calibration rather than one isolated theatre delay. The strongest repeated signals are procedure type, expected duration, anaesthetic type, admission/intended-management pathway, and some provisional time/location interactions. Theatre-room and consultant patterns are useful for prioritisation, but they must not be presented as performance rankings without case-mix adjustment and source-system validation."

## 3. Cleaning Notebook

Notebook: `notebooks/nbt_smallset_data_cleaning.ipynb`

### Question

Which rows and columns should remain in the final analysis dataset?

### Method

The cleaning notebook applied reproducible row and column decisions through the reusable pipeline code.

Important sections:

- Section 1: Cleaning decisions
- Section 2.1: Row-level cleaning audit
- Section 3: Column-exclusion audit
- Section 4: Create the cleaned analysis dataset
- Section 5: Missingness after column selection
- Section 6: Outcome and predictor separation

### Row Cleaning

Seven exact source-level duplicate rows were removed.

The 12 records flagged by `duration_timing_review_flag` were retained in the primary dataset. They were not deleted because the flag means "review needed", not "definitely wrong". These records were later tested in sensitivity analysis.

### Column Decisions

The final cleaned dataset kept 14,911 rows and 22 columns.

The column-selection procedure was not based only on missingness. A column was dropped if it was unsafe, duplicated other information, not available before the operation, too detailed for stable modelling, or likely to create leakage.

The decision process was:

1. First create useful derived features from the raw data.
2. Then remove raw fields that had been safely replaced by cleaner features.
3. Keep target/outcome columns for analysis, but never use them as predictors.
4. Keep only predictors that could reasonably be known before or at scheduling time.
5. Keep provisional timing information only for sensitivity analysis.

Dropped columns included:

- free text: `SessionIDdesc`, `theatre_notes`
- raw procedure code and procedure description
- staff identifiers
- unvalidated session fields
- duplicate theatre-room components such as `theatre_area`, room prefix, room number, and IR indicator
- raw and reconstructed timestamps
- provisional stage-duration features
- raw admission, management, and priority codes after readable labels were created
- duplicate outcome columns

Important retained columns included:

- `TheatreRoom`
- `ExpectedDurationMins`
- `age_at_operation`
- `ASAScore`
- `admission_type_label`
- `intended_management_label`
- `priority_level_label`
- `procedure_code_category`
- `procedure_code_group`
- `session_specialty`
- `operation_start_hour`
- `duration_status`
- `duration_timing_review_flag`

### Why These Decisions Were Made

Free text and staff identifiers were removed for governance, privacy, interpretability, and overfitting reasons.

Detailed procedure codes were replaced by grouped procedure features because the raw code had too many categories.

Raw timestamps and end-time information were removed because they could create leakage. `operation_start_hour` was retained only for sensitivity testing.

Missing clinical categories were not guessed or replaced with common values.

### Keep/Drop Explanation For Important Columns

`ExpectedDurationMins` was kept because it is the hospital's planned duration and is known before the operation. It is also the current benchmark that the model tries to improve.

`operation_length_mins`, `duration_error_mins`, and `meaningful_overrun_flag` were kept as targets/outcomes, but they were blocked from entering the predictor matrix.

`actual_proc_1_procedure_code` was dropped because it had too many detailed values. Instead, we kept `procedure_code_group` and `procedure_code_category`. This keeps procedure information but reduces sparsity and overfitting.

`ProcedureDescription` was dropped because it is high-cardinality text. It would require a separate text-processing method and stronger governance.

`TheatreRoom` was kept in the cleaned dataset because it is useful for EDA and sensitivity testing. However, it was not selected in the final primary model because it did not improve prediction after procedure and case-mix variables were included.

`theatre_area`, theatre prefix, room number, and IR indicator were dropped because they overlap with `TheatreRoom`. Keeping all of them would represent location several times and make interpretation unclear.

Staff identifiers were dropped because they are sensitive, high-cardinality, and could cause the model to learn individual allocation patterns rather than general scheduling rules.

Raw start/end timestamps and reconstructed stage durations were dropped from the primary model. End time is leakage because it happens after the operation. Start hour was kept only as `operation_start_hour` for sensitivity testing because it was reconstructed.

`ASAScore` was kept because it is clinically meaningful and available before surgery, but it was treated as categorical/ordinal clinical information rather than a normal continuous number.

## 4. Statistical Analysis Notebook

Notebook: `notebooks/nbt_smallset_statistical_analysis.ipynb`

### Question

Which differences in operation duration and overrun status are statistically supported?

Important sections:

- Section 3: Planned versus recorded duration
- Section 4: Operation-duration differences across groups
- Section 5: Duration-status associations
- Section 6: Age and recorded sex
- Section 7: Missingness analysis
- Section 8: Adjusted meaningful-overrun model
- Section 8.1: Theatre-room by procedure-group interaction screening
- Section 9: Conclusions and decision rules

### Planned Versus Actual Duration

Among 14,563 complete planned-versus-recorded pairs, the median planning error was -8 minutes. This means the typical operation finished slightly shorter than planned.

The mean error was close to zero: -0.49 minutes. This means positive and negative errors roughly cancelled out overall.

But individual cases were still uncertain. The approximate 95% agreement limits were about -134 to +133 minutes. So even if the average bias was small, individual cases could differ a lot from their planned duration.

### Group Differences

Section 4 used Kruskal-Wallis tests because operation durations were skewed.

The strongest unadjusted differences in operation length were:

- procedure category: epsilon-squared 0.638
- theatre room: 0.312
- anaesthetic type: 0.298
- intended management: 0.242
- specialty: 0.193
- ASA: 0.114
- admission type: 0.041
- priority: 0.016

This shows that procedure and operational context were much more informative than priority alone.

### Duration-Status Associations

Section 5 tested associations with meaningful overrun, within tolerance, and meaningful underrun.

The strongest associations were:

- procedure category: Cramer's V 0.481
- theatre room: 0.407
- specialty: 0.308
- admission type: 0.306
- intended management: 0.269
- anaesthetic type: 0.204
- ASA: 0.150

Priority was not significant after correction.

### Age and Sex

Section 6 showed that age and recorded sex were not major bottleneck drivers.

Age had a very weak relationship with operation length and planning error. Sex had a statistically significant but tiny difference: median operation length was 71 minutes for code 1 and 69 minutes for code 2.

### Missingness Analysis

Section 7 showed that missingness was not random.

For example:

- meaningful overrun was 44.0% when priority was missing versus 15.0% when priority was recorded
- meaningful overrun was 50.2% when anaesthetic type was missing versus 30.5% when recorded

This is why the modelling used missing-aware handling instead of simply dropping all incomplete rows.

### Adjusted Model

Section 8 fitted an adjusted logistic model for meaningful overrun.

The model used 14,563 cases and 4,569 meaningful overruns. McFadden pseudo-R2 was about 0.296, meaning the included variables improved fit over an intercept-only model.

The model confirmed that unadjusted room effects must be interpreted carefully. Some apparent room differences weakened after adjusting for case mix.

### Room by Procedure Screening

Section 8.1 showed that some procedure groups had different overrun rates across rooms.

Examples:

- `V2`: Cramer's V 0.619, overrun rates from 7.9% to 83.3% across rooms
- `S5`: Cramer's V 0.285, overrun rates from 34.7% to 91.2%
- `M1`: Cramer's V 0.190

These are screening results, not causal room-performance rankings.

## 5. Modelling Method Used In All Three Modelling Notebooks

Modelling notebooks:

- `notebooks/nbt_duration_error_regression.ipynb`
- `notebooks/nbt_operation_length_regression.ipynb`
- `notebooks/nbt_meaningful_overrun_classification.ipynb`

### General Method

Each modelling notebook followed the same structure:

1. Build the target-eligible population.
2. Remove outcome/leakage predictors.
3. Split into development and untouched test data.
4. Compare missing-data strategies.
5. Compare feature configurations.
6. Compare algorithms.
7. Tune the two leading algorithms.
8. Evaluate the final model once on the untouched test set.
9. Run sensitivity analyses.
10. Export results and save evidence.

The modelling process was deliberately staged. We did not choose the best model first. We first chose the best data treatment, then the best feature configuration, then the best algorithm.

The modelling hierarchy was:

1. Compare missing-data strategies using the same baseline model.
2. Freeze the best missing-data strategy.
3. Compare feature configurations using the same model and same folds.
4. Freeze the best feature configuration.
5. Compare algorithms fairly on the same data.
6. Tune the two strongest algorithms.
7. Evaluate only once on the untouched test set.

This protects the test set. The test set was not used repeatedly to make modelling decisions.

### Missing-Value Strategy

The best strategy was missing-aware preprocessing.

This means:

- categorical missing values became `Missing/not recorded`
- `age_at_operation` used median placeholder plus a missingness indicator
- `ExpectedDurationMins` was required and not imputed
- target values were never imputed
- leakage/outcome columns were never used as predictors

This was implemented inside sklearn pipelines so that imputation and encoding were fitted only on training folds.

This is important because missing values can carry information. For example, missing priority was associated with a different overrun rate in the statistical notebook. If we simply dropped all missing rows, we would analyse a selected subgroup rather than the full operational population.

For categorical variables, the model sees missingness as its own category:

`Missing/not recorded`

This does not mean that missing ASA or missing priority is treated as a normal clinical value. It means the model is told that the value was not recorded.

For numeric variables, only ordinary numeric predictors such as `age_at_operation` can use median imputation with a missingness indicator.

`ExpectedDurationMins` is different. It is a required planning variable and was not imputed. If planned duration is missing, the row is excluded for modelling because a median planned duration has no safe scheduling meaning.

### Feature Configuration

The winning feature configuration was `Both procedure levels`.

The final primary predictors were:

- `ExpectedDurationMins`
- `sex_national_code`
- `age_at_operation`
- `ASAScore`
- `anaesthetic_desc`
- `admission_type_label`
- `intended_management_label`
- `priority_level_label`
- `procedure_code_group`
- `procedure_code_category`

`TheatreRoom` was tested but did not improve the winning primary model.

`operation_start_hour` was tested only as sensitivity.

The feature configurations tested different possibilities:

- expected duration only
- clinical and planning variables
- procedure group only
- procedure category only
- both procedure group and category
- full approved set including location
- full approved set without location

The winner was `Both procedure levels`. This means the model benefited from keeping both:

- `procedure_code_group`, which is broader and more stable
- `procedure_code_category`, which is more clinically specific

Theatre room was tested. It was useful descriptively in the EDA, but it did not improve the final primary model enough once procedure, expected duration, anaesthetic type, admission pathway, intended management, ASA, age, sex, and priority were included.

### Algorithms Compared

The models compared were:

- Dummy benchmark
- Linear/logistic regression
- Decision tree
- Random forest
- SVR/SVC
- XGBoost
- Neural network

XGBoost was the best model for all three targets.

The model comparison used cross-validation on the development data. This means the data was repeatedly split into training and validation folds, so every algorithm was compared under the same conditions.

The dummy benchmark was included to show the value of modelling. For regression, the dummy model predicts a simple central value. For classification, the dummy model predicts based on class prevalence. A useful model should beat these simple baselines.

XGBoost performed best because it can handle nonlinear relationships and interactions. This matters because theatre scheduling is not linear. Procedure type, expected duration, anaesthetic type, and management pathway interact with each other.

The neural network was tested, but it did not beat XGBoost. This is reasonable because the dataset is tabular, and tree-based boosting methods often perform very strongly on tabular clinical/operational data.

### Why Some EDA Signals Look Different From Model Importance

EDA and model feature importance answer different questions.

EDA asks:

"Does this variable show a visible or statistical relationship with the outcome on its own?"

Model feature importance asks:

"How much extra predictive value does this variable add after the model already knows all the other variables?"

This explains the ASA result.

In EDA, `ASAScore` appeared important because operation duration and overrun rates differed across ASA groups. That is a real descriptive association.

In the final model, `ASAScore` had lower feature importance because some of its signal overlaps with stronger predictors:

- procedure type
- expected duration
- anaesthetic type
- intended management
- admission pathway
- age

For example, higher ASA patients may be more likely to have complex procedures, inpatient pathways, or general anaesthetic. Once the model already knows those variables, ASA adds less new information.

So the correct interpretation is:

`ASAScore` is clinically meaningful, but it was not one of the strongest unique scheduling predictors in the final model.

This does not mean ASA is unimportant medically. It only means that, for predicting duration/overrun in this dataset, much of its predictive information is already captured by other variables.

The same logic applies to `TheatreRoom`. Theatre room looked important in EDA, but after procedure and pathway variables were included, it added little extra predictive power.

## 6. Duration-Error Modelling Notebook

Notebook: `notebooks/nbt_duration_error_regression.ipynb`

### Question

How many minutes longer or shorter than planned will the operation be?

Target:

`duration_error_mins = operation_length_mins - ExpectedDurationMins`

### Result

Best model: XGBoost

Final test performance:

- MAE: 31.11 minutes
- RMSE: 50.71 minutes
- R2: 0.424
- within 10 minutes: 29.1%
- within 20 minutes: 52.2%
- within 30 minutes: 67.2%
- within 60 minutes: 86.9%

### Interpretation

The model improves over a simple benchmark, but this target is difficult because it predicts the error in the existing hospital estimate. Much of the predictable information is already inside `ExpectedDurationMins`, so the remaining error contains more noise.

Top predictive features:

- `ExpectedDurationMins`
- `anaesthetic_desc`
- `procedure_code_group`
- `procedure_code_category`
- `intended_management_label`

### Start-Hour Sensitivity

Adding `operation_start_hour` improved the result:

- MAE changed from 31.11 to 30.68 minutes
- R2 changed from 0.424 to 0.457

But this remains sensitivity-only because start hour was reconstructed and not directly validated.

### What This Result Means

`duration_error_mins` is operationally useful because it predicts the correction to the booked time.

For example, if the hospital booked 90 minutes and the model predicts a +20 minute error, the corrected estimate would be about 110 minutes.

However, this target is harder than predicting operation length. It asks the model to predict where the hospital estimate is wrong. That remaining error includes unrecorded complexity, workflow variation, data-quality issues, and random clinical variation.

This is why R2 was lower for duration error than for operation length.

## 7. Operation-Length Modelling Notebook

Notebook: `notebooks/nbt_operation_length_regression.ipynb`

### Question

How many minutes will the operation take?

Target:

`operation_length_mins`

### Result

Best model: XGBoost

Final test performance:

- MAE: 30.62 minutes
- RMSE: 49.50 minutes
- R2: 0.712
- within 10 minutes: 30.0%
- within 20 minutes: 52.4%
- within 30 minutes: 67.4%
- within 60 minutes: 87.1%

### Interpretation

This was the strongest regression target. It performed better than `duration_error_mins` because actual operation length has clearer structure from planned duration, procedure type, anaesthetic type, and management pathway.

Top predictive features:

- `ExpectedDurationMins`
- `anaesthetic_desc`
- `intended_management_label`
- `procedure_code_group`
- `procedure_code_category`

### Start-Hour Sensitivity

Adding `operation_start_hour` improved the result slightly:

- MAE changed from 30.62 to 30.38 minutes
- R2 changed from 0.712 to 0.720

Again, this remains sensitivity-only because start hour was reconstructed.

### What This Result Means

This model is the best if the aim is to predict the total expected operation length directly.

It achieved R2 0.712, meaning it explained a large part of the variation in recorded operation length. This makes sense because the model can use the hospital's planned duration plus procedure and case-mix information.

The model also beats the hospital expected-duration benchmark, showing that the machine-learning model adds information beyond the existing booking estimate.

## 8. Meaningful-Overrun Classification Notebook

Notebook: `notebooks/nbt_meaningful_overrun_classification.ipynb`

### Question

Will the operation exceed its planned duration by more than the working tolerance?

Target:

`meaningful_overrun_flag`

### Result

Best model: XGBoost

Final test performance:

- accuracy: 0.811
- balanced accuracy: 0.802
- ROC-AUC: 0.887
- PR-AUC: 0.797
- precision: 0.671
- recall: 0.778
- F1: 0.720
- Brier score: 0.135
- selected threshold: 0.54

### Interpretation

The classifier was useful because it identified a meaningful overrun risk rather than predicting every small difference in minutes.

Recall of 0.778 means the model found about 78% of true meaningful overruns on the test set.

Precision of 0.671 means that when the model predicted a meaningful overrun, about 67% were actually meaningful overruns.

Top predictive features:

- `ExpectedDurationMins`
- `anaesthetic_desc`
- `procedure_code_group`
- `intended_management_label`
- `procedure_code_category`

### Start-Hour Sensitivity

Adding `operation_start_hour` improved the result slightly:

- PR-AUC changed from 0.797 to 0.801
- recall changed from 0.778 to 0.785

It remains sensitivity-only because the start hour was reconstructed.

### What This Result Means

This model is useful if the operational question is risk screening:

"Which cases are likely to overrun enough to matter?"

Accuracy alone is not enough here because the classes are not perfectly balanced. That is why the notebook also reports PR-AUC, recall, precision, F1, ROC-AUC, and Brier score.

Recall was about 0.778, meaning the model identified most true meaningful overruns. Precision was about 0.671, meaning not every predicted overrun was a real overrun. This trade-off is expected in a safety-oriented scheduling model: missing a true overrun can be more operationally costly than flagging some extra cases for review.

## 9. Modelling Summary Notebook

Notebook: `notebooks/nbt_modeling_summary.ipynb`

### Final Winner Table

All three targets selected XGBoost.

| Target | Winning missing strategy | Winning feature configuration | Winning model | Main result |
|---|---|---|---|---|
| `duration_error_mins` | Missing-aware, priority retained | Both procedure levels | XGBoost | MAE 31.11, R2 0.424 |
| `operation_length_mins` | Missing-aware, priority retained | Both procedure levels | XGBoost | MAE 30.62, R2 0.712 |
| `meaningful_overrun_flag` | Missing-aware, priority retained | Both procedure levels | XGBoost | PR-AUC 0.797, recall 0.778 |

### Best Overall Modelling Target

The strongest regression model was `operation_length_mins`, because it achieved R2 0.712.

The most operationally direct correction target was `duration_error_mins`, but it was harder to predict.

The most useful risk-screening target was `meaningful_overrun_flag`, because it identifies cases likely to exceed the working tolerance.

## 10. Saved Final Models

The final fitted pipelines were saved as full sklearn pipelines, including preprocessing plus XGBoost.

Saved files:

- `result/modeling/final_models/duration_error_mins/final_pipeline.joblib`
- `result/modeling/final_models/operation_length_mins/final_pipeline.joblib`
- `result/modeling/final_models/meaningful_overrun_flag/final_pipeline.joblib`

Each model folder also includes `metadata.json`.

## 11. Main Limitations

The analysis does not prove causation.

Theatre-room differences are not room-performance rankings because rooms receive different procedures and case mixes.

Start-hour and theatre-flow features are provisional because they depend on reconstructed timestamps.

Staff identifiers were excluded for governance and overfitting reasons.

Priority has high missingness, so priority findings must be interpreted carefully.

External or temporal validation is needed before any operational deployment.

## 12. Simple Ending For Presentation

The main conclusion is that theatre scheduling error is predictable to a useful extent, but not perfectly. Procedure type, expected duration, anaesthetic type, and intended management were consistently the strongest predictors. XGBoost gave the best performance across all three prediction tasks. The direct operation-length model had the strongest numerical performance, while the overrun classifier is easiest to use for identifying high-risk cases. Start hour improved performance slightly, but it should remain sensitivity-only until the reconstructed timestamp is validated.
