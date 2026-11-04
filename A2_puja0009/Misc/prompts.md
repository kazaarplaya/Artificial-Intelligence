# A2 AI Declaration

This collection covers modelling, implementation, integration, and conceptual questions. Assistant responses are included where the source files supplied them. The final section contains prompts only because the source did not include their responses.

## Network-stress modelling and prototyping

### Exchange 1

##### Prompt

> For the electricity_transformed.parquet file, I want to conduct electricity network stress prediction classification, e.g. predict network stress risk as low, medium, high or critical. How could you define what low, medium, high or critical network stress is?

##### Output

#### Progress update 1

I’ll inspect the parquet’s columns, units, and distributions first, then suggest stress classes that are defensible for this dataset and avoid target leakage.

#### Progress update 2

The key limitation is already clear: this file has regional demand, generation, flows, prices, renewables, and curtailment, but no line thermal loading, reserve margin, voltage/frequency, outages, or constraint equations. So the label should be presented as a transparent “stress proxy,” not literal engineering network security. I’m quantifying suitable thresholds now.

#### Final response 3

The parquet file does not contain direct engineering stress measurements such as line loading, reserve margin, frequency, voltage, outages, or constraint violations. Therefore, I would define the target as a **regional network-stress proxy**, combining:

- Supply adequacy stress: high demand, high prices, import dependence, rapid demand changes.
- Congestion/surplus stress: curtailment, exports, negative prices, rapid renewable changes.

This distinction matters because both electricity shortages and renewable oversupply can stress the network.

#### Recommended definition

For each indicator, assign severity points using thresholds calculated from the chronological training period only:

| Points | Training percentile | Interpretation |
|---:|---:|---|
| 0 | Below 75th | Normal |
| 1 | 75th–90th | Elevated |
| 2 | 90th–97th | Severe |
| 3 | At or above 97th | Extreme |

Using the first 70% of this dataset as an example, the thresholds are:

| Indicator | Elevated | Severe | Extreme |
|---|---:|---:|---:|
| Gross demand | 9,324 MW | 10,253 MW | 11,296 MW |
| Price | $124.57/MWh | $228.57/MWh | $353.01/MWh |
| Imports ÷ gross demand | 12.1% | 15.5% | 19.2% |
| Absolute hourly demand ramp | 487 MW | 732 MW | 1,060 MW |
| Curtailment | 52.9 MW | 217.4 MW | 491.5 MW |
| Exports ÷ gross demand | 2.7% | 5.1% | 7.4% |
| Absolute hourly renewable ramp | 523 MW | 950 MW | 1,389 MW |

Then calculate two scores:

```text
adequacy_score =
    0.35 × demand_severity
  + 0.30 × positive_price_severity
  + 0.20 × import_share_severity
  + 0.15 × demand_ramp_severity

congestion_score =
    0.55 × curtailment_severity
  + 0.20 × export_share_severity
  + 0.15 × renewable_ramp_severity
  + 0.10 × negative_price_severity

stress_score = max(adequacy_score, congestion_score)
```

Negative-price severity could be:

- 0: price ≥ $0/MWh
- 1: $0 to −$50/MWh
- 2: −$50 to −$100/MWh
- 3: below −$100/MWh

Finally:

| Class | Score | Meaning |
|---|---:|---|
| **Low** | `< 0.75` | Normal conditions; no material stress indicators |
| **Medium** | `0.75–<1.50` | Elevated demand, flows, ramps, or curtailment |
| **High** | `1.50–<2.25` | Severe stress in one dimension or several concurrent elevated indicators |
| **Critical** | `≥ 2.25` | Multiple extreme indicators suggesting serious adequacy or congestion risk |

Applied to the complete file, this example definition produces approximately:

- Low: 65.2%
- Medium: 23.6%
- High: 9.6%
- Critical: 1.6%

That is a reasonable imbalance for a risk-classification problem; critical conditions should be uncommon.

#### Prediction setup

For a useful forecasting problem, define the target as the worst stress class in a future horizon:

```text
target_1h = stress class at t + 1 hour
target_6h = maximum stress class from t + 1 through t + 6 hours
```

Only information available at time `t` should be used as model input. Calculate thresholds from the training period and keep them unchanged for validation and test data. Otherwise, future information leaks into the labels.

I would call the target `network_stress_proxy`, not confirmed network stress. For a genuine engineering label, the dataset would need to be joined with AEMO reserve/LOR notices, constraint violations, interconnector limits, outages, frequency events, or transmission loading data. The available measurements and their limitations are documented in [data_dictionary.md](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A1/docs/data_dictionary.md).

### Exchange 2

##### Prompt

> what features could be used to define adequacy and congestion

##### Output

From the available columns, I would use separate feature groups for **adequacy stress** and **congestion stress**.

#### Adequacy features

Adequacy concerns whether available supply can meet demand.

| Feature | Calculation | Why it indicates stress |
|---|---|---|
| `demand_gross` | Existing column | High total demand places more pressure on available supply |
| Demand intensity | Percentile or z-score of `demand_gross` | Makes demand comparable across seasons and years |
| Import dependency | `flow_imports / demand_gross` | High dependence on imports suggests limited local supply |
| Residual demand | `demand_gross - generation_renewable_with_storage` | Demand that must be supplied by conventional generation or imports |
| Import-adjusted residual demand | `demand_gross - generation_renewable_with_storage - flow_imports + flow_exports` | Approximate demand remaining for local non-renewable supply |
| Demand ramp | `demand_gross - demand_gross.shift(1)` | Rapid demand increases require generation to respond quickly |
| Renewable shortfall | `1 - renewable_with_storage_proportion / 100` | Low renewable contribution increases reliance on other supply |
| Price level | Positive `price` | High prices can proxy supply scarcity |
| Price volatility | Rolling standard deviation of `price` | Volatility may indicate unstable supply-demand balance |
| Import ramp | `flow_imports.diff()` | Rapidly increasing imports may indicate deteriorating local adequacy |

The strongest starting set would be:

```text
demand_gross
import_share
residual_demand
positive_price
demand_ramp
price_rolling_std_24h
```

Avoid calling the calculated residual demand a true “reserve margin.” The dataset contains generation output, but not available generation capacity or operational reserves.

#### Congestion features

Congestion concerns whether electricity can be transported or absorbed without constraints.

| Feature | Calculation | Why it indicates stress |
|---|---|---|
| `curtailment` | Existing column, with negatives treated as zero for scoring | Direct proxy for renewable output constrained from entering the network |
| Curtailment ratio | `curtailment / (generation_renewable + curtailment)` | Normalises curtailment relative to available renewable output |
| Export dependency | `flow_exports / demand_gross` | High exports can indicate excess regional supply needing transmission capacity |
| Total interconnector flow | `flow_imports + flow_exports` | High flows indicate greater use of the transmission interface |
| Renewable penetration | `renewable_with_storage_proportion` | High renewable output may coincide with network export or absorption constraints |
| Renewable ramp | `generation_renewable.diff()` | Rapid renewable changes can produce balancing and flow-management pressure |
| Curtailment ramp | `curtailment.diff()` | A sudden increase may indicate emerging constraints |
| Negative-price severity | `max(-price, 0)` | Negative prices can indicate surplus generation or limited export/storage capability |
| Curtailment persistence | Rolling count or mean of positive curtailment | Distinguishes sustained congestion from a single noisy observation |
| Congestion interaction | `curtailment_ratio × renewable_proportion` | Highlights curtailment during high-renewable periods |

The strongest starting set would be:

```text
curtailment
curtailment_ratio
export_share
renewable_with_storage_proportion
renewable_ramp
negative_price_severity
curtailment_rolling_mean_24h
```

#### Important modelling distinction

Use the contemporaneous values above to **define the target label**. Use their historical values to predict it:

```text
Features at time t:
    demand_lag_1h
    price_lag_1h
    curtailment_lag_1h
    rolling statistics
    calendar features

Target:
    stress class at t + 1 hour
```

Using `price`, `demand_gross`, or `curtailment` from the target hour as predictors would leak the answer into the model.

I would also retain separate `adequacy_score` and `congestion_score` columns even if the final target is:

```text
network_stress = max(adequacy_score, congestion_score)
```

That makes a high or critical prediction explainable as either supply scarcity or network congestion.

### Exchange 3

##### Prompt

> I want to create minimal new columns since this should have been done in the feature engineering step of A1, what would you recommend now

##### Output

I recommend adding only **one permanent column**: the future stress target. Calculate adequacy and congestion scores temporarily in the modelling notebook rather than saving every intermediate calculation.

Use existing columns:

```text
Adequacy:
40% demand_gross
35% price
25% flow_imports

Congestion:
60% curtailment
25% flow_exports
15% renewable_with_storage_proportion
```

Convert each input to severity points using training-only percentiles:

```text
< 75th percentile       = 0
75th–90th percentile    = 1
90th–97th percentile    = 2
>= 97th percentile      = 3
```

Then calculate:

```python
adequacy_score = (
    0.40 * demand_severity
    + 0.35 * price_severity
    + 0.25 * import_severity
)

congestion_score = (
    0.60 * curtailment_severity
    + 0.25 * export_severity
    + 0.15 * renewable_proportion_severity
)

stress_score = np.maximum(adequacy_score, congestion_score)
```

Map the temporary score to classes:

```python
stress_class = pd.cut(
    stress_score,
    bins=[-np.inf, 0.75, 1.50, 2.25, np.inf],
    labels=["low", "medium", "high", "critical"],
    right=False,
)
```

For one-hour-ahead prediction, save only:

```python
df["stress_target_1h"] = stress_class.shift(-1)
```

The final row will have no future target and should be removed during modelling.

Your existing lag and rolling columns already provide suitable predictors:

- `demand_lag_1h`, `demand_lag_24h`, `demand_lag_168h`
- `price_lag_*` and `price_rolling_*`
- `curtailment_lag_*` and `curtailment_rolling_*`
- `generation_renewable_lag_*`
- Calendar features

Do not include contemporaneous `demand_gross`, `price`, `flow_imports`, `curtailment`, or the temporary scores as predictors for the future target unless they are known at prediction time.

If you want one additional interpretability column, add `stress_driver` containing `"adequacy"` or `"congestion"`. Otherwise, the single `stress_target_1h` column is sufficient. This keeps A2 focused on target construction rather than repeating A1 feature engineering.

### Exchange 4

##### Prompt

> Please implement this in my A2 notebook

##### Output

#### Progress update 1

I’ll add a compact target-construction section to the A2 notebook. It will calculate the two scores in memory and persist only `stress_target_1h`, then show the fitted thresholds and class balance for verification.

#### Progress update 2

The notebook logic is in place and the earlier broken exploratory bucket code has been replaced. I’m executing it end-to-end now to verify the target construction, assertions, and resulting class balance.

#### Progress update 3

The notebook kernel runner stalled before producing output, so I stopped it safely. I’m switching to a lightweight validation of the notebook’s code and will clear stale outputs so the saved notebook does not display results from the replaced cells.

#### Final response 4

Implemented in [electricity-network-stress-.ipynb](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/notebooks/electricity-network-stress-.ipynb).

It now:

- Fits thresholds using the first chronological 70% of data.
- Calculates adequacy and congestion scores temporarily.
- Adds only `stress_target_1h` to `df_clean`.
- Creates `model_df` without the final unavailable target.
- Displays thresholds and class balance.
- Includes assertions confirming only one column was added.

Validated class distribution:

- Low: 58.72%
- Medium: 24.54%
- High: 10.75%
- Critical: 5.98%

The notebook is currently untracked by Git, so it will need to be added when committing.

### Exchange 5

##### Prompt

> Why are thresholds only fitted on the first 70%?

##### Output

The thresholds are fitted on the first 70% to prevent **future information leakage**.

Because the data is chronological:

- First 70% represents the training period.
- Remaining 30% represents future validation/test periods.

If percentiles were calculated from the complete dataset, observations from 2025–2026 would influence how earlier periods are labelled. The model would indirectly receive information about future demand, price, curtailment, and renewable conditions.

For example, renewable generation and curtailment increase over time. Using the full dataset’s 97th-percentile curtailment threshold could make genuinely extreme congestion during the earlier training period appear ordinary.

Keeping thresholds fixed also allows distribution changes to remain visible. The higher proportion of critical observations in later data may represent genuine growth or drift rather than being normalised away.

There is an important trade-off: these classes mean “stress relative to the training period,” not permanent engineering limits. The 70% value itself is not mandatory—it simply matches a conventional chronological 70/30 split. A cleaner workflow is:

1. Decide the exact train/validation/test dates.
2. Fit thresholds using only the training dates.
3. Apply the same thresholds unchanged to validation and test data.
4. Train using information available before each target hour.

If this exercise is only descriptive clustering rather than predictive modelling, using the entire dataset would be acceptable. For evaluating a future stress-prediction model, training-only thresholds are preferable.

### Exchange 6

##### Prompt

> Why didnt you use train_test_split then?

##### Output

Because this is time-series data, a normal random `train_test_split` would be inappropriate.

Random splitting would mix future observations into training and earlier observations into testing, causing:

- Future operating regimes to influence training.
- Very similar adjacent hours to appear in both sets.
- Lag and rolling features to overlap across sets.
- Unrealistically optimistic model performance.

The notebook currently calculates the chronological boundary directly:

```python
train_end = int(len(df_clean) * 0.70)
training_for_thresholds = df_clean.iloc[:train_end]
```

That is effectively a chronological 70/30 split, but it is currently used only to fit the label thresholds—not yet to create model training and test datasets.

If you want explicit datasets, use:

```python
split_index = int(len(model_df) * 0.70)

train_df = model_df.iloc[:split_index].copy()
test_df = model_df.iloc[split_index:].copy()
```

Alternatively, `train_test_split` can preserve chronological order using:

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    model_df,
    test_size=0.30,
    shuffle=False,
)
```

However, direct `.iloc` slicing makes the chronological intent more obvious. For model tuning, `TimeSeriesSplit` is better than repeatedly calling `train_test_split`:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```

So `train_test_split` was not omitted because splitting is unnecessary; it was omitted because target construction and model splitting are separate stages, and chronological slicing is safer and clearer here.

### Exchange 7

##### Prompt

> How to train and compare Random Forsest and XGBoost for the dataset, read only

##### Output

#### Progress update 1

I’ll keep the workspace unchanged and check the installed library versions so the example matches your environment. Then I’ll outline a leakage-safe chronological comparison using the existing target.

#### Final response 2

I made no changes. Your environment has scikit-learn 1.9.1, but `xgboost` is not currently installed.

For a fair comparison, use the same chronological 70/30 split, features, and evaluation metrics for both models.

#### 1. Prepare chronological data

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.utils.class_weight import compute_sample_weight

model_df = model_df.sort_values("timestamp").reset_index(drop=True)

# These are valid for a one-hour-ahead prediction because they are known at time t.
excluded_columns = ["timestamp", "stress_target_1h"]
feature_columns = [
    column for column in model_df.select_dtypes(include="number").columns
    if column not in excluded_columns
]

class_to_number = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

X = model_df[feature_columns]
y = model_df["stress_target_1h"].map(class_to_number).astype(int)

split_index = int(len(model_df) * 0.70)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training period:", model_df["timestamp"].iloc[0],
      "to", model_df["timestamp"].iloc[split_index - 1])
print("Test period:", model_df["timestamp"].iloc[split_index],
      "to", model_df["timestamp"].iloc[-1])
```

Do not use a shuffled split because neighbouring electricity observations and their rolling features are highly related.

#### 2. Train Random Forest

```python
random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

random_forest.fit(X_train, y_train)
rf_prediction = random_forest.predict(X_test)
```

`class_weight="balanced_subsample"` helps compensate for the relatively uncommon high and critical classes.

#### 3. Train XGBoost

XGBoost would first need to be installed later:

```powershell
.\.venv\Scripts\python.exe -m pip install xgboost
```

After installation:

```python
from xgboost import XGBClassifier

training_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

xgboost_model = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

xgboost_model.fit(
    X_train,
    y_train,
    sample_weight=training_weights,
)

xgb_prediction = xgboost_model.predict(X_test)
```

#### 4. Compare the models

Accuracy alone will be misleading because the dataset contains substantially more low-stress observations.

```python
import pandas as pd

class_names = ["low", "medium", "high", "critical"]

def evaluate_model(name, actual, predicted):
    return {
        "model": name,
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro"),
        "weighted_f1": f1_score(actual, predicted, average="weighted"),
        "ordinal_mae": mean_absolute_error(actual, predicted),
        "quadratic_kappa": cohen_kappa_score(
            actual,
            predicted,
            weights="quadratic",
        ),
    }

comparison = pd.DataFrame([
    evaluate_model("Random Forest", y_test, rf_prediction),
    evaluate_model("XGBoost", y_test, xgb_prediction),
]).set_index("model")

display(comparison.round(4))
```

For this problem, I would primarily select by:

1. Macro F1
2. Critical-class recall
3. High-class recall
4. Balanced accuracy
5. Ordinal MAE

Macro F1 gives each stress class equal importance, while ordinal MAE penalises a `critical → low` error more heavily than a `critical → high` error.

#### 5. Examine individual classes

```python
print("Random Forest")
print(classification_report(
    y_test,
    rf_prediction,
    target_names=class_names,
    digits=3,
))

print("XGBoost")
print(classification_report(
    y_test,
    xgb_prediction,
    target_names=class_names,
    digits=3,
))
```

Confusion matrices:

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for axis, name, prediction in [
    (axes[0], "Random Forest", rf_prediction),
    (axes[1], "XGBoost", xgb_prediction),
]:
    matrix = confusion_matrix(y_test, prediction)

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )

    axis.set_title(name)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")

plt.tight_layout()
plt.show()
```

The most important cells are actual `high` and `critical` observations predicted as `low`. A model with slightly lower overall accuracy may still be preferable if it detects substantially more serious stress events.

### Exchange 8

##### Prompt

> How to do this now: •	Symbolic AI (Prolog):
> o	Interpret the predicted risk classification.
> o	Apply operational rules based on reserve margins and supply conditions.
> o	Recommend appropriate responses for different risk levels.
>  READ ONLY

##### Output

Use Prolog as a rule-based layer after the Random Forest/XGBoost prediction:

```text
Electricity features → ML risk prediction → Prolog facts → rules → interpretation/actions
```

No files were changed.

#### Important limitation

A true reserve margin is:

```text
(available generation capacity − expected demand)
------------------------------------------------- × 100
                expected demand
```

The parquet contains generation output (`power`), not available capacity. Therefore, do not calculate reserve margin using `power`. Obtain available capacity/reserve data externally, or explicitly call the result a `supply_margin_proxy`.

#### Prolog inputs

At each prediction timestamp, pass Prolog facts such as:

```prolog
predicted_risk(period_1, high).
reserve_margin(period_1, 8.2).
import_share(period_1, 0.21).
price(period_1, 375.0).
curtailment(period_1, 40.0).
stress_driver(period_1, adequacy).
```

These facts can be created in memory by Python; they do not need to become new parquet columns.

#### Example Prolog knowledge base

```prolog
% Risk ordering
risk_rank(low, 0).
risk_rank(medium, 1).
risk_rank(high, 2).
risk_rank(critical, 3).

% Reserve-margin rules.
% These thresholds are illustrative and require domain validation.
reserve_risk(T, critical) :-
    reserve_margin(T, Margin),
    Margin < 5, !.

reserve_risk(T, high) :-
    reserve_margin(T, Margin),
    Margin < 10, !.

reserve_risk(T, medium) :-
    reserve_margin(T, Margin),
    Margin < 15, !.

reserve_risk(_, low).

% Supply-condition rules using thresholds derived from training data.
supply_risk(T, high) :-
    import_share(T, ImportShare),
    price(T, Price),
    ImportShare >= 0.20,
    Price >= 353, !.

supply_risk(T, high) :-
    curtailment(T, Curtailment),
    Curtailment >= 491, !.

supply_risk(T, medium) :-
    import_share(T, ImportShare),
    ImportShare >= 0.15, !.

supply_risk(T, medium) :-
    curtailment(T, Curtailment),
    Curtailment >= 217, !.

supply_risk(_, low).

% Select the highest risk produced by the ML model, reserve rule,
% or supply-condition rule.
maximum_risk(A, B, A) :-
    risk_rank(A, RankA),
    risk_rank(B, RankB),
    RankA >= RankB, !.

maximum_risk(_, B, B).

operational_risk(T, FinalRisk) :-
    predicted_risk(T, PredictedRisk),
    reserve_risk(T, ReserveRisk),
    supply_risk(T, SupplyRisk),
    maximum_risk(PredictedRisk, ReserveRisk, IntermediateRisk),
    maximum_risk(IntermediateRisk, SupplyRisk, FinalRisk).
```

This allows Prolog to override an optimistic model prediction when an explicit operational rule identifies a more serious condition.

#### Interpretations

```prolog
interpretation(T, normal_operating_conditions) :-
    operational_risk(T, low).

interpretation(T, elevated_conditions_require_attention) :-
    operational_risk(T, medium).

interpretation(T, serious_conditions_require_intervention) :-
    operational_risk(T, high).

interpretation(T, emergency_conditions_require_immediate_action) :-
    operational_risk(T, critical).
```

#### Recommended responses

```prolog
recommendation(T, continue_normal_monitoring) :-
    operational_risk(T, low).

recommendation(T, review_demand_and_generation_forecasts) :-
    operational_risk(T, medium).

recommendation(T, verify_reserve_and_interconnector_availability) :-
    operational_risk(T, medium).

recommendation(T, prepare_fast_start_generation) :-
    operational_risk(T, high),
    stress_driver(T, adequacy).

recommendation(T, request_additional_imports) :-
    operational_risk(T, high),
    stress_driver(T, adequacy).

recommendation(T, initiate_demand_response) :-
    operational_risk(T, high),
    stress_driver(T, adequacy).

recommendation(T, investigate_binding_network_constraints) :-
    operational_risk(T, high),
    stress_driver(T, congestion).

recommendation(T, consider_redispatch_or_storage_charging) :-
    operational_risk(T, high),
    stress_driver(T, congestion).

recommendation(T, activate_emergency_reserves) :-
    operational_risk(T, critical).

recommendation(T, notify_system_operator) :-
    operational_risk(T, critical).

recommendation(T, prepare_controlled_load_shedding_as_last_resort) :-
    operational_risk(T, critical),
    stress_driver(T, adequacy).
```

Any emergency recommendation should be advisory and require authorised operator approval.

#### Connect Python to Prolog

Using `pyswip`:

```python
from pyswip import Prolog

prolog = Prolog()
prolog.consult("network_stress_rules.pl")

timestamp_id = "period_1"
predicted_class = "high"

# These values would come from the prediction record and external
# reserve/capacity information.
reserve_margin = 8.2
import_share = 0.21
price = 375.0
curtailment = 40.0
stress_driver = "adequacy"

prolog.assertz(f"predicted_risk({timestamp_id}, {predicted_class})")
prolog.assertz(f"reserve_margin({timestamp_id}, {reserve_margin})")
prolog.assertz(f"import_share({timestamp_id}, {import_share})")
prolog.assertz(f"price({timestamp_id}, {price})")
prolog.assertz(f"curtailment({timestamp_id}, {curtailment})")
prolog.assertz(f"stress_driver({timestamp_id}, {stress_driver})")

final_risk = list(
    prolog.query(f"operational_risk({timestamp_id}, Risk)")
)

interpretation = list(
    prolog.query(f"interpretation({timestamp_id}, Interpretation)")
)

recommendations = list(
    prolog.query(f"recommendation({timestamp_id}, Action)")
)

print(final_risk)
print(interpretation)
print(recommendations)
```

For example, the ML model might predict `medium`, but a reserve margin below 5% would cause the symbolic layer to classify the operational risk as `critical` and recommend emergency-reserve activation. This gives you a hybrid system: statistical prediction from ML and transparent operational reasoning from Prolog.

### Exchange 10

##### Prompt

> How to save final model and load model for inference? READ ONLY

##### Output

No files were changed.

The safest approach is to save the fitted model together with its feature order, class mapping, prediction horizon, and training metadata. Saving only the estimator can cause inference errors when columns arrive in a different order.

#### Random Forest: save with Joblib

After selecting and fitting the final model:

```python
import joblib
from pathlib import Path

MODEL_DIRECTORY = Path("models")
MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

class_names = {
    0: "low",
    1: "medium",
    2: "high",
    3: "critical",
}

model_bundle = {
    "model": random_forest,
    "model_type": "RandomForestClassifier",
    "feature_columns": feature_columns,
    "class_names": class_names,
    "prediction_horizon_hours": 1,
    "training_end": str(model_df["timestamp"].iloc[split_index - 1]),
}

joblib.dump(
    model_bundle,
    MODEL_DIRECTORY / "network_stress_random_forest.joblib",
)
```

If preprocessing uses a scikit-learn `Pipeline`, save the entire fitted pipeline as `"model"` rather than saving the classifier separately.

#### Load Random Forest for inference

```python
import joblib
import pandas as pd

bundle = joblib.load(
    "models/network_stress_random_forest.joblib"
)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
class_names = bundle["class_names"]
```

Never load a Joblib or Pickle model from an untrusted source, because loading can execute arbitrary code.

#### Perform inference

The input must contain exactly the features used during training:

```python
def predict_stress(input_data, bundle):
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    class_names = bundle["class_names"]

    missing_columns = [
        column
        for column in feature_columns
        if column not in input_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing inference features: {missing_columns}"
        )

    # Enforce the original training column order.
    model_input = input_data.reindex(columns=feature_columns)

    predicted_number = int(model.predict(model_input)[0])
    probabilities = model.predict_proba(model_input)[0]

    probability_by_class = {
        class_names[int(class_number)]: float(probability)
        for class_number, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    return {
        "predicted_class": class_names[predicted_number],
        "confidence": max(probability_by_class.values()),
        "probabilities": probability_by_class,
        "prediction_horizon_hours":
            bundle["prediction_horizon_hours"],
    }
```

For a row already containing transformed features:

```python
latest_observation = model_df.iloc[[-1]]

result = predict_stress(
    input_data=latest_observation,
    bundle=bundle,
)

print(result)
```

Example output:

```python
{
    "predicted_class": "high",
    "confidence": 0.73,
    "probabilities": {
        "low": 0.04,
        "medium": 0.18,
        "high": 0.73,
        "critical": 0.05,
    },
    "prediction_horizon_hours": 1,
}
```

#### XGBoost: native save format

For XGBoost, its native JSON format is more portable than Pickle:

```python
import json
from pathlib import Path

MODEL_DIRECTORY = Path("models")
MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

xgboost_model.save_model(
    MODEL_DIRECTORY / "network_stress_xgboost.json"
)

metadata = {
    "model_type": "XGBClassifier",
    "feature_columns": feature_columns,
    "class_names": {
        "0": "low",
        "1": "medium",
        "2": "high",
        "3": "critical",
    },
    "prediction_horizon_hours": 1,
    "training_end": str(
        model_df["timestamp"].iloc[split_index - 1]
    ),
}

with open(
    MODEL_DIRECTORY / "network_stress_xgboost_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, indent=2)
```

Load it using:

```python
import json
from xgboost import XGBClassifier

xgboost_model = XGBClassifier()
xgboost_model.load_model(
    "models/network_stress_xgboost.json"
)

with open(
    "models/network_stress_xgboost_metadata.json",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

xgb_bundle = {
    "model": xgboost_model,
    "feature_columns": metadata["feature_columns"],
    "class_names": {
        int(number): name
        for number, name in metadata["class_names"].items()
    },
    "prediction_horizon_hours":
        metadata["prediction_horizon_hours"],
}

result = predict_stress(latest_observation, xgb_bundle)
```

#### Important feature-engineering requirement

Your model uses lag and rolling features such as:

```text
demand_lag_168h
price_rolling_mean_24h
curtailment_lag_24h
```

Therefore, a live inference pipeline cannot accept just one raw observation. It needs sufficient historical data—at least 168 hours—to reconstruct the same features used during training.

The correct inference sequence is:

```text
Load recent 168+ hours
        ↓
Apply the same cleaning and feature-engineering functions
        ↓
Select the latest transformed row
        ↓
Reindex to saved feature_columns
        ↓
Load model and predict
        ↓
Pass prediction to Prolog
```

The feature-engineering code should be shared between training and inference rather than copied into separate notebooks.

#### Passing the result to Prolog

```python
prediction = predict_stress(latest_observation, bundle)

predicted_class = prediction["predicted_class"]
confidence = prediction["confidence"]

prolog.assertz(
    f"predicted_risk(period_1, {predicted_class})"
)

prolog.assertz(
    f"prediction_confidence(period_1, {confidence})"
)
```

Then query the symbolic system:

```python
final_risk = list(
    prolog.query("operational_risk(period_1, Risk)")
)

recommendations = list(
    prolog.query("recommendation(period_1, Action)")
)
```

For the assignment, retain the untouched test set until after the final model evaluation. Save the model that produced the reported test results; if you later retrain on more data for deployment, identify that as a separate deployment model.

### Exchange 11

##### Prompt

> Im confused on how to connect the machine learning model to a inference layer. Please give me a basic example of how this will kinda be done in this project. READ ONLY

##### Output

The word “inference” is being used in two related ways:

1. **ML inference:** load the trained model and call `predict()`.
2. **Symbolic inference:** give that prediction to Prolog, which applies rules and produces a recommendation.

Prolog does not train or call Random Forest directly. Python acts as the bridge.

```text
Electricity observation
        ↓
Python loads Random Forest
        ↓
ML predicts "medium"
        ↓
Python asserts predicted_risk(case_1, medium) into Prolog
        ↓
Prolog combines it with operational facts
        ↓
Final risk: high
Recommendation: investigate network constraints
```

No files were changed.

#### 1. Training happens separately

Your training code selects the final model and saves it:

```python
import joblib

model_bundle = {
    "model": random_forest,
    "feature_columns": feature_columns,
    "class_names": {
        0: "low",
        1: "medium",
        2: "high",
        3: "critical",
    },
}

joblib.dump(
    model_bundle,
    "network_stress_model.joblib",
)
```

This only needs to happen when the model is trained or retrained.

#### 2. A basic Prolog knowledge base

Conceptually, `stress_rules.pl` could contain:

```prolog
% Python will insert these facts at runtime.
:- dynamic predicted_risk/2.
:- dynamic prediction_confidence/2.
:- dynamic import_share/2.
:- dynamic electricity_price/2.
:- dynamic curtailment/2.

% A critical prediction always remains critical.
final_risk(Case, critical) :-
    predicted_risk(Case, critical), !.

% Escalate a high prediction when supply conditions are also severe.
final_risk(Case, critical) :-
    predicted_risk(Case, high),
    import_share(Case, ImportShare),
    electricity_price(Case, Price),
    ImportShare >= 0.20,
    Price >= 353, !.

% Severe curtailment can escalate medium risk to high.
final_risk(Case, high) :-
    predicted_risk(Case, medium),
    curtailment(Case, Curtailment),
    Curtailment >= 491, !.

% Otherwise retain the ML prediction.
final_risk(Case, Risk) :-
    predicted_risk(Case, Risk).

recommendation(Case, continue_normal_monitoring) :-
    final_risk(Case, low).

recommendation(Case, review_forecasts_and_available_supply) :-
    final_risk(Case, medium).

recommendation(Case, investigate_constraints_and_prepare_response) :-
    final_risk(Case, high).

recommendation(Case, notify_operator_and_activate_emergency_procedures) :-
    final_risk(Case, critical).
```

The thresholds above are illustrative and should be justified using your training data and domain sources.

#### 3. Python inference and integration

Python loads the model, makes a prediction, and passes facts to Prolog:

```python
import joblib
from pyswip import Prolog

# Load the trained ML model.
bundle = joblib.load("network_stress_model.joblib")

model = bundle["model"]
feature_columns = bundle["feature_columns"]
class_names = bundle["class_names"]

# This row represents conditions at time t.
# Its prediction represents stress at t + 1 hour.
latest_row = transformed_data.iloc[[-1]]

# Enforce the same feature order used during training.
model_input = latest_row.reindex(columns=feature_columns)

predicted_number = int(model.predict(model_input)[0])
probabilities = model.predict_proba(model_input)[0]

predicted_class = class_names[predicted_number]
confidence = float(probabilities.max())

print("ML prediction:", predicted_class)
print("Confidence:", confidence)
```

Now pass that prediction and the operating conditions to Prolog:

```python
prolog = Prolog()
prolog.consult("stress_rules.pl")

case_id = "case_1"

# Obtain operational facts from the latest observation.
import_share = float(
    latest_row["flow_imports"].iloc[0]
    / latest_row["demand_gross"].iloc[0]
)

price = float(latest_row["price"].iloc[0])
curtailment = max(float(latest_row["curtailment"].iloc[0]), 0.0)

# Insert the ML output and operational data as Prolog facts.
prolog.assertz(
    f"predicted_risk({case_id}, {predicted_class})"
)

prolog.assertz(
    f"prediction_confidence({case_id}, {confidence})"
)

prolog.assertz(
    f"import_share({case_id}, {import_share})"
)

prolog.assertz(
    f"electricity_price({case_id}, {price})"
)

prolog.assertz(
    f"curtailment({case_id}, {curtailment})"
)
```

Finally, query Prolog:

```python
risk_result = list(
    prolog.query(f"final_risk({case_id}, Risk)")
)

recommendation_result = list(
    prolog.query(f"recommendation({case_id}, Action)")
)

final_risk = risk_result[0]["Risk"]
recommendations = [
    result["Action"]
    for result in recommendation_result
]

print("Final hybrid risk:", final_risk)
print("Recommendations:", recommendations)
```

#### Example system behaviour

Suppose Random Forest produces:

```text
ML prediction: medium
Confidence: 0.64
Import share: 0.17
Price: $145/MWh
Curtailment: 530 MW
```

Python converts these into:

```prolog
predicted_risk(case_1, medium).
prediction_confidence(case_1, 0.64).
import_share(case_1, 0.17).
electricity_price(case_1, 145.0).
curtailment(case_1, 530.0).
```

Prolog matches this rule:

```prolog
final_risk(Case, high) :-
    predicted_risk(Case, medium),
    curtailment(Case, Curtailment),
    Curtailment >= 491.
```

The final output becomes:

```text
ML prediction: medium
Final hybrid risk: high
Reason: curtailment exceeded 491 MW
Recommendation: investigate constraints and prepare response
```

That is the hybrid element: the ML model recognises patterns, while Prolog transparently applies operational knowledge.

#### Suggested project structure

```text
Source/
├── train_model.py
├── inference.py
├── feature_engineering.py
├── stress_rules.pl
├── models/
│   └── network_stress_model.joblib
├── README.md
└── requirements.txt
```

For the video demonstration, run `inference.py` and show:

1. The selected input observation.
2. The ML prediction and probabilities.
3. The facts passed to Prolog.
4. The rule that fired.
5. The final risk and recommendation.

The most important implementation principle is that the Prolog facts must be created automatically from the actual model output—not manually entered for the demonstration.

### Exchange 12

##### Prompt

> How will the ML make a prediction, from a text input in the terminal?

##### Output

The ML model will not understand ordinary free-form text because Random Forest and XGBoost are tabular models. Terminal input can select or provide a structured observation, which Python converts into the numeric feature row expected by the model.

The easiest demonstration for your project is:

```text
User enters a timestamp
        ↓
Python finds that observation in the dataset
        ↓
Python selects the saved model features
        ↓
Model predicts next-hour stress
        ↓
Prediction is passed to Prolog
```

#### Simple terminal example

The user runs:

```powershell
python inference.py
```

The program asks:

```text
Enter an observation timestamp: 2026-09-23 22:00:00+10:00
```

Python then selects that row and predicts stress at `23:00`.

Conceptually, `inference.py` would contain:

```python
import joblib
import pandas as pd

DATA_PATH = "electricity_transformed.parquet"
MODEL_PATH = "models/network_stress_model.joblib"

# Load data and trained model.
data = pd.read_parquet(DATA_PATH)
data = data.sort_values("timestamp").reset_index(drop=True)

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
class_names = bundle["class_names"]

# Obtain structured terminal input.
timestamp_text = input(
    "Enter an observation timestamp: "
).strip()

requested_timestamp = pd.Timestamp(timestamp_text)

matching_rows = data.loc[
    data["timestamp"] == requested_timestamp
]

if matching_rows.empty:
    raise ValueError(
        f"No observation found for {requested_timestamp}"
    )

# Keep the row as a DataFrame, not a Series.
observation = matching_rows.iloc[[-1]]

# Select and order the model features.
model_input = observation.reindex(
    columns=feature_columns
)

# Predict next-hour stress.
predicted_number = int(
    model.predict(model_input)[0]
)

probabilities = model.predict_proba(
    model_input
)[0]

predicted_class = class_names[predicted_number]
confidence = float(probabilities.max())

print()
print("Prediction horizon: 1 hour")
print("Predicted stress:", predicted_class)
print(f"Confidence: {confidence:.2%}")
```

Example output:

```text
Enter an observation timestamp: 2026-09-23 22:00:00+10:00

Prediction horizon: 1 hour
Predicted stress: high
Confidence: 76.34%
```

#### Passing that prediction to Prolog

Continue the same script with:

```python
from pyswip import Prolog

prolog = Prolog()
prolog.consult("stress_rules.pl")

case_id = "current_case"

import_share = float(
    observation["flow_imports"].iloc[0]
    / observation["demand_gross"].iloc[0]
)

price = float(
    observation["price"].iloc[0]
)

curtailment = max(
    float(observation["curtailment"].iloc[0]),
    0.0,
)

prolog.assertz(
    f"predicted_risk({case_id}, {predicted_class})"
)

prolog.assertz(
    f"prediction_confidence({case_id}, {confidence})"
)

prolog.assertz(
    f"import_share({case_id}, {import_share})"
)

prolog.assertz(
    f"electricity_price({case_id}, {price})"
)

prolog.assertz(
    f"curtailment({case_id}, {curtailment})"
)

final_risk_result = list(
    prolog.query(
        f"final_risk({case_id}, Risk)"
    )
)

recommendation_results = list(
    prolog.query(
        f"recommendation({case_id}, Action)"
    )
)

final_risk = final_risk_result[0]["Risk"]

recommendations = [
    result["Action"]
    for result in recommendation_results
]

print("Final hybrid risk:", final_risk)
print("Recommendations:")

for recommendation in recommendations:
    print("-", recommendation)
```

The complete output might be:

```text
Selected observation: 2026-09-23 22:00
ML prediction for 23:00: medium
ML confidence: 64.20%

Operational conditions:
- Import share: 17.3%
- Price: $145.20/MWh
- Curtailment: 530.4 MW

Prolog result:
- Final hybrid risk: high
- Triggered condition: severe curtailment
- Recommendation: investigate constraints and prepare response
```

#### Command-line argument alternative

A more professional interface uses command-line arguments:

```powershell
python inference.py `
  --data "electricity_transformed.parquet" `
  --timestamp "2026-09-23 22:00:00+10:00"
```

Python obtains these using:

```python
import argparse

parser = argparse.ArgumentParser(
    description="Predict next-hour electricity network stress."
)

parser.add_argument(
    "--data",
    required=True,
    help="Path to the transformed electricity parquet file.",
)

parser.add_argument(
    "--timestamp",
    required=True,
    help="Timestamp used as the prediction origin.",
)

arguments = parser.parse_args()
```

#### What about entering conditions manually?

You could support:

```text
Demand: 10,500 MW
Price: $300/MWh
Imports: 1,500 MW
Curtailment: 400 MW
```

However, your model also uses lag and rolling features. A few manually entered values are insufficient to reconstruct fields such as:

```text
demand_lag_168h
price_rolling_mean_24h
generation_renewable_lag_24h
curtailment_rolling_std_24h
```

For this reason, selecting a timestamp from a dataset containing the engineered features is the simplest reliable demonstration.

A real deployment would instead load at least the previous 168 hours of measurements, recreate the lag and rolling features, and then predict the following hour.

Free-form prompts such as:

```text
Will the electricity network be stressed tomorrow evening?
```

would require an additional natural-language processing layer. That is not required for this hybrid ML–Prolog assignment and would unnecessarily expand the project.

### Exchange 13

##### Prompt

> how to fix this:DATA_PATH = "electricity_transformed.parquet"
> MODEL_PATH = "models/network_stress_xgboost.json"
> 
> # Load data and trained model.
> data = pd.read_parquet(DATA_PATH)
> data = data.sort_values("timestamp").reset_index(drop=True)
> 
> with open(
>     MODEL_PATH,
>     encoding="utf-8",
> ) as file:
>     metadata = json.load(file)
> 
> bundle = {
>     "model": xgboost_model,
>     "feature_columns": metadata["feature_columns"],
>     "class_names": {
>         int(number): name
>         for number, name in metadata["class_names"].items()
>     },
>     "prediction_horizon_hours":
>         metadata["prediction_horizon_hours"],
> }
> 
> model = bundle["model"]
> feature_columns = bundle["feature_columns"]
> class_names = bundle["class_names"]

##### Output

The problem is that `network_stress_xgboost.json` is the XGBoost model file, but your code tries to read it as your metadata file. You also reference `xgboost_model` before loading it.

Use two separate files:

```text
network_stress_xgboost.json
network_stress_xgboost_metadata.json
```

#### Save the model correctly

After training:

```python
import json
from pathlib import Path

MODEL_DIRECTORY = Path("models")
MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIRECTORY / "network_stress_xgboost.json"
)

METADATA_PATH = (
    MODEL_DIRECTORY
    / "network_stress_xgboost_metadata.json"
)

# Save the actual XGBoost model.
xgboost_model.save_model(MODEL_PATH)

# Save your application-specific information separately.
metadata = {
    "feature_columns": list(feature_columns),
    "class_names": {
        "0": "low",
        "1": "medium",
        "2": "high",
        "3": "critical",
    },
    "prediction_horizon_hours": 1,
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, indent=2)
```

#### Load the model correctly

```python
import json
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

# Find the repository root regardless of whether this runs from the
# repository root or A2/notebooks.
for base in (Path.cwd(), *Path.cwd().parents):
    if (
        base / "A1" / "data" / "processed"
        / "electricity_transformed.parquet"
    ).is_file():
        PROJECT_ROOT = base
        break
else:
    raise FileNotFoundError(
        "Could not locate the project root."
    )

DATA_PATH = (
    PROJECT_ROOT
    / "A1"
    / "data"
    / "processed"
    / "electricity_transformed.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "A2"
    / "models"
    / "network_stress_xgboost.json"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "A2"
    / "models"
    / "network_stress_xgboost_metadata.json"
)

# Check that all required files exist.
for required_path in [
    DATA_PATH,
    MODEL_PATH,
    METADATA_PATH,
]:
    if not required_path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {required_path}"
        )

# Load the dataset.
data = pd.read_parquet(DATA_PATH)
data = data.sort_values(
    "timestamp"
).reset_index(drop=True)

# Load the trained XGBoost model.
xgboost_model = XGBClassifier()
xgboost_model.load_model(MODEL_PATH)

# Load the separate metadata file.
with open(
    METADATA_PATH,
    encoding="utf-8",
) as file:
    metadata = json.load(file)

bundle = {
    "model": xgboost_model,
    "feature_columns": metadata["feature_columns"],
    "class_names": {
        int(number): name
        for number, name
        in metadata["class_names"].items()
    },
    "prediction_horizon_hours":
        metadata["prediction_horizon_hours"],
}

model = bundle["model"]
feature_columns = bundle["feature_columns"]
class_names = bundle["class_names"]

print("Model loaded successfully.")
print("Number of features:", len(feature_columns))
print("Classes:", class_names)
```

#### Confirm feature compatibility

After loading, check that the dataset contains every required feature:

```python
missing_features = [
    column
    for column in feature_columns
    if column not in data.columns
]

if missing_features:
    raise ValueError(
        f"Dataset is missing features: {missing_features}"
    )
```

Then make a prediction:

```python
observation = data.iloc[[-1]]
model_input = observation.reindex(
    columns=feature_columns
)

prediction_number = int(
    model.predict(model_input)[0]
)

probabilities = model.predict_proba(
    model_input
)[0]

prediction_name = class_names[
    prediction_number
]

print("Predicted stress:", prediction_name)
print("Confidence:", probabilities.max())
```

If you receive `ModuleNotFoundError: No module named 'xgboost'`, XGBoost was not installed in your `.venv` when I last checked. It would need to be installed before loading the model:

```powershell
.\.venv\Scripts\python.exe -m pip install xgboost
```

The essential fix is: load the model with `XGBClassifier.load_model()` and load your feature/class metadata from a different JSON file.

### Exchange 14

##### Prompt

> how to obtain operaitnal facts from this:
> 
> # Obtain structured terminal input.
> timestamp_text = input(
>     "Enter an observation timestamp: "
> ).strip()
> 
> requested_timestamp = pd.Timestamp(timestamp_text)
> 
> matching_rows = data.loc[
>     data["timestamp"] == requested_timestamp
> ]
> 
> if matching_rows.empty:
>     raise ValueError(
>         f"No observation found for {requested_timestamp}"
>     )
> 
> # Keep the row as a DataFrame, not a Series.
> observation = matching_rows.iloc[[-1]]
> 
> # Select and order the model features.
> model_input = observation.reindex(
>     columns=feature_columns
> )
> 
> # Predict next-hour stress.
> predicted_number = int(
>     model.predict(model_input)[0]
> )
> 
> probabilities = model.predict_proba(
>     model_input
> )[0]
> 
> predicted_class = class_names[predicted_number]
> confidence = float(probabilities.max())
> 
> print()
> print("Prediction horizon: 1 hour")
> print("Predicted stress:", predicted_class)
> print(f"Confidence: {confidence:.2%}")

##### Output

The operational facts come from `observation`, which still contains the complete selected dataset row. `model_input` contains only the columns required by XGBoost, so extract operational facts from `observation`, not `model_input`.

Add this after producing the prediction:

```python
import numpy as np

# Convert the one-row DataFrame into a Series for convenient access.
row = observation.iloc[0]

def get_numeric_fact(column_name):
    """Return a finite numeric value from the selected observation."""
    if column_name not in row.index:
        raise KeyError(
            f"Operational column not found: {column_name}"
        )

    value = float(row[column_name])

    if not np.isfinite(value):
        raise ValueError(
            f"{column_name} contains an invalid value: {value}"
        )

    return value

demand = get_numeric_fact("demand")
gross_demand = get_numeric_fact("demand_gross")
imports = get_numeric_fact("flow_imports")
exports = get_numeric_fact("flow_exports")
price = get_numeric_fact("price")
curtailment = get_numeric_fact("curtailment")
renewable_percentage = get_numeric_fact(
    "renewable_with_storage_proportion"
)

# Derived operational condition; no new dataframe column is needed.
import_share = (
    imports / gross_demand
    if gross_demand > 0
    else 0.0
)

# Negative curtailment is retained in the original data but should not
# represent congestion stress.
curtailment_for_rules = max(curtailment, 0.0)

operational_facts = {
    "predicted_risk": predicted_class,
    "prediction_confidence": confidence,
    "demand": demand,
    "gross_demand": gross_demand,
    "imports": imports,
    "exports": exports,
    "import_share": import_share,
    "price": price,
    "curtailment": curtailment_for_rules,
    "renewable_percentage": renewable_percentage,
}

print("\nOperational facts")

for fact_name, fact_value in operational_facts.items():
    print(f"- {fact_name}: {fact_value}")
```

Example output:

```text
Prediction horizon: 1 hour
Predicted stress: medium
Confidence: 64.21%

Operational facts
- predicted_risk: medium
- prediction_confidence: 0.6421
- demand: 9450.2
- gross_demand: 10120.7
- imports: 1850.4
- exports: 0.0
- import_share: 0.1828
- price: 245.6
- curtailment: 525.3
- renewable_percentage: 58.4
```

#### Pass the facts to Prolog

Your Prolog file needs to declare the facts that Python will insert:

```prolog
:- dynamic predicted_risk/2.
:- dynamic prediction_confidence/2.
:- dynamic demand/2.
:- dynamic gross_demand/2.
:- dynamic imports/2.
:- dynamic exports/2.
:- dynamic import_share/2.
:- dynamic electricity_price/2.
:- dynamic curtailment/2.
:- dynamic renewable_percentage/2.
```

Then Python can assert them:

```python
from pyswip import Prolog

prolog = Prolog()
prolog.consult("stress_rules.pl")

case_id = "current_case"

prolog.assertz(
    f"predicted_risk({case_id}, {predicted_class})"
)

prolog.assertz(
    f"prediction_confidence({case_id}, {confidence})"
)

prolog.assertz(
    f"demand({case_id}, {demand})"
)

prolog.assertz(
    f"gross_demand({case_id}, {gross_demand})"
)

prolog.assertz(
    f"imports({case_id}, {imports})"
)

prolog.assertz(
    f"exports({case_id}, {exports})"
)

prolog.assertz(
    f"import_share({case_id}, {import_share})"
)

prolog.assertz(
    f"electricity_price({case_id}, {price})"
)

prolog.assertz(
    f"curtailment({case_id}, {curtailment_for_rules})"
)

prolog.assertz(
    f"renewable_percentage("
    f"{case_id}, {renewable_percentage})"
)
```

These produce runtime Prolog facts equivalent to:

```prolog
predicted_risk(current_case, medium).
prediction_confidence(current_case, 0.6421).
demand(current_case, 9450.2).
gross_demand(current_case, 10120.7).
imports(current_case, 1850.4).
exports(current_case, 0.0).
import_share(current_case, 0.1828).
electricity_price(current_case, 245.6).
curtailment(current_case, 525.3).
renewable_percentage(current_case, 58.4).
```

#### Let Prolog interpret the numbers

Python should provide relatively raw facts. Prolog should determine what those facts mean:

```prolog
high_import_dependency(Case) :-
    import_share(Case, Share),
    Share >= 0.20.

severe_curtailment(Case) :-
    curtailment(Case, Curtailment),
    Curtailment >= 491.

high_price(Case) :-
    electricity_price(Case, Price),
    Price >= 353.

final_risk(Case, critical) :-
    predicted_risk(Case, high),
    high_import_dependency(Case),
    high_price(Case), !.

final_risk(Case, high) :-
    predicted_risk(Case, medium),
    severe_curtailment(Case), !.

final_risk(Case, Risk) :-
    predicted_risk(Case, Risk).
```

Query the result:

```python
risk_results = list(
    prolog.query(
        f"final_risk({case_id}, Risk)"
    )
)

if not risk_results:
    raise RuntimeError(
        "Prolog did not produce a final risk."
    )

final_risk = risk_results[0]["Risk"]

print("ML prediction:", predicted_class)
print("Final hybrid risk:", final_risk)
```

#### Timestamp meaning

The selected row contains conditions observed at time `t`, while the ML prediction represents stress at `t + 1 hour`.

Therefore, describe the Prolog facts accurately:

```text
Current operational conditions at time t
+ ML prediction for time t+1
→ final next-hour recommendation
```

They are not known future conditions unless you also have forecasts for the next hour.

Finally, a true reserve margin cannot be extracted from this observation. The dataset does not contain available generation capacity. Reserve margin must come from an external source or be entered separately; it should not be calculated from the `power` column.

### Exchange 15

##### Prompt

> could you investigate my A2/notebooks folder and see what is wrong my current "pipeline". i understand the pipeline is in notebook for now just during dev tho

##### Output

#### Progress update 1

I’ll review the A2 notebooks as a development pipeline: execution order, data/target construction, train/test leakage, model saving/loading, inference inputs, and the Python-to-Prolog handoff. I’ll keep this diagnostic-only and won’t modify the notebooks.

#### Progress update 2

I’ve found several concrete integration breaks—mainly mismatched Prolog predicate names and missing runtime facts. I’m doing two read-only smoke tests now: one for the saved XGBoost artifact/feature schema and one for the Prolog query path, so I can separate confirmed failures from design concerns.

#### Final response 3

I inspected the notebooks and ran read-only smoke tests. The ML portion works, but the Python-to-Prolog stage currently cannot complete.

#### What is working

- The saved XGBoost model loads successfully.
- Its 43 model features exactly match the metadata.
- All required features exist in the transformed parquet.
- `predict()` and `predict_proba()` work.
- The chronological 70/30 split is appropriate.
- The one-hour target shift is conceptually correct.
- XGBoost currently performs better overall than Random Forest:
  - XGBoost macro-F1: 0.747
  - Random Forest macro-F1: 0.729
- A1’s lag and rolling features correctly exclude the current hour.

#### Blocking Prolog errors

##### 1. Python queries a nonexistent predicate

[inference.ipynb](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/notebooks/inference.ipynb) queries:

```prolog
final_risk(current_case, Risk)
```

But [stress_rules.pl](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/notebooks/stress_rules.pl) defines:

```prolog
operational_risk(T, FinalRisk)
```

The confirmed runtime error is:

```text
existence_error(procedure, final_risk/2)
```

You need to consistently use one name. For example, query:

```python
prolog.query(
    f"operational_risk({case_id}, Risk)"
)
```

or rename the Prolog predicate to `final_risk/2`.

##### 2. `reserve_margin/2` is missing

Every call to `operational_risk/2` calls:

```prolog
reserve_risk(T, ReserveRisk)
```

That calls:

```prolog
reserve_margin(T, Margin)
```

However:

- `reserve_margin/2` is not declared dynamic.
- Python never asserts a reserve-margin fact.
- The parquet cannot provide a true reserve margin.

The confirmed error is:

```text
existence_error(procedure, reserve_margin/2)
```

For the current dataset, you should either:

- Temporarily remove reserve margin from the executable rules.
- Obtain it from an external source.
- Make it optional and default to `low` when unavailable.

Simply declaring it dynamic prevents the undefined-predicate error, but you must still decide what absence means:

```prolog
:- dynamic reserve_margin/2.

reserve_risk(T, low) :-
    \+ reserve_margin(T, _), !.
```

Do not derive reserve margin from `power`; it is output rather than available capacity.

##### 3. Price predicate names do not match

Python asserts:

```prolog
electricity_price(current_case, 250.0).
```

But Prolog queries:

```prolog
price(T, Price)
```

Therefore, this rule can never use the fact supplied by Python:

```prolog
supply_risk(T, high) :-
    import_share(T, ImportShare),
    price(T, Price),
    ImportShare >= 0.20,
    Price >= 353.
```

Use `electricity_price/2` consistently in both places.

##### 4. `stress_driver/2` is never supplied

Several recommendations require:

```prolog
stress_driver(T, adequacy).
```

or:

```prolog
stress_driver(T, congestion).
```

But Python never asserts `stress_driver/2`, and it is not declared dynamic. Consequently, most high-risk recommendations cannot succeed.

You must either:

- Calculate and pass a driver.
- Derive the driver inside Prolog from the operational facts.
- Remove the driver requirement and attach recommendations directly to supply/congestion conditions.

Deriving it inside Prolog is probably the cleaner symbolic-AI design.

#### The pipeline has not actually run end-to-end

The saved execution state shows:

- Model loading ran.
- Timestamp prediction ran.
- Operational-fact extraction did not run.
- Prolog assertion did not run.
- Final Prolog query did not run.

So the displayed `critical, 81.66%` result proves only the ML stage. It does not prove the hybrid pipeline.

After correcting the predicates, restart the kernel and use **Run All**. A successful notebook should show:

```text
timestamp selected
→ ML prediction
→ operational facts
→ Prolog final risk
→ interpretation
→ recommendation
```

#### Training-methodology issues

##### Test data is being used to select the model

[model_training.ipynb](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/notebooks/model_training.ipynb) compares Random Forest and XGBoost on the 30% test set and then saves XGBoost.

If XGBoost is selected because it performs better on that test set, the test set has effectively become a validation set. You no longer have an untouched final test estimate.

A better chronological split is:

```text
First 60%: training
Next 20%: validation/model selection
Final 20%: untouched test
```

Alternatively, use time-series cross-validation within the first 70%, then evaluate once on the final 30%.

##### The final save cell has not been executed

The model-saving cell has `execution_count=None`, even though model files exist in `A2/models`. This means the notebook does not demonstrate that those artifacts were created by the current clean execution.

The artifacts themselves are valid, but they could be stale relative to later notebook edits. Run training from a clean kernel and regenerate them once the methodology is final.

##### Prolog thresholds are manually copied

The notebook calculates training thresholds, but the metadata does not save them. Prolog then hardcodes values such as:

```prolog
Price >= 353
Curtailment >= 491
Curtailment >= 217
```

These values currently correspond approximately to training percentiles, but the connection is manual. Retraining on new data could change the target thresholds while leaving Prolog unchanged.

Save the threshold values and their meanings in metadata, or explicitly version and justify the Prolog thresholds as independent domain rules.

#### Inference limitations

##### It is retrospective inference, not yet live inference

The current terminal input chooses a timestamp from an already transformed historical parquet. That is acceptable for development and demonstration, but it does not yet prove that new raw observations can be transformed for inference.

For a complete production-style pipeline, inference needs:

```text
raw recent history
→ A1 feature-engineering function
→ latest 43-feature row
→ XGBoost
→ Prolog
```

Because the model uses 168-hour lags, new inference requires at least 168 hours of history.

##### Paths depend on the working directory

Training saves to:

```python
Path("models")
```

and inference consults:

```python
prolog.consult("stress_rules.pl")
```

Both depend on where Jupyter was launched. Your data/model loading has robust project-root resolution, but these two paths do not.

This can cause:

- The model being saved into an unexpected `models/` directory.
- Prolog failing to find `stress_rules.pl`.

Use paths based on `PROJECT_ROOT` consistently.

##### A small preprocessing leakage exists in A1

A1’s rolling and lag features are causal, which is good. However, four imputed rows were generated using interpolation between observations on both sides of a gap. This means future values contributed to those rows.

It affects very few observations, but for strict forecasting you should either:

- Exclude `was_imputed == 1` rows from modelling, or
- Use causal imputation fitted only from previous observations.

#### Repository and submission concerns

- `electricity_transformed.parquet` is duplicated inside `A2/notebooks`, but the notebooks load the A1 copy. The duplicate is unused.
- The assignment says not to include datasets in the submission ZIP.
- `A2/src/inference.py` is empty.
- The current `A2/src/train_model.py` is empty.
- `A2/src/stress_rules.pl` duplicates only part of the notebook Prolog file.
- The new notebooks, models, source files, and data are currently untracked by Git.
- Two older notebooks are currently marked deleted.

#### Recommended correction order

1. Standardise `final_risk/2` versus `operational_risk/2`.
2. Resolve or temporarily remove the missing reserve-margin dependency.
3. Standardise `price/2` versus `electricity_price/2`.
4. Derive or supply the stress driver.
5. Query interpretation and recommendations, not just risk.
6. Restart and run the complete inference notebook end-to-end.
7. Introduce train/validation/test separation.
8. Save thresholds and model provenance in metadata.
9. Replace working-directory-relative paths.
10. Move the stable notebook logic into the currently empty Python scripts.

The immediate failure is not the XGBoost model—it is the inconsistent Prolog interface.

### Exchange 16

##### Prompt

> Please implement all of this

##### Output

#### Progress update 1

I’ll turn the current notebook logic into a reproducible development pipeline while keeping notebooks usable for exploration. The canonical implementation will live under `A2/src`, and the notebooks will call that code so training, saved artifacts, and inference cannot silently drift apart.

#### Progress update 2

The canonical scripts and Prolog rules are now in place. Next I’m simplifying both notebooks so they call those canonical functions instead of maintaining a second, divergent implementation; this also makes “Run All” exercise the same code used by the command line.

#### Progress update 3

The Prolog smoke test now returns a single final risk, an inferred driver, explanations, and recommendations. I also caught and corrected a rule-unification edge case that was emitting a generic recommendation alongside driver-specific ones. I’m now running the full training pipeline to regenerate a model and metadata that match the new 60/20/20 methodology.

#### Progress update 4

The full no-reserve inference now succeeds. While testing the optional reserve input, I found a Windows/PySwip integration quirk: PySwip forwards Python’s command-line arguments into SWI-Prolog during import, and `--reserve-margin` can crash its initialization. I’m isolating that import from application arguments so both CLI modes are reliable.

#### Progress update 5

The regenerated model now uses the corrected methodology: XGBoost won on validation macro-F1, then achieved 0.7527 macro-F1 and 0.8915 critical recall on the untouched test period. I’m adding lightweight rule tests for the failure cases we found—missing reserve margin, reserve escalation, congestion escalation, and the price predicate bridge—before the final audit.

#### Progress update 6

The complete pipeline now passes end-to-end inference and four symbolic-rule tests. I’m doing a final artifact audit—model/metadata consistency, notebook syntax, submission contents, and Git state—without staging or committing anything.

#### Final response 7

Implemented the complete development pipeline.

Key files:

- [model_training.ipynb](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/notebooks/model_training.ipynb)
- [inference.ipynb](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/notebooks/inference.ipynb)
- [train_model.py](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/src/train_model.py)
- [inference.py](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/src/inference.py)
- [stress_rules.pl](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/src/stress_rules.pl)
- [README.md](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/src/README.md)
- [test_hybrid.py](C:/Users/HansP/Documents/COMP3742-Capstone-Project/A2/src/test_hybrid.py)

What changed:

- Added chronological 60/20/20 training, validation, and untouched test periods.
- Removed imputed current/target rows from modelling.
- Compared Random Forest and XGBoost using validation macro-F1.
- Refit the selected model on training plus validation data.
- Saved a generic model bundle, native XGBoost model, metadata, features, thresholds, versions, splits, and metrics.
- Standardised the Python/Prolog interface around `final_risk/2`.
- Fixed `price/2` versus `electricity_price/2`.
- Made reserve margin optional.
- Added symbolic driver derivation, explanations, interpretations, and recommendations.
- Linked Prolog thresholds to the saved model thresholds instead of manually copied values.
- Added robust project-relative paths.
- Added requirements, documentation, source headers, and rule tests.
- Removed the unused duplicated parquet from `A2/notebooks`; the source remains under A1.
- Made the notebooks call the canonical source implementation.

Results:

- Selected model: XGBoost
- Test accuracy: 0.8107
- Test macro-F1: 0.7527
- Critical recall: 0.8915
- Four symbolic-rule tests pass.
- Full notebook inference passes.
- A 4.5% externally supplied reserve margin correctly escalates an ML `low` prediction to symbolic `critical`.

Run training:

```powershell
.\.venv\Scripts\python.exe A2\src\train_model.py
```

Run normal inference:

```powershell
.\.venv\Scripts\python.exe A2\src\inference.py `
  --timestamp "2026-09-23 23:00:00+10:00"
```

Run with an external 4.5% reserve margin:

```powershell
.\.venv\Scripts\python.exe A2\src\inference.py `
  --timestamp "2026-09-23 23:00:00+10:00" `
  --reserve-margin-bps 450
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest A2.src.test_hybrid -v
```

Before submission, replace `[Add your name]` and `[Add your FAN]` in every source header. Nothing was staged or committed. Git still shows the two older notebooks as deleted; those deletions already existed and were not restored.

### Exchange 17

##### Prompt

> What is the purpose of the 4.5 external reserve margin

##### Output

The 4.5% reserve margin is an optional example showing how Prolog can combine external operational knowledge with the ML prediction.

Reserve margin estimates how much available generation capacity remains above expected demand:

```text
Reserve margin =
(available capacity − expected demand) / expected demand × 100
```

For example, if expected demand is 10,000 MW and available capacity is 10,450 MW:

```text
Reserve margin = (10,450 − 10,000) / 10,000 × 100 = 4.5%
```

Your Prolog rules currently interpret:

```text
Below 5%     → critical
5–<10%       → high
10–<15%      → medium
15% or more  → low
```

Therefore, passing:

```powershell
--reserve-margin-bps 450
```

means 450 basis points = 4.5%. It demonstrates that the symbolic layer can override an optimistic ML result:

```text
ML prediction: low
External reserve margin: 4.5%
Final symbolic risk: critical
```

This is useful because the ML model detects patterns from historical data, while the symbolic system can enforce an explicit safety rule.

The reserve margin is external because the parquet contains generation output, but not total available generation capacity. Without an authoritative external value, omit the argument:

```powershell
python A2\src\inference.py --timestamp "..."
```

Prolog then treats reserve-margin risk as unavailable/low and bases the final decision on the ML prediction, imports, price, curtailment, and other available conditions.

The 4.5% value is only a demonstration scenario—not a measured reserve margin for that timestamp. The 5%, 10%, and 15% boundaries must also be justified with credible electricity-system sources or described as illustrative development thresholds.

## End-to-end pipeline structure

### Reviewing end-to-end pipeline requirements

#### User prompt

I need to restructure A2/source to ensure the entire pipeline runs end-to-end, could you check what needs to be updated to ensure this happens

#### Full final assistant response

Yes. **A2 needs a small restructuring plus several integration fixes** before an end-to-end run can be verified. The current copy of `A1_data_pipeline` includes `fetch_instance()`, so the data-loading component is now present.

| Area | What needs updating |
|---|---|
| **Single entry point** | Add a top-level runner that connects data preparation → training all four models → inference → combined assessment. These currently require separate launches. |
| **Imports** | `A1_data_pipeline/run_pipeline.py` uses `from DataLoader import DataLoader`. Make it package-compatible so the top-level runner can import it reliably. |
| **Paths** | Centralise paths for datasets, saved models, reports, rules and `.env`. The code currently saves beneath `Source`, while documentation and notebooks still reference `A2/...`. |
| **Dependencies** | The repository’s `.venv` is missing `openelectricity`, `xgboost`, `kanren`, `unification` and `pyswip`. SWI-Prolog is also unavailable on the current command path. |
| **Model artifacts** | There is no `Source/saved_models` directory yet. Only a network-stress model exists, under `models/`; the loaders expect artifacts under `saved_models/`. All four models need successful training. |
| **Live history** | `fetch_instance()` fetches 16 days and removes 168 warm-up rows, but only checks that **one** row remains. Price and renewable inference require **169** rows. Validate the shared history before running any model. |
| **Training dates** | Demand assumes a test period beginning September 2025 and a preceding 720-hour training window. Price uses fixed 2024/2025 split boundaries. Validate dataset coverage and class availability before starting expensive training. |
| **Unattended execution** | Demand training calls `plt.show()`, which can pause the pipeline. Save the chart by default and make displaying it optional. |
| **Failure reporting** | Inference catches subsystem failures and returns an assessment, but the command can still exit successfully. Return a failure status when required stages fail, while retaining the diagnostic output. |
| **Documentation/notebooks** | Update folder discovery, dataset paths and model paths. The README also still describes two-model execution, although training and inference now include four. |

I suggest this layout, keeping the existing model and author assignments:

```text
A2_puja0009/
├── Source/
│   ├── run_pipeline.py          # Complete workflow
│   ├── config.py               # Shared paths and settings
│   ├── training_pipeline.py
│   ├── inference.py
│   ├── A1_data_pipeline/
│   ├── models/                 # Python model implementations
│   ├── prolog/                 # Rules and reasoning adapters
│   ├── requirements.txt
│   └── README.md
├── data/
│   ├── raw/
│   └── processed/
├── saved_models/
├── Reports/
└── Misc/
```

Two further improvements matter for reliable verification:

- Add **historical inference** using an existing dataset and an explicit prediction clock. Price already supports a clock argument, but the shared inference runner does not expose it.
- Review gap filling: the shared transform interpolates using later observations, while Price replaces imputed values with past values. Align and document preprocessing so training and live inference use consistent assumptions.

**No files were changed during this review.** Full execution remains unverified because the checked environment lacks dependencies, training data and the complete set of saved models.

### Restricting top-level folders

#### User prompt

The only root Folders I can have is Source, Reports, and Misc. What would the updated file structure look like now

#### Full final assistant response

Keep all executable code in `Source`, generated reports in `Reports`, and datasets, trained models and notebooks in `Misc`:

```text
A2_puja0009/
├── Source/
│   ├── run_pipeline.py          # End-to-end entry point
│   ├── config.py                # Shared paths and settings
│   ├── training_pipeline.py
│   ├── inference.py
│   ├── requirements.txt
│   ├── README.md
│   ├── .env                    # Local API key; excluded from Git
│   ├── A1_data_pipeline/
│   │   ├── __init__.py
│   │   ├── DataLoader.py
│   │   ├── helper_functions.py
│   │   └── run_pipeline.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── DemandModel.py
│   │   ├── NetworkStressModel.py
│   │   ├── PriceModel.py
│   │   └── RenewableModel.py
│   └── prolog/
│       ├── __init__.py
│       ├── RenewableRules.py
│       ├── SystemAssessment.py
│       ├── uncertainty.pl
│       ├── stress_rules.pl
│       ├── price_spike.pl
│       ├── test_price_spike.pl
│       └── system_assessment.pl
├── Reports/
│   ├── puja0009_A2_SelfAssessment.pdf
│   ├── data_pipeline/           # Cleaning and transformation reports
│   ├── demand/                  # Evaluation results and plots
│   ├── network_stress/
│   ├── price/
│   ├── renewable/
│   └── inference/               # Saved combined assessments
└── Misc/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── saved_models/
    │   ├── demand_gp.npz
    │   ├── network_stress_model.joblib
    │   ├── network_stress_model_metadata.json
    │   ├── price_spike_model.joblib
    │   └── renewable_gb.npz
    └── notebooks/
        ├── model_training.ipynb
        └── inference.ipynb
```

`config.py` would resolve these locations relative to the A2 folder, so the pipeline works regardless of where you launch it. Only **Source, Reports and Misc** remain at the top level.

### Implementing the structure with separate pipelines

#### User prompt

Yes, please implement this but do not add a single entry point, I will manually just run machine learning and inference  pipeline separately

#### Full final assistant response

Implemented the layout with only **Source, Reports and Misc** at A2’s top level.

- Centralised paths in `Source/config.py`.
- Moved saved models and notebooks into `Misc`.
- Updated training, inference, data preparation and documentation.
- Kept training and inference separate; no combined entry point added.

From the repository root:

```bash
python A2_puja0009/Source/training_pipeline.py
python A2_puja0009/Source/inference.py
```

Syntax, path and file-integrity checks passed. Full execution still requires the missing dependencies, prepared dataset and all four trained models.

### Changing report output locations

#### User prompt

I do not want the pipeline to create data_pipeline and price folder in Reports

#### Full final assistant response

Updated both pipelines to save directly into `Reports`. Moved the existing reports there and removed the empty `data_pipeline` and `price` folders.

Documentation updated; syntax checks passed.

### Correcting report locations to Misc subfolders

#### User prompt

Sorry, they should be saved in Misc folder, keep the data pipeline and price folders

#### Full final assistant response

Done. Existing and future generated reports now go to:

- `Misc/data_pipeline/`
- `Misc/price/`

Updated the pipeline paths and README. Your self-assessment PDF remains in `Reports`. Syntax and moved-file integrity checks passed.

## System integration and conceptual questions

### Planning network-stress integration

> In the current master branch, I need to transfer/add my work in hans/network-stress-prediction to the A2 folder using the same structure as what is already defined in master. Could you look at this and plan the change and let me know what you will do before I approve for you to implement?

### Authorising integration

> Don't worry about adding /tests folder in the master branch. Implement everything else. Do not commit anything to github

### Reviewing inference complexity

> is the inference function in networkstressmodel overkill? Could it be simplified?
