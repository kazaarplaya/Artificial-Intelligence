# AI Prompts Used for the Project

## 1. Data-cleaning and exploratory-analysis plan

> For this project, I got assigned to the explanatory analysis/data-cleaning of
> the dataset. What steps should I take to do this

**Purpose:** Establish a structured workflow for understanding, validating,
cleaning, analysing, and documenting the electricity dataset.

### Output

Your role should produce two things: a reproducible cleaned dataset and a clear explanation of what the data contains, what problems were found, and why each cleaning decision was made.

The current analysis notebook is empty, so you have a clean starting point.

#### Recommended workflow

##### 1. Establish the modelling question

Before cleaning, agree with the team on:

- What is being predicted or classified?
- What is the target column?
- What prediction horizon is required?
- Which information would actually be available at prediction time?
- What date range should be used?

This matters because data that is acceptable for descriptive analysis may cause leakage in a predictive model.

For example, several current variables may be mathematically or temporally related:

- `power` and `energy`
- `demand` and `demand_energy`
- `price`, `energy`, and `market_value`
- renewable generation and renewable proportion

Do not automatically give the model every column.

##### 2. Preserve the raw data

The current pipeline only prints the combined dataframe. Change the workflow eventually so that extraction produces an immutable raw file, preferably Parquet:

```text
A2/
  data/
    raw/
      electricity_raw.parquet
    processed/
      electricity_clean.parquet
  notebooks/
    wutt0019_notebook.ipynb
  src/
    data_cleaning.py
  reports/
    data_quality_summary.csv
```

Avoid repeatedly downloading 26 years of data while developing the notebook.

##### 3. Create a data dictionary

Document every field before modifying anything:

| Field group   | Examples              | Important considerations                              |
| ------------- | --------------------- | ----------------------------------------------------- |
| Time          | `timestamp`           | Timezone, hourly frequency, daylight saving           |
| Market        | `price`, `demand`     | Prices can legitimately be negative or extremely high |
| Generation    | `power`, `energy`     | Units differ: MW versus MWh                           |
| Environmental | `emissions`           | Missing historical periods may be structural          |
| Financial     | `market_value`        | May be derived from price and generation              |
| Renewable     | generation/proportion | Confirm whether proportion is 0–1 or 0–100            |
| Storage       | `storage_battery`     | Negative values may represent charging and be valid   |

Record the source, type, unit, expected range, and meaning of missing values.

##### 4. Perform an initial data audit

Your first notebook section should report:

```python
df.shape
df.head()
df.tail()
df.info()
df.describe(include="all")
df.isna().sum()
df.duplicated().sum()
```

Also calculate:

- Earliest and latest timestamps
- Data type of every column
- Missing percentage per column
- Missing percentage per year
- Number of unique timestamps
- Duplicate timestamps
- Time differences between consecutive rows
- Columns that are entirely empty over particular periods
- Constant or nearly constant columns

Missingness by year will be particularly important because newer OpenElectricity metrics may not exist back to 2000.

##### 5. Clean the timestamp carefully

For hourly electricity data:

1. Parse timestamps with `pd.to_datetime()`.
2. Confirm whether the API returns UTC or local NEM time.
3. Convert to UTC for storage and consistency.
4. Sort chronologically.
5. Check duplicate timestamps.
6. Check the expected one-hour cadence.
7. Identify continuous gaps.

NSW observes daylight saving time, so local calendar days can contain 23 or 25 hours. Do not delete these as duplicates without checking timezone information.

You can derive local-time features afterward:

```python
df["year"] = local_time.dt.year
df["month"] = local_time.dt.month
df["day_of_week"] = local_time.dt.dayofweek
df["hour"] = local_time.dt.hour
df["is_weekend"] = df["day_of_week"] >= 5
```

##### 6. Handle missing data according to its cause

Separate missing values into three categories:

- **Structural missingness:** a metric did not exist during older years.
- **Extraction gaps:** API requests or chunk boundaries skipped data.
- **Isolated observations:** one or two hourly readings are absent.

Do not apply a blanket `fillna(0)`. Zero means an actual measured value and is very different from unavailable data.

Possible rules:

- Preserve structurally unavailable periods as missing.
- Interpolate only short gaps when justified.
- Do not interpolate long gaps.
- Add an imputation indicator column if values are filled.
- Remove a column if it is unavailable for most of the intended modelling period.
- Consider shortening the study period if newer metrics are essential.

Create a cleaning log showing how many values each rule affected.

##### 7. Validate ranges without deleting valid energy-market events

Basic checks might include:

- Renewable proportion within its documented bounds
- Demand generally non-negative
- Emissions generally non-negative
- Timestamps within the requested range
- Numeric columns containing no strings or infinities

Be careful with “outliers”:

- Electricity prices can legitimately be negative.
- Price spikes can be genuine market events.
- Battery values may be signed because charging and discharging differ.
- Demand and generation extremes may correspond to heatwaves, outages, or market events.

Flag unusual values first. Investigate them against adjacent timestamps before removing or capping anything. Rolling median and median absolute deviation are generally more appropriate than a global z-score for this time series.

##### 8. Check internal consistency

Useful cross-checks include:

- At hourly resolution, whether energy and average power are approximately consistent.
- Whether `demand_energy` is consistent with hourly demand.
- Whether renewable proportion agrees approximately with renewable generation and gross demand.
- Whether market value behaves consistently with price and energy.
- Whether merged market and network timestamps align.

These checks can reveal extraction, aggregation, or unit problems that simple null checks will miss.

##### 9. Conduct explanatory data analysis

Your notebook should answer questions, not merely display plots.

Recommended analyses:

- Availability and missingness over time
- Overall distributions of each important metric
- Price distribution with negative prices and spikes clearly shown
- Long-term annual trends
- Monthly or seasonal patterns
- Hour-of-day and day-of-week patterns
- Weekday versus weekend demand
- Renewable proportion changes over time
- Relationships between price, demand, renewables, storage, and emissions
- Pearson and Spearman correlations
- Rolling averages and volatility
- Major regime changes or anomalous periods

For dense hourly data, aggregate plots to daily, monthly, or annual resolution so trends remain readable.

##### 10. Protect against modelling leakage

If this becomes a forecasting project:

- Split train, validation, and test data chronologically.
- Never randomly shuffle the full time series before splitting.
- Fit scalers and imputers on the training set only.
- Create rolling features using past observations only.
- Do not use future observations to interpolate training data.
- Confirm that derived variables are available at prediction time.

A reasonable split might be:

```text
Training   → oldest period
Validation → subsequent period
Testing    → most recent period
```

The exact dates should reflect the chosen modelling objective.

##### 11. Make cleaning reproducible

Use the notebook for exploration and explanation, but place the final rules in a Python module such as `A2/src/data_cleaning.py`.

A useful interface would be:

```python
def clean_electricity_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned dataset and a cleaning audit."""
```

The accompanying audit should capture:

- Rows before and after cleaning
- Duplicate rows removed
- Invalid timestamps removed
- Missing values before and after
- Values imputed
- Outliers flagged
- Final date range and column list

##### 12. Write the analysis as an argument

For every cleaning action, explain:

1. What problem you found.
2. How you detected it.
3. Why it matters.
4. What action you took.
5. How many records were affected.
6. What limitations remain.

For example:

> Price values below zero were retained because negative wholesale prices are valid market outcomes. They were flagged for analysis but were not treated as data-entry errors.

#### Your best immediate next steps

1. Ask the team to confirm the modelling target and usable date range.
2. Have the extraction pipeline save one raw Parquet file.
3. Load that file into your notebook.
4. Generate the schema, missingness, duplicate, timestamp-gap, and range reports.
5. Decide cleaning rules only after inspecting those results.
6. Move the accepted rules into a reusable cleaning module.
7. Produce a cleaned file, quality report, and explanatory notebook.

The most important principle is: **do not silently remove, replace, or cap anything**. Electricity-market data contains many values that look strange statistically but are meaningful in the real world.

## 2. OpenElectricity data dictionary

> I want you to document the fields I have provided and record source, type,
> unit, expected range and meaning of missing values of all of these from the
> Open Electricity API:
>
> ```python
> # Network metrics
> network_metrics: list[DataMetric] = [
>     DataMetric.POWER,
>     DataMetric.ENERGY,
>     DataMetric.EMISSIONS,
>     DataMetric.MARKET_VALUE,
>     DataMetric.STORAGE_BATTERY,
> ]
>
> # Market metrics
> market_metrics: list[MarketMetric] = [
>     MarketMetric.PRICE,
>     MarketMetric.DEMAND,
>     MarketMetric.DEMAND_ENERGY,
>     MarketMetric.DEMAND_GROSS,
>     MarketMetric.DEMAND_GROSS_ENERGY,
>     MarketMetric.GENERATION_RENEWABLE,
>     MarketMetric.GENERATION_RENEWABLE_ENERGY,
>     MarketMetric.GENERATION_RENEWABLE_WITH_STORAGE,
>     MarketMetric.GENERATION_RENEWABLE_WITH_STORAGE_ENERGY,
>     MarketMetric.RENEWABLE_PROPORTION,
>     MarketMetric.RENEWABLE_WITH_STORAGE_PROPORTION,
>     MarketMetric.CURTAILMENT,
>     MarketMetric.CURTAILMENT_ENERGY,
>     MarketMetric.CURTAILMENT_SOLAR_UTILITY,
>     MarketMetric.CURTAILMENT_SOLAR_UTILITY_ENERGY,
>     MarketMetric.CURTAILMENT_WIND,
>     MarketMetric.CURTAILMENT_WIND_ENERGY,
>     MarketMetric.FLOW_EXPORTS,
>     MarketMetric.FLOW_EXPORTS_ENERGY,
>     MarketMetric.FLOW_IMPORTS,
>     MarketMetric.FLOW_IMPORTS_ENERGY
> ]
> ```

**Purpose:** Produce a sourced data dictionary covering all API fields used by
the ingestion pipeline.

### Output

Created the complete data dictionary at:

[OpenElectricity metric data dictionary](A1/docs/data_dictionary.md)

It documents all 26 requested metrics with:

- API and pandas type
- Unit
- Source and derivation
- Expected range or validation rule
- Meaning of missing values
- Recommended missing-data policy
- Links to official sources

Important distinctions captured include:

- Negative electricity prices and market values can be valid.
- Renewable proportions can legitimately exceed 100%.
- Missing battery state of charge does not mean an empty battery.
- Import/export fields are separate non-negative directional magnitudes.
- Curtailment is conceptually non-negative, but negative calculated values should be investigated rather than automatically removed.
- Historical missingness may be structural because metric coverage varies.
- Missing values should not receive blanket zero imputation.

The definitions were checked against the live `/v4/metrics` catalogue and OpenElectricity's official guides for [demand](https://docs.openelectricity.org.au/guides/demand/), [renewables](https://docs.openelectricity.org.au/guides/renewables/), [curtailment](https://docs.openelectricity.org.au/guides/curtailment/), [emissions](https://docs.openelectricity.org.au/guides/emissions/), [prices](https://docs.openelectricity.org.au/guides/price/), and [batteries](https://docs.openelectricity.org.au/guides/batteries/).

## 3. Comparative storage-format evaluation

> Compare Parquet with other formats that could store this multi-year hourly
> electricity dataset, including CSV, compressed CSV, SQLite, Feather, Pickle,
> Excel, and JSON. Evaluate compression, performance, type preservation,
> portability, security, dependency requirements, and suitability for the
> pipeline, then recommend an appropriate storage strategy.

**Purpose:** Evaluate alternative data-storage formats against the project's
technical requirements instead of selecting a format without justification.

### Output

Parquet is useful here because hourly data across 26 years can become large. It stores typed columns efficiently, compresses well, and loads faster than CSV. Crucially, timestamps and numeric types survive a save/load cycle.

However, the project's requirements did not initially include `pyarrow`, so Parquet required:

```powershell
pip install pyarrow
```

Then:

```python
df.to_parquet("electricity_raw.parquet", index=False)
df = pd.read_parquet("electricity_raw.parquet")
```

#### Alternatives

| Format             | Advantages                                        | Disadvantages                                                    | Suitable here?                      |
| ------------------ | ------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------- |
| **Parquet**        | Small, fast, preserves types and nulls            | Binary; requires `pyarrow`; poor Git diffs                       | Best working format                 |
| **CSV**            | Universal, human-readable, easy to submit         | Large, slow, loses types and timezone metadata                   | Best compatibility format           |
| **Compressed CSV** | Universal and considerably smaller                | Still loses types; slower than Parquet                           | Good compromise                     |
| **SQLite**         | Queryable, typed, handles multiple related tables | More setup; binary database file                                 | Good if the dataset becomes complex |
| **Feather**        | Very fast and preserves types                     | Less compression and interoperability than Parquet               | Good for temporary local work       |
| **Pickle**         | Preserves Python/Pandas objects                   | Python-specific and unsafe to load from untrusted sources        | Avoid for shared project data       |
| **Excel**          | Easy for manual inspection                        | Row limit, type problems, inefficient, easy to edit accidentally | Only for small summaries            |
| **JSON**           | Human-readable and flexible                       | Large and inefficient for tabular time series                    | Not recommended                     |

##### CSV

The simplest option:

```python
df.to_csv("electricity_raw.csv", index=False)

df = pd.read_csv(
    "electricity_raw.csv",
    parse_dates=["timestamp"]
)
```

CSV does not remember that `timestamp` is a datetime, so types must be explicitly reconstructed when loading it.

Compressed CSV saves space without installing anything:

```python
df.to_csv("electricity_raw.csv.gz", index=False, compression="gzip")

df = pd.read_csv(
    "electricity_raw.csv.gz",
    parse_dates=["timestamp"]
)
```

##### SQLite

SQLite is worth considering if the project later stores multiple regions, facilities, or related tables:

```python
import sqlite3

with sqlite3.connect("electricity.db") as connection:
    df.to_sql("hourly_electricity", connection, if_exists="replace", index=False)
```

It allows subsets to be requested without loading everything:

```python
pd.read_sql_query(
    """
    SELECT *
    FROM hourly_electricity
    WHERE timestamp >= '2025-01-01'
    """,
    connection
)
```

#### Recommendation

For this project, retain:

- `electricity_raw.parquet` as the immutable working dataset.
- `electricity_clean.parquet` as the cleaned modelling dataset.
- `data_quality_summary.csv` for human-readable results.
- Optionally, a compressed CSV export if markers or teammates require a universally readable copy.

If avoiding another dependency is important, compressed CSV is a reasonable alternative, but the pipeline should include an explicit schema-loading function so timestamps and numeric columns are reconstructed consistently. Large generated datasets should not be committed to ordinary Git unless the team has agreed to use Git LFS or another data-storage mechanism.

## 4. Evidence-based cybersecurity review

> Review the implemented ingestion pipeline against the confidentiality,
> integrity, availability, and auditability objectives. Identify controls that
> are demonstrably present in the code, investigate credential exposure in Git
> history, identify remaining weaknesses in validation, error handling,
> resilience, file integrity, and dependency security, and recommend prioritised
> improvements without overstating the pipeline's current protections.

**Purpose:** Identify the confidentiality, integrity, availability, validation,
credential-management, and audit controls applied to the ingestion pipeline,
as well as its remaining security limitations.

### Output

Cybersecurity is partially applied to the ingestion pipeline. The strongest controls protect the API credential and validate data integrity, but secret-history, resilience, output integrity, and dependency-security controls could be improved.

#### Existing security practices

##### 1. API-key confidentiality

The key is loaded from a local `.env` file rather than being hardcoded:

```python
load_dotenv(PROJECT_DIR / "src" / ".env")
API_KEY = os.getenv("OPENELECTRICITY_API_KEY")
```

The repository's `.gitignore` excludes:

- `A1/src/.env`
- Raw and processed datasets
- Generated reports
- Virtual environments

The pipeline also never deliberately prints the API key. The OpenElectricity client sends it as a bearer credential over HTTPS, protecting it in transit.

However, `.env` is still plaintext on the local computer. Its protection depends on Windows account permissions.

##### 2. Restricted API requests

The API request parameters are defined in code:

- Network: `NEM`
- Region: `NSW1`
- Interval: `1h`
- Metrics: predefined SDK enumerations
- Dates: pipeline constants

This reduces the attack surface because users cannot inject arbitrary URLs, endpoints, metrics, or file paths through command-line input.

##### 3. Data integrity validation

The pipeline performs substantial validation before cleaning:

- Ensures required columns exist
- Rejects an empty dataset
- Requires timezone-aware timestamps
- Rejects missing timestamps
- Rejects duplicate timestamps
- Requires measurement columns to be numeric
- Rejects positive and negative infinity
- Verifies a continuous hourly timeline

These checks protect against corrupted, incomplete, malformed, or unexpectedly structured API responses.

##### 4. Duplicate protection at chunk boundaries

Because API request boundaries are inclusive, the same observation can occur in consecutive chunks. The pipeline resolves this explicitly:

```python
.drop_duplicates(
    subset=["timestamp", "metric"],
    keep="last"
)
```

This is preferable to `pivot_table()`, which might silently average duplicated values and conceal an ingestion problem.

##### 5. Availability through chunking

The pipeline retrieves data in 30-day chunks rather than requesting the entire period at once. This:

- Respects API range limits
- Reduces the size of each response
- Limits the impact of an individual failed request
- Reduces peak memory and network usage
- Avoids placing excessive load on the API

##### 6. Raw-data preservation and auditability

The raw API result is saved before cleaning or transformation:

```python
df.to_parquet(RAW_OUTPUT_PATH, ...)
```

The pipeline also produces JSON reports describing cleaning, transformation, and modelling decisions. This supports:

- Traceability
- Reproducibility
- Investigation of unexpected changes
- Comparison between raw and processed data

Preserving the raw dataset is important for detecting accidental or malicious alteration later.

##### 7. Safer serialization format

The pipeline uses Parquet rather than Python Pickle. Loading an untrusted Pickle file can execute arbitrary Python code, whereas Parquet is a data-oriented interchange format and does not intentionally deserialize Python objects.

Parquet files should still only be processed with maintained libraries, because file parsers can contain vulnerabilities.

#### Security weaknesses

##### Historical `.env` entry

Although `.env` is not currently tracked, a non-empty `A1/src/.env` existed in Git history in commit `9d94a92b`. It does not match the current key, and the commit message indicates it was intended as a dummy value.

Nevertheless:

- Confirm that it really was a dummy.
- If it was ever valid, revoke and replace it.
- Never assume adding a file to `.gitignore` removes it from existing history.
- Add automated secret scanning to the repository.

##### No explicit missing-key check

The pipeline currently allows `API_KEY` to be `None` and passes it into `DataLoader`. A clearer fail-fast check would be:

```python
if not API_KEY:
    raise RuntimeError(
        "OPENELECTRICITY_API_KEY is not configured."
    )
```

Do not include the key itself in the error message.

##### Broad exception handling

Both API functions catch every exception and call `sys.exit()`:

```python
except Exception as e:
    sys.exit(...)
```

Problems with this approach include:

- Network, authentication, validation, and programming errors are treated identically.
- Exception messages are printed without a redaction policy.
- The entire multi-year download is lost if a late chunk fails.
- Calling code cannot recover or test the failure properly.

It would be better to catch expected SDK exceptions, log a sanitised message, and re-raise an application-specific exception.

##### No retry, backoff, or checkpointing

The code has no explicit:

- Request timeout
- Retry limit
- Exponential backoff
- Rate-limit handling
- Progress checkpoint
- Resume capability

A temporary outage near the end of the 26-year ingestion could cause the whole process to fail. This is primarily an availability and reliability risk.

##### API units are discarded

Each API response includes a unit, but the pipeline discards it when pivoting. A compromised or changed API could return the right metric name with an unexpected unit, such as GWh instead of MWh, without being detected.

Before pivoting, validate:

```python
EXPECTED_UNITS = {
    "power": "MW",
    "energy": "MWh",
    "price": "$/MWh",
    # ...
}
```

The pipeline should fail if the API's reported unit differs from the expected unit.

##### No cryptographic integrity verification

Generated Parquet files do not have checksums or signatures. A file could be altered after ingestion without automatic detection.

A SHA-256 manifest could record:

- File hash
- Creation time
- Source endpoint
- Requested date range
- Row count
- Schema
- Pipeline version or Git commit

##### Writes are not atomic

The pipeline writes directly to the final filename. If it stops during writing, it could leave a corrupt or incomplete file.

A safer process is:

1. Write to a temporary file in the same directory.
2. Validate that the file can be reopened.
3. Calculate its checksum.
4. Atomically replace the final file.

##### Dependency supply-chain controls are limited

`requirements.txt` pins versions, which improves reproducibility, but it does not include hashes or vulnerability checking.

Recommended controls include:

- A lockfile with package hashes
- `pip-audit` in continuous integration
- Dependabot or Renovate alerts
- Regularly updating `openelectricity`, `pandas`, `pyarrow`, and networking dependencies

#### Summary by security objective

| Objective             | Current controls                                                            | Main gap                                                |
| --------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------- |
| Confidentiality       | Key stored in `.env`, ignored by Git, not printed, HTTPS API                | Plaintext secret and historical `.env` entry            |
| Integrity             | Schema, type, timestamp, duplicate and continuity validation                | No unit validation, checksums or signatures             |
| Availability          | API requests divided into 30-day chunks                                     | No retries, checkpoints, timeouts or resume             |
| Auditability          | Raw snapshot plus cleaning and transformation reports                       | Reports are not cryptographically linked to the dataset |
| Supply-chain security | Dependencies pinned to versions                                             | No hashes or automated vulnerability scanning           |
| Safe storage          | Fixed project paths, datasets excluded from Git, Parquet rather than Pickle | No atomic writes, access-control checks or encryption   |

A suitable report statement would be:

> The ingestion pipeline protects its API credential through environment-based configuration and HTTPS bearer authentication. It limits requests to predefined endpoints and metrics, retrieves data in bounded chunks, preserves an immutable raw-data stage, and validates schema, timestamps, duplicates, numeric types, infinite values, and hourly continuity. These controls improve confidentiality, integrity, availability, and auditability. Further hardening should introduce secret scanning, explicit credential validation, sanitised error handling, retry and checkpoint support, API-unit validation, atomic output writes, cryptographic checksums, and automated dependency vulnerability scanning.

## 5. Comparative survey of hybrid AI architectures

> Conduct a comparative survey of hybrid AI approaches for forecasting
> electricity-network stress and recommending operational responses. Compare
> LSTM with a rule-based expert system, XGBoost with a rule-based expert system,
> ANFIS, and deep neuro-symbolic models using forecasting accuracy,
> interpretability, ability to incorporate expert knowledge, implementation
> complexity, and suitability for electricity-network data. Recommend the most
> appropriate architecture, justify the decision, present its conceptual
> workflow, and provide an example showing how a forecast would lead to an
> explainable recommendation.

**Purpose:** Compare alternative hybrid forecasting and reasoning methods using
explicit selection criteria, then justify an architecture that balances
predictive capability, expert knowledge, explainability, and implementation
complexity.

### Output

A good survey for your project would compare hybrid approaches based on **forecasting accuracy, interpretability, ability to incorporate expert knowledge, complexity, and suitability for electricity-network data**.

#### Model Survey

Several hybrid AI approaches were considered for predicting electricity-network stress and recommending appropriate responses.

| Hybrid approach                        | Description                                                                                                                                              | Advantages                                                                                                                 | Limitations                                                                               | Suitability     |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------- |
| **LSTM + Rule-Based Expert System**    | An LSTM learns temporal patterns in demand, weather and renewable generation, while predefined logical rules evaluate forecasts and recommend responses. | Strong time-series forecasting; rules are transparent and easy to modify; clearly separates prediction and reasoning.      | Requires sufficient historical data and expert-defined rules.                             | **High**        |
| **XGBoost + Rule-Based Expert System** | XGBoost predicts demand or stress from historical and environmental features, with rules applied to its predictions.                                     | Accurate on structured data, computationally efficient and relatively interpretable.                                       | Does not naturally model long-term temporal dependencies without engineered lag features. | **High**        |
| **ANFIS**                              | Combines neural networks with fuzzy-logic rules, allowing learned relationships to interact with fuzzy concepts such as "high demand".                   | Handles uncertainty and non-linear relationships well and has been successfully applied to electricity-demand forecasting. | Rule bases can become complicated as the number of input variables increases.             | **Medium–High** |
| **Deep Neuro-Symbolic Models**         | Models such as Logic Tensor Networks integrate logical constraints directly into neural-network learning.                                                | Can enforce domain knowledge during learning and combine reasoning with pattern recognition.                               | Considerably more complex to design, train and explain.                                   | **Medium**      |

Hybrid and ensemble approaches are particularly suitable for energy forecasting because electricity data contains complex, temporal and non-linear relationships.

**Selected Architecture:** An **LSTM + rule-based expert system** is proposed. The LSTM can analyse historical demand, renewable generation, weather and grid conditions to forecast future electricity demand or network stress. The resulting predictions are then passed to a logical inference system containing expert-defined rules and operational constraints. The inference system can validate predicted conditions, classify the severity of network stress and recommend appropriate responses.

This architecture was selected because it provides a clear separation between **data-driven forecasting** and **explainable decision-making**, while remaining considerably simpler to implement and evaluate than tightly integrated neuro-symbolic approaches.

Conceptually, your final system would therefore be:

**Input data → ETL → LSTM forecast → Stress assessment → Logical inference/rules → Recommendation**

For example: the **LSTM predicts unusually high demand**, then your logical system evaluates something like _high forecast demand + low renewable generation + low reserve margin → high network stress → recommend demand-response measures_. This makes the hybrid aspect of your project very clear.
