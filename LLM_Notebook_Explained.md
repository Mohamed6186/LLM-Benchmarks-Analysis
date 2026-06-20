# LLM Benchmarks Analysis — Full Walkthrough (Plain-English Explanation)

This document explains **every line of code**, **every chart**, and **every column** in `LLM_Benchmarks_Analysis_finaledation.ipynb`. It's written so someone with basic Python knowledge can follow exactly what the notebook does and why.

---

## 0. What this notebook is

It's a team project that analyzes a CSV file called `llm_price_performance_tracker.csv`, containing **453 rows (models)** and **34 columns** (later grown to 36 with two engineered columns). Each row is one AI language model (like GPT-5, Claude, Llama, DeepSeek, etc.) from a provider (OpenAI, Google, Anthropic, Meta...). The columns describe its benchmark scores, price, speed, and metadata.

The notebook is split into 8 numbered sections, each owned by a different team member, answering one business question:

| # | Section | Question it answers |
|---|---------|----------------------|
| 1 | Data Cleaning & Preparation | Is the data trustworthy and ready to use? |
| 2 | Who Are the Players? | Who makes these models, and how big/open are they? |
| 3 | Who Performs Best? | Which models score highest, and do different metrics agree? |
| 4 | Who Is Cheapest and Fastest? | Which models cost least / respond quickest? |
| 5 | Best Value for Money | Which models give the most "intelligence" per dollar? |
| 6 | Who Is the Safest? | Which providers/models are safest and least likely to refuse? |
| 7 | Open vs. Closed Source | How do the two licensing models compare? |
| 8 | The Full Picture | Summary of all findings + README |

---

## 1. Column Dictionary — what every column means

| Column | What it represents |
|---|---|
| `model_name` | The human-readable name of the model (e.g. "GPT-5 (high)"). |
| `model_slug` | A URL/code-friendly version of the model name (lowercase, dashes). |
| `provider` | The company that built the model (OpenAI, Google, Anthropic...). |
| `provider_slug` | Code-friendly version of the provider name. |
| `aa_id` | The unique ID assigned by **Artificial Analysis**, the third-party site this data was scraped from. |
| `aa_intelligence_index` | Artificial Analysis's overall "Intelligence Index" — a single blended score summarizing general capability. |
| `aa_coding_index` | Artificial Analysis's coding-specific capability score. |
| `aa_math_index` | Artificial Analysis's math-specific capability score. |
| `composite_benchmark` | An aggregated benchmark score combining multiple tests into one number — the main "how good is this model" metric used throughout the notebook. |
| `mmlu_pro` | Score on **MMLU-Pro**, a harder version of the classic "Massive Multitask Language Understanding" exam (multi-subject knowledge test). |
| `gpqa_diamond` | Score on **GPQA Diamond** — graduate-level, PhD-difficulty science questions (physics, biology, chemistry). |
| `humanitys_last_exam` | Score on **Humanity's Last Exam (HLE)** — an extremely difficult, expert-level benchmark designed to be near the frontier of what models can't yet solve. |
| `livecodebench` | Score on **LiveCodeBench**, a coding benchmark that uses recently-published problems so models can't have memorized the answers. |
| `scicode` | Score on **SciCode** — coding tasks drawn from real scientific research problems. |
| `math_500` | Score on **MATH-500**, a 500-problem subset of competition-level math problems. |
| `aime_2025` | Score on the **2025 AIME** (American Invitational Mathematics Examination) — high-school competition math, very hard. |
| `input_cost_usd_per_1m` | Price in US dollars to send 1 million tokens of **input** (prompt) to the model. |
| `output_cost_usd_per_1m` | Price in US dollars for the model to generate 1 million tokens of **output** (response). |
| `blended_cost_usd_per_1m` | A weighted average of input and output cost — the single "how expensive is this model" number used in cost analysis. |
| `pricing_tier` | A category label (e.g. budget/standard/premium) bucketing models by price level. |
| `output_tokens_per_second` | Generation speed — how many tokens the model produces per second (higher = faster). |
| `time_to_first_token_s` | Latency — how many seconds before the model starts streaming its first token. |
| `time_to_first_answer_s` | Latency — how many seconds before a full first answer is ready. |
| `chatbot_arena_elo` | The model's **ELO rating** from Chatbot Arena, a crowd-sourced site where humans vote on which of two anonymous model responses they prefer. Higher = more preferred by humans. |
| `arena_elo_ci95` | The 95% confidence interval (margin of error) around the ELO rating — tells you how statistically reliable that ELO number is. |
| `arena_votes` | How many human votes went into computing that model's ELO — more votes = more reliable score. |
| `parameter_count` | The model's size, in **billions of parameters** (the model's internal "weights"). Bigger usually (not always) means more capable but slower/pricier. |
| `release_year` | The year the model was released. |
| `is_open_source` | `True`/`False` — whether the model's weights are publicly downloadable (open) or locked behind an API (closed). |
| `intelligence_per_dollar` | A pre-computed value metric: benchmark score divided by cost. (The notebook also recomputes its own version in Section 5.) |
| `price_performance_ratio` | Another pre-computed price-vs-performance metric, similar in spirit to `intelligence_per_dollar`. |
| `elo_benchmark_blend` | A pre-computed metric blending Arena ELO with benchmark scores into one number. |
| `speed_per_dollar` | A pre-computed metric: how much speed (tokens/sec) you get per dollar spent. |
| `scrape_date` | The date this row's data was collected/scraped from the source website. |
| `parameter_size` *(engineered in the notebook)* | A bucketed label — `Small` (<10B params), `Medium` (<100B), `Large` (≥100B), or `Unknown` (missing data). |
| `source_label` *(engineered in the notebook)* | Essentially a copy of `provider`, kept as a separate clean column for grouping/labeling in charts. |

**Why so many benchmark columns?** No single test fully captures "intelligence." `mmlu_pro`, `gpqa_diamond`, `humanitys_last_exam`, `livecodebench`, `scicode`, `math_500`, `aime_2025` are all *individual* tests; `composite_benchmark`, `aa_intelligence_index`, `aa_coding_index`, `aa_math_index` are *aggregated* scores built from groups of those individual tests.

---

## 2. Setup Cell

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
```

- `pandas` (`pd`) — loads and manipulates the table of data (the "DataFrame").
- `numpy` (`np`) — does fast numeric/array math (used later for `np.select`, `np.polyfit`, `np.linspace`).
- `matplotlib.pyplot` (`plt`) — the core plotting engine; every chart ultimately uses this.
- `seaborn` (`sns`) — a layer on top of matplotlib that makes nicer-looking statistical charts (boxplots, heatmaps, pairplots) with less code.
- `sns.set_theme(...)` — sets a global visual style: white background with grid lines, and text 1.1× the normal size so labels are readable.
- `plt.rcParams['figure.dpi'] = 120` — increases the resolution (sharpness) of every chart drawn afterward.

---

## 3. Section 1 — Data Cleaning & Preparation

### 3.1 Load the data
```python
df = pd.read_csv("llm_price_performance_tracker.csv")
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
```
- `pd.read_csv(...)` reads the CSV file into a DataFrame named `df` (the table the whole notebook works on).
- `.shape` returns `(rows, columns)` — confirms how big the dataset is (453 rows × 34 columns at this point).
- `.columns.tolist()` prints every column name, as a sanity check that the file loaded correctly.

### 3.2 Strip whitespace
```python
text_cols = df.select_dtypes(include='object').columns
for col in text_cols:
    df[col] = df[col].str.strip()
if ((df['model_name'].str.contains('^\\s|\\s$').sum())==0):
    print('Space removing done')
```
- `select_dtypes(include='object')` grabs only **text** columns (numbers are a different dtype).
- The `for` loop applies `.str.strip()` to each text column — this removes accidental leading/trailing spaces (a common cause of "duplicate" categories like `"OpenAI"` vs `"OpenAI "` that are really the same thing but treated as different by code).
- The `if` check re-scans `model_name` with a regex (`^\s` = starts with space, `\s$` = ends with space) to confirm zero such rows remain, and prints a confirmation message.

### 3.3 Fix column types
```python
df['arena_votes'] = df['arena_votes'].astype('Int64')
df['release_year'] = df['release_year'].astype('Int64')
df["scrape_date"] = pd.to_datetime(df["scrape_date"])
```
- `.astype('Int64')` (capital I) converts these columns to pandas' **nullable integer** type. Plain Python `int` can't hold a missing value (`NaN`), but `Int64` can — important since both columns have gaps.
- `pd.to_datetime(...)` converts `scrape_date` from a plain text string (e.g. `"3/31/2026"`) into a true date type, so later code could (in principle) do date math, sort by date, or compute date differences correctly.

### 3.4 Missing value audit
```python
values = [df[col].isnull().sum() for col in df.columns]
labels = [col for col in df.columns]
missing = pd.Series(values, index=labels)
print(missing[missing > 0])
```
- Builds a list of "how many NaNs per column," wraps it as a pandas `Series`, and prints only the columns with at least one missing value. This is a manual, more verbose way of doing what `df.isnull().sum()` does in one line — but it makes the looping logic explicit for teaching purposes.
- Result: columns like `chatbot_arena_elo` are missing in 407 of 453 rows (only 46 models have Arena ELO data — that benchmark is sparse), while core benchmark columns like `aa_intelligence_index` are missing only 7 times.

### 3.5 Parameter count statistics
```python
df["parameter_count"].describe()
```
- `.describe()` returns count, mean, std, min, 25%, 50% (median), 75%, max for the `parameter_count` column. This is used to decide sensible cutoffs for the Small/Medium/Large buckets created next (median ≈ 49B, 75th percentile ≈ 235B).

### 3.6 Feature engineering — `parameter_size` and `source_label`
```python
conditions = [
    df["parameter_count"] < 10,
    df["parameter_count"] < 100,
    df["parameter_count"] >= 100
]
choices = ["Small", "Medium", "Large"]
df["parameter_size"] = np.select(conditions, choices, default="Unknown")

providers = df["provider"].unique().tolist()
conditions = [df["provider"] == prov for prov in providers]
df["source_label"] = np.select(conditions, providers, default="Other")
```
- `np.select(conditions, choices, default=...)` checks each condition **in order** and assigns the matching label. So: under 10B params → `"Small"`; under 100B (but not under 10B) → `"Medium"`; 100B or more → `"Large"`; missing/unmatched → `"Unknown"`.
- The second block builds `source_label` by re-checking against every unique provider name and copying it across — functionally this just duplicates the `provider` column (since every provider has its own condition), giving a clean, explicitly-labeled column to group by in later charts.

### 3.7 Info summary & missing percentages
```python
df.info()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
```
- `df.info()` prints a compact table: column name, non-null count, dtype, and total memory used. A final sanity check that the type conversions worked.
- The missing-percentage calculation turns raw missing counts into percentages of the whole dataset (rounded to 2 decimals) and sorts them worst-first — easier to judge "is this column reliable enough to analyze?" than raw counts alone.

---

## 4. Section 2 — Who Are the Players? (4 charts)

### Chart 1 — "How many models are there per provider?" (vertical bar chart)
```python
provider_counts = df['provider'].value_counts()
provider_counts.plot(kind='bar', color='blue', edgecolor='black')
```
- `value_counts()` counts how many rows (models) belong to each `provider`.
- **X-axis**: provider name. **Y-axis**: number of models.
- **Purpose**: shows who the dominant players are in this dataset (e.g. Alibaba and OpenAI have the most listed models). It's a market-coverage map, not a quality measure.

### Chart 2 — "Distribution of Model Parameter Counts" (histogram)
```python
plt.hist(df['parameter_count'], bins=10, color='green', edgecolor='black')
```
- A histogram groups all parameter-count values into 10 equal-width bins and counts how many models fall in each.
- **X-axis**: parameter count (billions). **Y-axis**: frequency (number of models).
- **Purpose**: shows whether the market is dominated by small/efficient models or huge frontier models. It's saved to disk as `parameters_histogram.png`.
- **Relates to** Chart 1: combine "who" with "how big" — e.g. you can later ask "does the dominant provider mostly ship small or large models?"

### Chart 3 — "Percentage of Open Source vs. Closed Source Models" (pie chart)
```python
source_counts = df['is_open_source'].value_counts()
plt.pie(source_counts, labels=..., autopct='%1.1f%%', explode=(0.05, 0), shadow=True)
```
- Counts `True`/`False` in `is_open_source`, relabels them to readable text, then draws a pie chart with each slice's percentage shown (`autopct`).
- `explode=(0.05, 0)` slightly pulls the first slice out for visual emphasis; `shadow=True` adds a drop-shadow for depth.
- **Purpose**: instantly shows the open-source vs. closed-source market split. This sets up Section 7's deeper dive.

### Chart 4 — "Number of Models Released per Year" (bar chart)
```python
year_counts = df['release_year'].value_counts().sort_index()
year_counts.plot(kind='bar', color='yellow', edgecolor='black')
```
- Counts models per `release_year`, then `.sort_index()` puts years in chronological order (left-to-right) instead of by frequency.
- **Purpose**: shows the pace of releases — is the industry accelerating or slowing down? Useful context for interpreting whether "newer" models in other charts are actually newer.

**How these 4 charts work together**: Chart 1 tells you *who's shipping*, Chart 2 tells you *how big* what they ship is, Chart 3 tells you *how open* it is, and Chart 4 tells you *when* it happened. Together they paint the full competitive landscape before any performance judgments are made.

---

## 5. Section 3 — Who Performs Best? (6 visuals)

### Chart 5 — "Top 10 Models by Composite Benchmark Score" (horizontal bar)
```python
top_models = df.sort_values(by="composite_benchmark", ascending=False).head(10)
top_models.plot(x="model_name", y="composite_benchmark", kind="barh", color="green")
```
- Sorts all models by `composite_benchmark` (highest first) and keeps the top 10.
- `kind="barh"` draws a **horizontal** bar chart (model names are long, so horizontal bars keep them readable).
- The code also prints a small flagged list (`strange_models`) noting a few rows where the `provider` label looks mismatched with the model name (a data-quality note, not a real chart).
- **Purpose**: ranks models by the project's single best "overall capability" number.

### Chart 6 — "Top 10 Models by Chatbot Arena ELO" (horizontal bar)
```python
top_chatbot_arena_models = df.sort_values(by='chatbot_arena_elo', ascending=False).head(10)
top_chatbot_arena_models.plot(x='model_name', y='chatbot_arena_elo', kind='barh', color='green')
```
- Same idea as Chart 5, but ranked by **human-judged preference (ELO)** instead of an automated benchmark.
- **Purpose**: lets you compare "what the test says is best" (Chart 5) against "what humans actually prefer" (Chart 6) — these can disagree.

### Chart 7 — "Chatbot Arena ELO vs Composite Benchmark" (scatter plot)
```python
plt.scatter(x=df['chatbot_arena_elo'], y=df['composite_benchmark'])
```
- Every model becomes one dot: **X** = its ELO (human preference), **Y** = its composite benchmark (automated score).
- **Purpose**: directly tests whether Chart 5's and Chart 6's rankings agree. If dots roughly trend upward together, human taste and automated benchmarks align; scattered/flat dots mean they measure different things (e.g. humans may reward "likeable" answers that benchmarks don't capture, like tone or formatting).

### Chart 8 — "Top 10 Models by AA Coding Index" (horizontal bar)
```python
top_coding_models = df.sort_values(by='aa_coding_index', ascending=False).head(10)
```
- Ranks the top 10 by `aa_coding_index` — coding-specific skill.
- **Purpose**: identifies specialists. A model can rank lower on Chart 5 (general benchmark) but top this list if it's coding-focused.

### Chart 9 — "Top 10 Models by AA Math Index" (horizontal bar)
```python
top_math_models = df.sort_values(by='aa_math_index', ascending=False).head(10)
```
- Same as Chart 8, but ranked by `aa_math_index` (math-specific skill).
- **Purpose**: compared against Chart 8, this reveals whether the same models dominate both coding and math, or whether providers specialize (e.g. one company's models are math-strong but not code-strong).

### Chart 10 — "Correlation Matrix of Key Performance Metrics" (heatmap)
```python
columns = ['composite_benchmark', 'chatbot_arena_elo', 'aa_math_index', 'aa_intelligence_index', 'aa_coding_index']
corr = df[columns].corr()
sns.heatmap(corr, annot=True, cmap='jet', fmt='.2f')
```
- `.corr()` computes the **Pearson correlation coefficient** between every pair of the 5 listed columns — a number from -1 (perfectly opposite) to +1 (perfectly together), with 0 meaning no linear relationship.
- `sns.heatmap` colors each cell by its correlation value and `annot=True` prints the number on top, so you can read exact relationships at a glance instead of squinting at color alone.
- **Purpose**: a single 5×5 table answering "if a model is strong on metric A, is it also usually strong on metric B?" — e.g. you'd expect `composite_benchmark` and `aa_intelligence_index` to be highly correlated (≈1.0) since they measure similar things, while `chatbot_arena_elo` might correlate more loosely (it's human taste, not raw test scores).

### Chart 11 — "Pairwise Relationships Between Key Performance Metrics" (pairplot — a grid of charts)
```python
sns.pairplot(df[columns])
```
- This single command produces a **5×5 grid** of small charts: every off-diagonal cell is a scatter plot between two of the 5 metrics; every diagonal cell is a KDE (smoothed histogram) showing that single metric's distribution.
- **Purpose**: the heatmap (Chart 10) gives you one number per pair; the pairplot lets you actually *see* the shape of each relationship — e.g. spotting outliers, curved (non-linear) relationships, or clusters of models that a single correlation number would hide.

**How Section 3's charts relate**: Charts 5–6 rank models by two different "best" definitions, Chart 7 checks if those definitions agree, Charts 8–9 break "overall" performance into specialties, and Charts 10–11 quantify and visualize how all these metrics move together across the whole dataset.

---

## 6. Section 4 — Who Is Cheapest and Fastest? (3 charts)

### Chart 12 — "Top 5 Cheapest Models" (line + marker chart)
```python
top_5_cheapest = df.nsmallest(5, 'blended_cost_usd_per_1m')
plt.plot(top_5_cheapest['model_name'], top_5_cheapest['blended_cost_usd_per_1m'], marker='o', linestyle='-', color='blue')
```
- `.nsmallest(5, col)` is a shortcut for "sort ascending, keep the 5 smallest" — more efficient than a full sort for just grabbing extremes.
- Plotted as a connected line with circular markers (`marker='o'`) rather than bars — a deliberate stylistic choice to emphasize the *gap size* between consecutive cheapest models, not just their individual heights.
- **X-axis**: model name. **Y-axis**: blended cost (USD per 1M tokens).
- **Purpose**: identifies the most budget-friendly models.

### Chart 13 — "Top 5 Fastest Models" (seaborn bar chart)
```python
top_5_fastest = df.sort_values("output_tokens_per_second", ascending=False).head(5)
sns.barplot(data=top_5_fastest, x="model_name", y="output_tokens_per_second", palette="deep")
```
- Sorts descending by `output_tokens_per_second` and keeps the top 5 — the fastest text generators.
- `sns.barplot` with the `"deep"` color palette automatically assigns distinct, visually pleasant colors to each bar.
- **Purpose**: identifies the highest-throughput models — important for high-volume or latency-sensitive applications.

### Chart 14 — "Average Cost and Speed by Pricing Tier" (grouped bar chart)
```python
grouped = df.groupby("pricing_tier")[["blended_cost_usd_per_1m", "output_tokens_per_second"]].mean()
grouped.plot(kind="bar", figsize=(10,5))
```
- `groupby("pricing_tier")` buckets all models by their tier label, then `.mean()` computes the average cost and average speed *within each tier*.
- Plotting a DataFrame with two numeric columns via `.plot(kind="bar")` automatically draws **two bars side-by-side per tier** (one for cost, one for speed) — that's a grouped/clustered bar chart.
- **Purpose**: tests the intuitive assumption "do you pay more for speed?" by comparing tiers directly, rather than looking at 453 individual models.

**Relationship**: Charts 12–13 spotlight individual extreme models; Chart 14 zooms out to see if the cheap-vs-fast trade-off holds true at the *category* level across the whole pricing structure.

---

## 7. Section 5 — Who Offers the Best Value for Money? (3 charts)

### Setup
```python
value_df = df[required_cols].copy()
value_df['blended_cost_usd_per_1m'] = pd.to_numeric(value_df['blended_cost_usd_per_1m'], errors='coerce')
value_df['composite_benchmark'] = pd.to_numeric(value_df['composite_benchmark'], errors='coerce')
value_df = value_df.dropna(subset=['blended_cost_usd_per_1m', 'composite_benchmark'])
value_df = value_df[value_df['blended_cost_usd_per_1m'] > 0].copy()
```
- Builds a dedicated mini-DataFrame (`value_df`) with just the 4 columns needed, to avoid touching the main `df`.
- `pd.to_numeric(..., errors='coerce')` forces these columns to be true numbers, turning anything unparseable into `NaN` instead of crashing.
- `.dropna(...)` removes rows missing cost or benchmark — you can't compute a ratio without both.
- Filters out cost ≤ 0 — protects against a division-by-zero error later, since value metrics divide by cost.
- A folder `section5_outputs/` is created (`Path(...).mkdir(exist_ok=True)`) to save this section's chart images separately.

### Chart 15 — "Cost vs. Performance" (scatter + trend line)
```python
scatter = ax.scatter(value_df['blended_cost_usd_per_1m'], value_df['composite_benchmark'], c=value_df['composite_benchmark'], cmap='viridis', ...)
slope, intercept = np.polyfit(value_df['blended_cost_usd_per_1m'], value_df['composite_benchmark'], 1)
ax.plot(cost_range, slope * cost_range + intercept, color='red', linestyle='--', label=f'Trend: y={slope:.2f}x + {intercept:.2f}')
```
- Each model is a dot: **X** = cost, **Y** = benchmark score. The dot's *color* also encodes the benchmark score (`c=...`, `cmap='viridis'`) — a visual double-encoding that makes high-scoring models pop even more.
- `np.polyfit(x, y, 1)` fits a straight line (degree 1 = linear) through the data, returning its slope and intercept — i.e., "on average, how much does benchmark score increase per extra dollar of cost?"
- `np.linspace(...)` generates 100 evenly-spaced x-values across the cost range so the trend line draws smoothly from minimum to maximum cost.
- **Purpose**: visually answers "do you really get what you pay for?" Dots far *above* the red line are bargains (higher score than their price would predict); dots far *below* are overpriced for their performance.

### Chart 16 — "Top 5 Models by Value for Money" (horizontal bar, color-graded)
```python
value_df['intelligence_per_dollar'] = value_df['composite_benchmark'] / value_df['blended_cost_usd_per_1m']
top_5_value = value_df.nlargest(5, 'intelligence_per_dollar')[...]
colors = plt.cm.RdYlGn(np.linspace(0.35, 0.9, len(top_5_value)))
bars = ax.barh(range(len(top_5_value)), top_5_value['intelligence_per_dollar'].values, color=colors)
```
- Computes `intelligence_per_dollar` = benchmark score ÷ cost — the higher, the more "intelligence" you get per dollar.
- `.nlargest(5, ...)` keeps the 5 best-value models.
- `plt.cm.RdYlGn(np.linspace(0.35, 0.9, 5))` pulls 5 colors from the Red-Yellow-Green colormap to give each bar a distinct, value-themed shade (green = good value, intuitive).
- `ax.invert_yaxis()` flips the chart so the #1 ranked model appears at the **top**, matching natural reading order.
- Text labels are added at the end of each bar (`ax.text(...)`) showing the exact ratio value.
- Saved to `section5_outputs/section5_top_value_models.png`.
- **Purpose**: a clean leaderboard of the best bang-for-buck models — directly answers the section's core question.

### Chart 17 — "Value Comparison: Open-Source vs. Closed-Source Models" (boxplot + stripplot overlay)
```python
sns.boxplot(data=comparison_df, x='model_type', y='price_performance_ratio', ...)
sns.stripplot(data=comparison_df, x='model_type', y='price_performance_ratio', color='black', alpha=0.5, jitter=0.15, ...)
```
- The **boxplot** shows the statistical spread of `price_performance_ratio` for each group: the box spans the 25th–75th percentile, the line inside is the median, and whiskers/outlier dots show the range.
- The **stripplot**, drawn on top in black, shows every *individual* model as a dot, with `jitter=0.15` spreading them out horizontally so overlapping points are visible rather than stacking invisibly.
- Combining both is a common trick: the boxplot gives you the statistical summary, the stripplot keeps you honest about how many actual data points back it up (a “box” built from only 3 dots looks identical to one built from 300).
- The code then prints a `groupby('model_type').agg(Mean, Median, Std, Count)` table and computes the percentage difference between open- and closed-source mean value.
- **Purpose**: directly tests the hypothesis "open-source models are better value" with both a visual and a hard statistical summary.

**How Section 5's 3 charts connect**: Chart 15 establishes the overall cost/performance relationship across *all* models; Chart 16 zooms into the individual best performers; Chart 17 zooms out again to compare the *open vs. closed* category split, using the very same ratio computed for Chart 16.

---

## 8. Section 6 — Who Is the Safest? (2 charts)

> Note: there's no public "safety score" column in this dataset, so the team builds a **proxy metric**: `safety_score = (gpqa_diamond * 100 + composite_benchmark) / 2`. This isn't a real measured safety score — it's a stand-in built from existing benchmark numbers, used because no direct safety column exists. This is clearly worth flagging as a modeling assumption, not a ground truth.

### Setup
```python
df_clean = df.dropna(subset=['composite_benchmark', 'gpqa_diamond']).copy()
df_clean['safety_score'] = (df_clean['gpqa_diamond'] * 100 + df_clean['composite_benchmark']) / 2
```
- Drops rows missing either input to the formula, then computes the proxy `safety_score` for each remaining row.

### Chart 18 — "Average Safety Score by Provider" (bar chart with value labels)
```python
tp = ['OpenAI','Google','Anthropic','DeepSeek','xAI','Mistral','Meta']
p = df_clean[df_clean['provider'].isin(tp)].groupby('provider')['safety_score'].mean().sort_values(ascending=False)
b = plt.bar(p['provider'], p['avg'], color=[...7 hex colors...])
for x, y in zip(b, p['avg']):
    plt.text(x.get_x()+x.get_width()/2, y+0.4, f'{y:.1f}', ha='center', va='bottom', ...)
```
- Filters to 7 major providers (`.isin(tp)`), groups by provider, and averages the proxy `safety_score`, sorted highest-first.
- Each bar gets its own fixed color (a manual palette, one hex code per provider) rather than an auto-generated palette.
- The `for` loop manually places a text label with the exact average value just above each bar — `x.get_x()+x.get_width()/2` finds each bar's horizontal center, and `y+0.4` places the label slightly above the bar's top.
- Saved as `safety_by_provider.png`.
- **Purpose**: ranks providers by this proxy safety measure.

### Chart 19 — "Lowest Refusal Rate (Proxy)" (seaborn horizontal bar)
```python
df_clean['refusal_proxy'] = 100 - df_clean['composite_benchmark']
lowest_refusal = df_clean.sort_values('refusal_proxy', ascending=True).head(10)
sns.barplot(data=lowest_refusal, x='refusal_proxy', y='model_name', palette='viridis')
```
- Another proxy: `refusal_proxy = 100 - composite_benchmark`. This assumes models with *lower* benchmark scores might refuse more often — again, a stand-in assumption, not measured refusal data.
- Sorts ascending (smallest refusal-proxy = "best") and takes the top 10.
- **Purpose**: a second angle on safety/usability — models that are practically usable without excessive refusals, by this proxy's definition.

**Relationship between the two charts**: Chart 18 is a provider-level (company-wide) safety view; Chart 19 is a model-level usability view. Together they're meant to balance "is this provider generally careful?" against "does this specific model actually answer questions?" — though both rely on the same underlying composite-benchmark and GPQA numbers rather than independent safety data.

---

## 9. Section 7 — Open-Source vs. Closed-Source — Deep Dive

### Count by type (printed table, no chart)
```python
source_counts = df['is_open_source'].value_counts()
source_counts.index = ['Open Source' if x == True else 'Closed Source' for x in source_counts.index]
print(source_counts)
```
- Same counting logic as Chart 3 in Section 2, just printed as a plain table here instead of a pie.

### Chart 20 — "Open Source vs Closed Source Models" (bar chart)
```python
source_counts.plot(kind='bar', color=['green', 'red'])
plt.xticks(rotation=0)
```
- The same counts as a bar chart instead of a pie — gives an alternative, more precise-to-read view of the same split shown earlier as percentages.

### Average benchmark by type (printed, no chart)
```python
avg_performance = df.groupby('is_open_source')['composite_benchmark'].mean()
```
- Computes the mean `composite_benchmark` separately for open-source and closed-source models, printed as plain numbers (closed-source ≈ 51.4 vs. open-source ≈ 40.4 in this dataset) — i.e. on raw capability, closed-source models score higher on average, even though Section 5 showed open-source wins on *value for money*.

### Differences table (printed, no chart)
```python
differences = { "Source Code Access": {...}, "Modification": {...}, "Cost": {...}, "Security": {...}, "Support": {...} }
for category, values in differences.items():
    print(...)
```
- A hard-coded dictionary of conceptual (not data-derived) differences between open- and closed-source models, printed in a readable block format. This is general industry knowledge included for context, not something computed from the CSV.

---

## 10. Section 8 — The Full Picture & README

This final section is text only (no code/charts) — it's the project's executive summary, stating:
- **Performance**: benchmark and Arena-ELO leaders mostly agree; coding/math scores correlate with overall score (consistent with the high correlations seen in Chart 10).
- **Value**: open-source models win on cost-efficiency (consistent with Chart 17), even though Section 7 showed they score lower on raw benchmarks.
- **Safety** (by the notebook's proxy metric): Anthropic ranks highest, DeepSeek lowest.
- **Openness**: closed-source models score higher on average capability, but open-source wins on cost and transparency — the project's central tension.

The README block documents the dataset source, project purpose, and the 7-person team's role assignments (matching the 8 sections above).

---

## 11. Quick reference — all 20 charts at a glance

| # | Chart title | Type | Section |
|---|---|---|---|
| 1 | How many models are there per provider? | Bar | 2 |
| 2 | Distribution of Model Parameter Counts | Histogram | 2 |
| 3 | Percentage of Open Source vs. Closed Source Models | Pie | 2 |
| 4 | Number of Models Released per Year | Bar | 2 |
| 5 | Top 10 Models by Composite Benchmark Score | Horizontal bar | 3 |
| 6 | Top 10 Models by Chatbot Arena ELO | Horizontal bar | 3 |
| 7 | Chatbot Arena ELO vs Composite Benchmark | Scatter | 3 |
| 8 | Top 10 Models by AA Coding Index | Horizontal bar | 3 |
| 9 | Top 10 Models by AA Math Index | Horizontal bar | 3 |
| 10 | Correlation Matrix of Key Performance Metrics | Heatmap | 3 |
| 11 | Pairwise Relationships Between Key Performance Metrics | Pairplot (grid) | 3 |
| 12 | Top 5 Cheapest Models | Line + markers | 4 |
| 13 | Top 5 Fastest Models | Bar | 4 |
| 14 | Average Cost and Speed by Pricing Tier | Grouped bar | 4 |
| 15 | Cost vs. Performance (with trend line) | Scatter + regression line | 5 |
| 16 | Top 5 Models by Value for Money | Horizontal bar | 5 |
| 17 | Value Comparison: Open-Source vs. Closed-Source | Boxplot + stripplot | 5 |
| 18 | Average Safety Score by Provider | Bar w/ labels | 6 |
| 19 | Lowest Refusal Rate (Proxy) | Horizontal bar | 6 |
| 20 | Open Source vs Closed Source Models (recount) | Bar | 7 |

---

## 12. Things worth double-checking (as a 20-year-data-analyst would flag)

1. **The "safety" and "refusal" metrics in Section 6 are not real measured safety data** — they're formulas built from `gpqa_diamond` and `composite_benchmark`. The summary in Section 8 states "Claude has the highest refusal rate" and specific safety scores (95/100, 50/100), but those numbers don't actually appear anywhere in the executed code/output shown — worth re-deriving and citing directly from a cell's printed output before stating it as a finding.
2. **Several precomputed columns duplicate work the notebook redoes manually** — `intelligence_per_dollar`, `price_performance_ratio`, `elo_benchmark_blend`, and `speed_per_dollar` already exist in the raw CSV, but Section 5 recomputes its own `intelligence_per_dollar` from scratch. Worth checking whether the original and recomputed columns actually agree.
3. **Sparse columns** like `chatbot_arena_elo` (only 46 of 453 models have it) mean Chart 6, 7, and 10/11 (which include `chatbot_arena_elo`) are implicitly working with a much smaller, possibly biased subset of models — worth noting in any conclusions drawn from those specific charts.
4. **The "strange_models" flag** in Section 3 (mismatched provider labels for a few models) is noted but never actually cleaned/fixed in the data — it's flagged but not acted on.
