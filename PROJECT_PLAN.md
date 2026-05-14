# 🎯 CS280 / CS485 — Company Bankruptcy Prediction
## Master Plan, Roadmap & Implementation Prompts

> **Domain:** Banking & Finance — *Company Bankruptcy Prediction (Taiwan Economic Journal, 1999–2009)*
> **Dataset:** `data.csv` — **6,819 rows × 96 columns** (95 financial ratios + 1 binary target `Bankrupt?`)
> **Type:** Supervised binary classification — **severely imbalanced** (220 positives = 3.23 %).
> **Final deliverables:** ONE clean executed `.ipynb` + ONE 7–12 page PDF report.

### Verified dataset facts (from independent inspection)

| Property | Value |
|---|---|
| Rows × Cols | 6 819 × 96 |
| Missing values | 0 |
| Duplicated rows | 0 |
| Positive class | 220 (3.23 %) |
| Constant columns | `Net Income Flag` (=1, must be dropped) |
| Pre-existing binary flag | `Liability-Assets Flag` (0/1, must NOT be scaled) |
| Features with \|skew\| > 1 | 81 / 95 |
| Features with \|skew\| > 3 | 68 / 95 |
| Features with max ≥ 1×10⁹ | 23 (extreme-scale / saturation artefacts; e.g. `Current Asset Turnover Rate`, `Cash Turnover Rate`, `Total Asset Growth Rate`) |

> **Statistical reality check:** a 20 % stratified holdout contains **only ~44 positives**. A single point estimate on that split is noisy. Our **primary evaluation is therefore repeated stratified 5-fold CV (5 repeats × 5 folds = 25 splits)** on the training set; the holdout is used **once** as a final confirmation.

> **Empirical audit update:** a quick leakage-safe 5-fold benchmark showed gradient boosting is the strongest model family on this dataset: `HistGradientBoostingClassifier` and `XGBoost` led default PR-AUC, while Logistic Regression gave the highest recall at lower precision. The roadmap therefore keeps the original 4 distinct model families but adds **HistGradientBoosting** as a justified boosting challenger.

---

## 0. Strategic Framing (what the graders will look for)

Reading the PDF carefully, the evaluation is **not about chasing 99 % accuracy**. It is about:

1. **A scientifically defensible workflow** (no data leakage, no AutoML, no copy-paste).
2. **Justified metric choice** — the PDF requires metrics to match the application and explicitly warns that accuracy alone can be inappropriate in sensitive settings.
3. **Genuinely distinct model families** (≥ 3) compared on a level playing field.
4. **Comparison tables + plots/curves** (mandatory).
5. **Inference stage** on unseen data.
6. **Ability to defend every choice orally.**

### ⚠ Pitfalls in the inspiration notebooks we MUST avoid
| Pitfall seen | Why it's wrong | What we do instead |
|---|---|---|
| Reporting "99.67 % F1" via DT + One-Class SVM trained only on minority | Likely data leakage / metric inflation | Use proper stratified split BEFORE any resampling; report PR-AUC, recall, F1 honestly |
| SMOTE applied to the full dataset before splitting | Synthetic minority points leak into the test set | Apply SMOTE **only inside CV folds** via `imblearn.Pipeline` |
| Scaling/PCA fit on full data then split | Test statistics leak into the scaler | Fit scaler/PCA on train only, transform test |
| Accuracy as sole headline metric | Trivially 97 % by predicting "not bankrupt" everywhere | Headline = **PR-AUC + Recall@high-precision**, also report F1, ROC-AUC, MCC, Balanced Acc |
| 8+ models with no justification | "Spray and pray" — graders will ask *why* each one | Pick 4 **conceptual families** and add at most one justified challenger |
| Dropping `Net Income Flag` blindly | It's a constant column (all = 1) — must verify | Detect and drop with an explicit `nunique()==1` check |

### Non-negotiable leakage rule: split before learned preprocessing

The notebook may do descriptive EDA on the full dataset to understand the problem, but the **modelling workflow must split before any preprocessing step that learns parameters from data**.

Correct modelling order:

1. Load `data.csv`, strip column names, inspect shape/missing/duplicates.
2. Drop only verified structural columns such as `Net Income Flag`.
3. Separate `X` and `y`.
4. Run the stratified train/test split.
5. Fit all learned preprocessing **inside pipelines** on training folds only: winsorisation quantiles, scaling statistics, feature-selection scores, SMOTE synthetic sampling, and model parameters.
6. Transform/evaluate the untouched test set only through the fitted final pipeline.

Never fit `Winsorizer`, `StandardScaler`, `SelectKBest`, PCA, SMOTE, or any model on the full dataset before splitting.

---

## 1. Final Roadmap (8 phases)

| # | Phase | Output |
|---|---|---|
| 1 | Project Setup & Problem Definition | Intro markdown, imports, reproducibility seed, dataset load |
| 2 | Exploratory Data Analysis (EDA) | Class imbalance, distributions, correlation, outliers, target relationships |
| 3 | Preprocessing & Feature Engineering | Constant-column drop, outlier strategy, scaling, optional log-transform & feature selection |
| 4 | Train/Test Split & Imbalance Strategy | Stratified 80/20 holdout; SMOTE / class_weight discussion |
| 5 | Modelling — 4 distinct families + 1 boosting challenger | Logistic Regression, Random Forest, XGBoost, HistGradientBoosting, Tiny MLP (≤ 3 layers) |
| 6 | Hyperparameter Tuning | `StratifiedKFold` + `RandomizedSearchCV` (or Optuna) inside leakage-safe pipelines |
| 7 | Evaluation & Scientific Comparison | Metrics table, ROC curves, PR curves, confusion matrices, threshold analysis, calibration check |
| 8 | Model Selection & Inference | Choose champion, retrain on full train, run inference on held-out & synthetic new rows |

### Chosen models (justification for the oral exam)

| Model | Family | Why we picked it |
|---|---|---|
| **Logistic Regression (L2)** | Linear / probabilistic | Interpretable baseline, coefficients map directly to financial-ratio risk |
| **Random Forest** | Bagging tree ensemble | Handles non-linearities & feature interactions, robust to outliers, gives feature importance |
| **XGBoost** | Gradient boosting | State-of-the-art on tabular; native handling of imbalance via `scale_pos_weight` |
| **HistGradientBoosting** | Gradient boosting challenger | Sklearn-native boosted trees; quick audit performed very well, useful as a robust non-external benchmark |
| **Tiny MLP (1 hidden layer)** | Neural network | Optional course-compliant NN, kept unquestionably tiny under the PDF's ≤ 3-layer rule |

### Chosen primary metrics (defendable)

> "In bankruptcy prediction, **missing a bankrupt firm (false negative) costs the bank far more than a false alarm**. We therefore optimise for **Recall** and the area under the **Precision–Recall curve (PR-AUC)**, which is the appropriate summary statistic when the positive class is ~3 %. ROC-AUC, F1, MCC and Balanced Accuracy are reported as secondary metrics. Plain accuracy is reported only to show how misleading it is (≈ 0.968 by always predicting *not bankrupt*)."

---

## 2. Implementation Prompts — one per phase

Each block below is a **self-contained prompt** that can be handed to an implementer AI (or used yourself). They are ordered; do not skip ahead. Each phase ends with a checklist the implementer must satisfy before moving on.

---

### 🟦 PHASE 1 — Project Setup & Problem Definition

**Prompt to implementer:**
> Create a new notebook called `bankruptcy_prediction_final.ipynb` in the workspace root. Add an opening markdown cell titled **"Company Bankruptcy Prediction — Banking & Finance (CS280/CS485 Lab Project)"** with:
> - Team name / placeholder for member names.
> - One paragraph stating the **business problem**: given 95 financial ratios of a Taiwanese company, predict whether it will go bankrupt. False negatives are very costly for lenders → primary metric = PR-AUC and Recall.
> - One paragraph stating the **dataset source** (Taiwan Economic Journal, 1999–2009, originally on Kaggle), shape (6 819 × 96), target column `Bankrupt?`, base rate ≈ 3.2 %.
> - A bullet list of the **planned workflow** (the 8 phases above).
>
> Then add a code cell that:
> 1. Imports `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `warnings` (`warnings.filterwarnings('ignore')`).
> 2. Sets `RANDOM_STATE = 42`, `np.random.seed(42)`.
> 3. Configures `sns.set_theme(style='whitegrid')`, `pd.set_option('display.max_columns', 100)`.
> 4. Loads `data.csv` into `df`, strips column whitespace with `df.columns = df.columns.str.strip()`.
> 5. Prints `df.shape`, `df['Bankrupt?'].value_counts(normalize=True)`, and `df.head()`.
>
> **Checklist before proceeding:** shape == (6819, 96); base rate of positive class printed and approximately 0.032.

---

### 🟦 PHASE 2 — Exploratory Data Analysis

**Prompt to implementer:**
> Add a markdown header **"## 2. Exploratory Data Analysis"** followed by EDA in clearly separated code cells, each preceded by a markdown explanation of what we are looking at and what we expect to find.
>
> Implement the following EDA steps:
> 1. **Sanity checks** — `df.info()`, `df.isna().sum().sum()` (expect 0), `df.duplicated().sum()`.
> 2. **Constant-column detection** — compute `df.nunique()` and explicitly highlight any column with a single unique value (you will find `Net Income Flag` is constant = 1). Print them in a markdown table.
> 3. **Class imbalance plot** — a single bar chart of `Bankrupt?` counts with the percentages annotated on top of each bar. Markdown commentary: "Base rate 3.23 % — accuracy is a misleading metric here."
> 4. **Summary statistics** — `df.describe().T` and flag columns whose `max` is enormous (e.g. > 1e9) as suspicious scales. Explicitly report that 23 features cross this threshold and that some high values are frequent, not isolated single-row outliers.
> 5. **Skewness scan** — `df.drop(columns=['Bankrupt?']).skew().abs().sort_values(ascending=False).head(20)` — show how many features have |skew| > 1.
> 6. **Correlation with target** — `df.corr(numeric_only=True)['Bankrupt?'].drop('Bankrupt?').sort_values()` — plot the top-15 most negatively and top-15 most positively correlated features as a horizontal bar chart.
> 7. **Feature–feature correlation heatmap** — compute the correlation matrix of the top-30 most target-correlated features and plot as a heatmap to reveal multicollinearity clusters (you will see ROA(A)/ROA(B)/ROA(C) are essentially the same feature).
> 8. **Distribution comparison** — for the top-6 features most correlated with the target, draw KDE plots split by `Bankrupt?` to visually confirm separability.
> 9. **Outlier visualisation** — boxplots (log-scaled y-axis) for those same top-6 features.
>
> End the phase with a markdown **"Key takeaways from EDA"** block listing:
> - Severe class imbalance (3.23 %).
> - `Net Income Flag` is constant → drop.
> - Many ratios are heavily skewed and on very different scales → scaling required.
> - High multicollinearity within ROA / Net-Value families → tree models robust; linear models may need regularisation.
> - Strong predictors visible: `Net Income to Total Assets`, `ROA(A/B/C)`, `Persistent EPS`, `Debt ratio %`, `Borrowing dependency`.
>
> **Checklist:** at least 6 figures produced; markdown commentary under EACH figure; explicit identification of the constant column.

---

### 🟦 PHASE 3 — Preprocessing & Feature Engineering

**Prompt to implementer:**
> Add markdown header **"## 3. Preprocessing & Feature Engineering"**. Implement:
>
> 1. **Drop constant columns**: `df = df.drop(columns=['Net Income Flag'])`. Print remaining shape (expect `(6819, 95)`).
> 2. **Separate X / y**: `X = df.drop(columns=['Bankrupt?'])` → 94 features; `y = df['Bankrupt?']`.
> 3. **Identify the binary flag column:** `binary_cols = ['Liability-Assets Flag']`, `numeric_cols = [c for c in X.columns if c not in binary_cols]` (= 93 numeric ratios).
> 4. **Outlier strategy — winsorisation (NOT row deletion).** Justify in markdown: "With only 220 positives we cannot afford to delete rows. Several columns contain data-entry artefacts saturating at ~1×10¹⁰ (e.g. `Current Asset Turnover Rate`, `Cash Turnover Rate`). We cap each numeric feature at its 1st and 99th training-set percentile." Provide a small custom transformer `Winsorizer(BaseEstimator, TransformerMixin)` that stores `low_/high_` quantiles in `fit` and clips in `transform`. **Important:** define the transformer here, but do not fit it on the full dataset.
> 5. **Build a `ColumnTransformer`** that applies `Winsorizer → StandardScaler` to `numeric_cols` and `'passthrough'` to `binary_cols`. This becomes the front block of every model pipeline. The `ColumnTransformer` must be fitted only after the train/test split, inside cross-validation or final training.
> 6. **(Linear-model-only branch)** add `SelectKBest(f_classif, k=30)` AFTER the column transformer for Logistic Regression. Tree-based and NN models keep all 94 features.
> 7. Do NOT apply a global log-transform — justify: tree models are scale-invariant, and Winsorizer + StandardScaler already tame the tails for linear/NN models.
>
> **Checklist:** `X.shape == (6819, 94)`; `Winsorizer` has a unit-test cell (fit on `[[1],[2],…,[100]]`, assert `transform([[0],[200]])` returns `[[1.99],[99.01]]` ± tolerance); `ColumnTransformer` correctly leaves `Liability-Assets Flag` untouched (assert via a quick smoke-test).

---

### 🟦 PHASE 4 — Train / Test Split & Imbalance Strategy

**Prompt to implementer:**
> Add markdown header **"## 4. Train/Test Split & Imbalance Strategy"**.
>
> 1. **Stratified holdout split** — `train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)`. Print class distribution in train and test to confirm preservation. **Touch the test set ONLY at the very end of Phase 7.**
> 2. Markdown **"## 4.1 Why we split before learned preprocessing"** explaining that `Winsorizer`, `StandardScaler`, `SelectKBest`, PCA, SMOTE, and the models themselves must learn only from training data. The full dataset can be inspected for EDA, but it cannot be used to fit transformations used for modelling.
> 3. Markdown **"## 4.2 Why we DO NOT oversample the full dataset"** explaining the leakage trap and that SMOTE will live *inside* every CV pipeline.
> 4. Markdown **"## 4.3 Two complementary imbalance strategies"**:
>    - **Algorithm-level:** `class_weight='balanced'` / `balanced_subsample` (Logistic Regression, RF, HistGradientBoosting) or `scale_pos_weight=(neg/pos)` (XGBoost) or class-aware sampling for MLP.
>    - **Data-level:** `SMOTE(random_state=42, k_neighbors=5)` applied only to the training fold via `imblearn.pipeline.Pipeline`.
>    - We will test both and let the CV decide per model.
>
> **Checklist:** train/test class proportions printed; explanation cells present; no resampling has been applied to `X_train` yet.

---

### 🟦 PHASE 5 — Modelling (4 distinct families)

**Prompt to implementer:**
> Add markdown header **"## 5. Modelling"**. For each model, create a **subsection** with: (a) a markdown explanation of the algorithm (1 short paragraph: main idea + key hyperparameters), (b) the pipeline definition, (c) a default-config fit on the training set with 5-fold stratified CV reporting mean PR-AUC, ROC-AUC, F1, Recall.
>
> Use `from imblearn.pipeline import Pipeline as ImbPipeline` everywhere so SMOTE is fold-safe.
>
> ### 5.1 Logistic Regression
> Pipeline: `ColumnTransformer(Winsorizer → StandardScaler on numeric ratios, passthrough binary flag) → SelectKBest(k=30) → SMOTE → LogisticRegression(class_weight='balanced', max_iter=2000, solver='liblinear')`.
> Markdown: "Linear model on log-odds; coefficients directly interpretable as risk contributions; serves as our transparent baseline."
>
> ### 5.2 Random Forest
> Pipeline: `ColumnTransformer(Winsorizer on numeric ratios, passthrough binary flag) → SMOTE → RandomForestClassifier(class_weight='balanced_subsample', n_estimators=300, n_jobs=-1, random_state=42)`.
> Markdown: "Bagged decision trees; reduces variance via bootstrap + feature subsampling; robust to outliers and scale."
>
> ### 5.3 XGBoost
> Pipeline: `ColumnTransformer(Winsorizer on numeric ratios, passthrough binary flag) → XGBClassifier(scale_pos_weight=neg/pos, n_estimators=400, eval_metric='aucpr', tree_method='hist', random_state=42)`. Do NOT use SMOTE here — `scale_pos_weight` is the principled solution.
> Markdown: "Sequential boosting of weak trees minimising a regularised loss; current SOTA on tabular data."
>
> ### 5.4 HistGradientBoosting challenger
> Pipeline: `ColumnTransformer(Winsorizer on numeric ratios, passthrough binary flag) → HistGradientBoostingClassifier(class_weight='balanced', max_iter=250, learning_rate=0.04, random_state=42)`. Do NOT use SMOTE here; this model uses weighted loss and histogram-based split finding.
> Markdown: "A sklearn-native boosted tree model. It belongs to the same broad family as XGBoost, so it is not counted as a separate conceptual family, but the data audit showed it is a strong practical challenger worth evaluating."
>
> ### 5.5 Tiny MLP (≤ 3 layers, course rule)
> Pipeline: `ColumnTransformer(Winsorizer → StandardScaler on numeric ratios, passthrough binary flag) → SMOTE → MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=300, early_stopping=True, random_state=42)`.
> Markdown: "One hidden layer plus the output layer, so it remains unquestionably tiny and compliant with the PDF's ≤ 3-layer rule. Learns a smooth non-linear decision boundary without becoming a deep-learning project."
>
> ### 5.6 Cross-validation harness
> Write a helper function `evaluate_cv(pipeline, X, y, name)` that runs `cross_validate(..., cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42), scoring=['average_precision','roc_auc','f1','recall','precision','balanced_accuracy'], n_jobs=-1, return_train_score=False)` and returns a 1-row DataFrame with mean ± std across the 25 folds. Concatenate the model rows into `cv_results_default` and display it. **This repeated-CV protocol is our primary evaluation** because the holdout test set contains only ~44 positives and a single point estimate would be statistically fragile.
>
> **Checklist:** 4 conceptual families are represented (linear, bagging, boosting, neural), plus the HistGradientBoosting challenger; each has markdown explanation + pipeline + CV row; combined comparison table printed.

---

### 🟦 PHASE 6 — Hyperparameter Tuning

**Prompt to implementer:**
> Add markdown header **"## 6. Hyperparameter Tuning"**.
>
> For each candidate model, build a `RandomizedSearchCV` with:
> - `scoring='average_precision'` (PR-AUC — our primary metric),
> - `cv=StratifiedKFold(5, shuffle=True, random_state=42)` (single-pass 5-fold here to keep tuning runtime reasonable; the *winner* will be re-validated with the full repeated CV in Phase 6's final step),
> - `n_iter=30`, `n_jobs=-1`, `random_state=42`,
> - `refit=True`.
>
> Suggested search spaces (the implementer may refine):
>
> | Model | Search space |
> |---|---|
> | Logistic Regression | `C ∈ loguniform(1e-3, 1e2)`, `penalty ∈ {'l1','l2'}`, `selectkbest__k ∈ [20,30,50,70]` |
> | Random Forest | `n_estimators ∈ [200,400,600]`, `max_depth ∈ [None,6,10,16]`, `min_samples_leaf ∈ [1,2,4,8]`, `max_features ∈ ['sqrt','log2',0.5]` |
> | XGBoost | `n_estimators ∈ [300,500,800]`, `max_depth ∈ [3,5,7,9]`, `learning_rate ∈ loguniform(1e-3, 0.3)`, `subsample ∈ [0.7,0.85,1.0]`, `colsample_bytree ∈ [0.6,0.8,1.0]`, `reg_lambda ∈ loguniform(1e-2, 10)` |
> | HistGradientBoosting | `max_iter ∈ [150,250,400]`, `learning_rate ∈ loguniform(1e-3, 0.2)`, `max_leaf_nodes ∈ [15,31,63]`, `l2_regularization ∈ loguniform(1e-4, 10)`, `min_samples_leaf ∈ [10,20,40]` |
> | MLP | `hidden_layer_sizes ∈ [(8,),(16,),(32,),(64,)]`, `alpha ∈ loguniform(1e-5, 1e-2)`, `learning_rate_init ∈ loguniform(1e-4, 1e-2)` |
>
> After each search, print `best_params_` and `best_score_` and store the fitted best estimator in a dict `best_models = {...}`. Build a comparison DataFrame `tuned_cv_results` with the same columns as Phase 5 — but this time re-evaluate each tuned estimator with the **full `RepeatedStratifiedKFold(5, n_repeats=5)`** so the tuned-vs-default lift is measured on the same protocol.
>
> Display **side-by-side bar chart** of default vs tuned PR-AUC for all candidate models.
>
> **Checklist:** `best_models` dict has all fitted candidate pipelines; `tuned_cv_results` DataFrame displayed; lift plot present; total runtime printed and reasonable (< ~10 min on CPU if possible, otherwise document actual runtime).

---

### 🟦 PHASE 7 — Evaluation, Comparison & Threshold Analysis

**Prompt to implementer:**
> Add markdown header **"## 7. Evaluation on the Held-Out Test Set"**. **This is the first time we touch `X_test`.**
>
> 1. For each tuned candidate model, compute on `X_test`: `y_pred_proba`, then at default threshold 0.5: `precision`, `recall`, `f1`, `balanced_accuracy`, `mcc`, `roc_auc`, `pr_auc`, and `brier_score_loss`. Assemble into `final_test_results` DataFrame, sorted by PR-AUC descending. Display with `.style.background_gradient(cmap='Greens', subset=['pr_auc','recall','f1'])`.
> 2. **ROC curve overlay** — one figure with all candidate models' ROC curves + the diagonal chance line + AUC values in the legend.
> 3. **Precision–Recall curve overlay** — same idea, with the base-rate horizontal line (`y = 0.032`) drawn as the no-skill baseline. Mark each curve's PR-AUC in the legend.
> 4. **Confusion matrices** — a grid of `ConfusionMatrixDisplay` plots (one per model), all at threshold 0.5.
> 5. **Threshold sweep for the champion** — for the model with the highest PR-AUC, plot Precision, Recall and F1 vs threshold ∈ [0, 1]. Mark the threshold that maximises F1 and the one that gives Recall ≥ 0.85, and print the resulting confusion matrices at both. Markdown discussion: "In a banking setting, a recall-oriented threshold (e.g. 0.30) is usually preferred over the F1-optimal threshold; we expose this trade-off explicitly."
> 6. **Feature importance** — for the boosted/tree champion, plot the top-15 feature importances. For Logistic Regression, plot the top-15 |coefficient| values. Comment on agreement with EDA Phase 2.
> 7. **Probability calibration check** — plot a calibration curve and report Brier score for the champion. If probabilities are poorly calibrated, add a short note that the classification threshold is decision-oriented and the raw probabilities should be interpreted as risk scores rather than exact default probabilities.
>
> End with a markdown **"Discussion"** block that:
> - Ranks the models honestly.
> - Explicitly cautions against claims like "99 % F1 in other notebooks" — show why those numbers are likely the product of leakage or evaluation on resampled data.
> - States which model we select and *why*, in terms of the chosen primary metrics.
>
> **Checklist:** 1 metrics table + 5 figure groups (ROC overlay, PR overlay, confusion grid, threshold sweep, calibration curve) + feature-importance plot + discussion markdown.

---

### 🟦 PHASE 8 — Model Selection, Final Refit & Inference

**Prompt to implementer:**
> Add markdown header **"## 8. Final Model Selection & Inference"**.
>
> 1. Declare the champion model (likely a boosted-tree model, but let the data decide). Refit the chosen pipeline on the **full training set** (`X_train, y_train`) — we keep the test set untouched as our certificate.
> 2. Print the final test metrics one more time as the official numbers reported in the PDF report.
> 3. **Inference utility** — write a function `predict_bankruptcy(model, new_rows_df, threshold=0.30)` that:
>    - Validates that all 94 feature columns are present.
>    - Returns a DataFrame with columns `['risk_score', 'prediction', 'risk_level']` where risk levels are `'Low' (<0.15)`, `'Medium' (0.15–0.5)`, `'High' (≥0.5)`. Use `risk_score` rather than overclaiming calibrated probability unless the Phase 7 calibration check is good.
> 4. **Demonstrate inference** on:
>    - The first 5 rows of `X_test` (compare to true labels).
>    - 2 synthetic rows: one "healthy" company (copy a `y==0` example and slightly perturb), one "distressed" company (copy a `y==1` example and slightly perturb). Print risk scores and risk levels.
> 5. **Save artefacts** — `joblib.dump(champion_pipeline, 'bankruptcy_champion.joblib')` and save the test-metric DataFrame to `final_metrics.csv` (these are referenced from the PDF report).
>
> End with a markdown **"## Conclusion"** block (5–8 lines): summary of the problem, chosen model, headline metric values, limitations (small positive sample, possible drift across years), and one-sentence future-work idea (e.g., calibration via Platt scaling, cost-sensitive learning with explicit FN cost).
>
> **Checklist:** champion declared explicitly; refit done; inference function works on the 5 + 2 examples; joblib & csv files saved; conclusion markdown present.

---

## 3. Report (PDF) — Skeleton (do this AFTER the notebook is fully executed)

Use the Overleaf template recommended in the PDF. Suggested 7–12 page structure:

1. **Introduction** (~1 p.) — business motivation, asymmetric cost of FN vs FP, research question.
2. **Background** (~1.5 p.) — brief literature on credit-risk modelling; quick descriptions of LR / RF / gradient boosting / MLP with the math intuition + 1 hyperparameter table per model.
3. **Methodology** (~3 p.) — dataset description, EDA findings, preprocessing pipeline diagram, train/test protocol, imbalance handling, CV scheme, tuning procedure, metric justification.
4. **Results & Interpretation** (~3 p.) — default-vs-tuned comparison table, ROC & PR overlays, confusion matrices, threshold analysis figure, calibration curve, feature-importance interpretation in financial terms.
5. **Discussion & Limitations** (~0.5 p.) — honest discussion of what could fail (concept drift, dataset age, geographic specificity).
6. **Conclusion** (~0.5 p.).
7. **References** (Kaggle source + key sklearn / XGBoost papers).

---

## 4. Defence-Cheat-Sheet (what to memorise before the oral)

- **Why PR-AUC over ROC-AUC?** With 3 % positives, ROC-AUC is optimistically biased by the abundance of true negatives; PR-AUC focuses on the rare positive class.
- **Why no SMOTE before split?** Synthetic samples generated near a real test point would leak ground-truth structure → optimistic metrics.
- **Why winsorise instead of drop outliers?** Only ~220 positives; deleting rows can erase signal. Capping caps tail influence on linear/NN models while keeping every observation.
- **Why these model families?** Linear (LR) ⇄ Bagging (RF) ⇄ Boosting (XGB/HGB) ⇄ Neural (MLP) — four genuinely different inductive biases, satisfying the "distinct families" PDF requirement while letting the data choose the best boosting implementation.
- **Why is the MLP only 1 hidden layer?** The PDF caps NN depth at 3 layers and warns that this is not a deep-learning project. A one-hidden-layer MLP is unquestionably tiny and easy to defend.
- **Why `scale_pos_weight` for XGB and SMOTE for the others?** XGB's loss is directly re-weightable; SMOTE plays nicer with distance-based methods (LR, MLP) and tree ensembles that have no built-in weighting.
- **Top financial predictors and their sign:** `Net Income to Total Assets` (↓ → bankrupt), `Debt ratio %` (↑ → bankrupt), `Borrowing dependency` (↑ → bankrupt), `Persistent EPS` (↓ → bankrupt). These match economic intuition.

---

## 5. Reproducibility Checklist (run before submission)

- [ ] Notebook restarts → "Run All" → executes end-to-end without errors.
- [ ] All random states fixed to 42.
- [ ] No `pip install` cell that fails offline; all imports listed at top.
- [ ] Every figure has a title + axis labels + (where relevant) legend.
- [ ] Every code section has a preceding markdown explanation.
- [ ] No commented-out dead code blocks.
- [ ] Submitted files: `bankruptcy_prediction_final.ipynb` + `report.pdf` only.

---

*End of plan. Hand each phase prompt one at a time to the implementer AI, validate the phase checklist, then move on. Good luck — this plan is defensible end-to-end.*
