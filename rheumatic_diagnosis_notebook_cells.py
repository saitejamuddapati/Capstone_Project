# ==============================================================================================
#  EXPLAINABLE AI FOR EARLY DIFFERENTIAL DIAGNOSIS OF RHEUMATIC AND AUTOIMMUNE DISEASES
#  Kaggle notebook code, one section per cell (26 cells). Run the cells in order, top to bottom.
#  Each "# %%" line starts a new notebook cell. Attach the dataset with "Add Data" first; no GPU needed.
#  The final cell writes /kaggle/working/project_artifacts.zip with the model, tables and figures.
# ==============================================================================================

# %% [markdown]
# ## Cell 1 – Setup and libraries
# Imports every library the project uses and installs any that are missing (LightGBM, XGBoost, CatBoost, Optuna, SHAP). Defines a small log() helper that prints the elapsed time before each message.

# %%
# ==============================================================================================
# CELL 01 – SETUP AND LIBRARIES
# What it does: Imports every library the project uses and installs any that are missing
# (LightGBM, XGBoost, CatBoost, Optuna, SHAP). Defines a small log() helper that prints the
# elapsed time before each message.
# Output: No numeric output; a line per library installed if one was missing.
# ==============================================================================================
import os, sys, glob, json, time, warnings, subprocess, importlib, shutil, textwrap
warnings.filterwarnings("ignore")
T0 = time.time()
def log(m): print(f"[{(time.time()-T0)/60:5.1f} min] {m}", flush=True)

def _ensure(pkg):
    try: importlib.import_module(pkg)
    except ImportError:
        log(f"installing {pkg} ..."); subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
for _p in ["lightgbm", "xgboost", "catboost", "optuna", "shap", "seaborn", "openpyxl", "tabulate"]: _ensure(_p)
try:      # optional, used only for the oversampling check; NOT installed on purpose (installing it can upgrade scikit-learn mid-run)
    import imblearn; HAVE_IMBLEARN = True
except Exception: HAVE_IMBLEARN = False

import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, log_loss,
                             confusion_matrix, matthews_corrcoef, cohen_kappa_score, precision_recall_fscore_support,
                             roc_curve, auc)
import lightgbm as lgb, xgboost as xgb, optuna, shap
from catboost import CatBoostClassifier
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style="whitegrid", context="notebook")


# %% [markdown]
# ## Cell 2 – Project settings
# Collects every adjustable setting in one place: random seed (42), test-set size (15%), number of tuning trials, number of repeated splits (10), safe-mode accuracy target (95%), robustness rule and sensitivity-mode limit. Creates the output folders.

# %%
# ==============================================================================================
# CELL 02 – PROJECT SETTINGS
# What it does: Collects every adjustable setting in one place: random seed (42), test-set size
# (15%), number of tuning trials, number of repeated splits (10), safe-mode accuracy target (95%),
# robustness rule and sensitivity-mode limit. Creates the output folders.
# Output: Nothing printed; the settings are used by all later cells.
# ==============================================================================================
FAST_MODE = False or os.environ.get("FAST_MODE", "0") == "1"   # True = quick smoke test only
SCRIPT_VERSION = "1.0"
SEED = 42
TEST_SIZE = 0.15                 # frozen test set, never used for any decision
N_JOBS = max(1, os.cpu_count() or 2)
TRIALS_PER_ROUND = dict(rf=2, lgb=4, xgb=3, cat=2) if FAST_MODE else dict(rf=5, lgb=12, xgb=9, cat=4)
TUNE_TIMEOUT_CAP = 3600          # safety cap only (seconds); the trial counts decide
MAX_ROUNDS = 1 if FAST_MODE else 2
TARGET_OOF_MACRO_F1 = 0.86
N_REPEATS = 2 if FAST_MODE else 10            # repeated-split check
CEILING_SIM = 50_000 if FAST_MODE else 400_000
SHAP_SAMPLES = 24 if FAST_MODE else 120
SHAP_BACKGROUND = 20 if FAST_MODE else 30
SHAP_PERMUTATIONS = 3 if FAST_MODE else 8
N_BOOT = 100 if FAST_MODE else 1000
SELECTIVE_TARGET_ACC = 0.95
MIN_MEASURED_TESTS = 3
ROBUST_PRIMARY = "noise_5pct"        # disturbance that decides deployment (fixed before the run)
ROBUST_SWITCH_MARGIN = 0.010         # switch only if an alternative is >= 1.0 point more accurate under it
ROBUST_REPEATS = 1 if FAST_MODE else 3
SENS_MAX_ACC_DROP = 0.010            # sensitivity mode may cost at most 1.0 point of overall accuracy (training CV)
H2H_TREE_GRID = (100,) if FAST_MODE else (100, 300, 600)
GB_TREES = 30 if FAST_MODE else 100
TARGETS = {"accuracy": 0.85, "macro_f1": 0.85, "balanced_accuracy": 0.85, "auc_ovr": 0.985,
           "ece": 0.03, "SLE_f1": 0.98, "selective_accuracy": 0.95, "top2_accuracy": 0.97}
OUT = "/kaggle/working/outputs" if os.path.isdir("/kaggle/working") else os.path.abspath("./outputs")
if os.path.isdir(OUT): shutil.rmtree(OUT)
for d in ["artifacts", "paper_results", "figures"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)


# %% [markdown]
# ## Cell 3 – Shared helper functions and the range-aware Bayes model
# Defines the functions that both training and the deployed backend use: reading raw values ("Positive", "weak positive", blanks) into numbers without inventing missing values, the 15 optional engineered features, the range-aware Bayes model (fit and predict), one predictor for every model type, exact per-test explanations, and the next-best-test recommender. The same text is later written into inference.py, so training and deployment use identical code.

# %%
# ==============================================================================================
# CELL 03 – SHARED HELPER FUNCTIONS AND THE RANGE-AWARE BAYES MODEL
# What it does: Defines the functions that both training and the deployed backend use: reading raw
# values ("Positive", "weak positive", blanks) into numbers without inventing missing values, the
# 15 optional engineered features, the range-aware Bayes model (fit and predict), one predictor
# for every model type, exact per-test explanations, and the next-best-test recommender. The same
# text is later written into inference.py, so training and deployment use identical code.
# Output: Nothing printed.
# ==============================================================================================
SHARED_CODE = r'''
import numpy as np, pandas as pd
BASE_COLS = ["Age","Gender","ESR","CRP","RF","Anti-CCP","HLA-B27","ANA","Anti-Ro","Anti-La","Anti-dsDNA","Anti-Sm","C3","C4"]
SEROLOGY = ["HLA-B27","ANA","Anti-Ro","Anti-La","Anti-dsDNA","Anti-Sm"]
LABS = ["ESR","CRP","RF","Anti-CCP","C3","C4"]
TESTS = LABS + SEROLOGY
_POS = {"positive","pos","+","++","+++","1","1.0","true","yes","reactive","detected","present",
        "weak positive","weakly positive","low positive","strong positive","strongly positive"}
_NEG = {"negative","neg","-","0","0.0","false","no","non-reactive","nonreactive","not detected","absent"}
_BLANK = {"","nan","none","null","na","n/a","not done","not tested","nd","missing"}
_GENDER = {"male":1.0,"m":1.0,"man":1.0,"female":0.0,"f":0.0,"woman":0.0}
ESR_SCALE, CRP_SCALE = 49.0, 30.0          # dataset upper limits, used only for scaling (trees ignore scale)
C3_LOW_REF, C4_LOW_REF = 90.0, 10.0        # typical lower reference limits (mg/dL; labs vary)
def _norm(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip().lower()
def encode_raw(df, return_issues=False):
    """Raw table (original names/text) -> numeric table. Unmeasured tests stay NaN (never filled).
    Values that cannot be understood are reported (return_issues=True), never silently dropped."""
    X = pd.DataFrame(index=df.index); issues = [[] for _ in range(len(df))]
    def num(col):
        out = []
        for i, v in enumerate(df[col].tolist()):
            s = _norm(v)
            if s in _BLANK: out.append(np.nan); continue
            try: out.append(float(s))
            except ValueError: out.append(np.nan); issues[i].append(f"{col}: could not read '{v}' (treated as not tested)")
        return out
    X["Age"] = num("Age")
    g = []
    for i, v in enumerate(df["Gender"].tolist()):
        s = _norm(v)
        if s in _GENDER: g.append(_GENDER[s])
        else:
            g.append(np.nan)
            if s not in _BLANK: issues[i].append(f"Gender: could not read '{v}'")
    X["Gender"] = g
    for c in LABS: X[c] = num(c)
    for c in SEROLOGY:
        vals = []
        for i, v in enumerate(df[c].tolist()):
            s = _norm(v)
            if s in _POS: vals.append(1.0)
            elif s in _NEG: vals.append(0.0)
            else:
                vals.append(np.nan)
                if s not in _BLANK: issues[i].append(f"{c}: could not read '{v}' (treated as not tested)")
        X[c] = vals
    X = X[BASE_COLS].astype(float)
    return (X, issues) if return_issues else X
def engineer(X):
    """Clinical features. Every ratio is NaN unless BOTH source tests were really measured (no fake values)."""
    F = X.copy()
    F["n_missing_labs"] = X[LABS].isna().sum(axis=1)
    F["n_missing_sero"] = X[SEROLOGY].isna().sum(axis=1)
    F["n_pos_sero"] = (X[SEROLOGY] == 1).sum(axis=1)
    F["n_neg_sero"] = (X[SEROLOGY] == 0).sum(axis=1)
    F["ESR_CRP_ratio"] = X["ESR"] / (X["CRP"] + 0.5)
    F["inflam_index"] = X["ESR"] / ESR_SCALE + X["CRP"] / CRP_SCALE
    F["rf_ccp_min"] = X[["RF","Anti-CCP"]].min(axis=1, skipna=False)
    F["rf_ccp_max"] = X[["RF","Anti-CCP"]].max(axis=1, skipna=False)
    F["rf_x_ccp"] = X["RF"] * X["Anti-CCP"]
    F["C3_x_C4"] = X["C3"] * X["C4"]
    F["complement_vs_ref_min"] = pd.concat([X["C3"]/C3_LOW_REF, X["C4"]/C4_LOW_REF], axis=1).min(axis=1, skipna=False)
    F["low_C3"] = (X["C3"] < C3_LOW_REF).astype(float).where(X["C3"].notna())
    F["low_C4"] = (X["C4"] < C4_LOW_REF).astype(float).where(X["C4"].notna())
    F["sjogren_ab_pos"] = X[["ANA","Anti-Ro","Anti-La"]].max(axis=1, skipna=True).where(X[["ANA","Anti-Ro","Anti-La"]].notna().any(axis=1))
    F["lupus_ab_pos"] = X[["Anti-dsDNA","Anti-Sm"]].max(axis=1, skipna=True).where(X[["Anti-dsDNA","Anti-Sm"]].notna().any(axis=1))
    return F
def member_input(name, F):
    """Random Forest cannot read blanks, so it gets a clear 'not tested' code (-1). Boosters read blanks natively."""
    return F.fillna(-1.0) if name == "rf" else F
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

# ---- Range-aware Bayes model (plain dict of arrays -> safe to save and load anywhere) ----
def _resolution(v):
    u = np.unique(v); return float(np.min(np.diff(u))) if len(u) > 1 else 1.0
def bayes_fit(X, y, K, sub=None, alpha=0.1, prior="empirical"):
    """Each test is independent evidence. Lab density = histogram whose bin edges sit exactly on every
    disease's min/max for that lab (learned from the training data), optionally sub-divided to width <= sub[lab].
    Serology = probability of 'positive' per disease. A test that was not done is simply left out."""
    sub = sub or {}; y = np.asarray(y)
    cnt = np.bincount(y, minlength=K).astype(float)
    pri = np.log(cnt / cnt.sum()) if prior == "empirical" else np.log(np.full(K, 1.0 / K))
    tabs = {}
    for c in LABS:
        col = X[c].values.astype(float); v_all = col[~np.isnan(col)]; eps = _resolution(v_all) / 2
        pts = {v_all.min() - eps, v_all.max() + eps}
        for k in range(K):
            vk = col[y == k]; vk = vk[~np.isnan(vk)]
            if len(vk): pts |= {vk.min() - eps, vk.max() + eps}
        pts = sorted(pts); edges = [pts[0]]; w = sub.get(c)
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.ceil((b - a) / w))) if w else 1
            edges += list(np.linspace(a, b, n + 1)[1:])
        edges = np.array([pts[0] - 1e6] + edges + [pts[-1] + 1e6])
        width = np.diff(edges); width[[0, -1]] = 1.0
        mids = (edges[:-1] + edges[1:]) / 2; mids[0] = edges[1] - eps; mids[-1] = edges[-2] + eps
        Pb = np.zeros((K, len(width)))
        for k in range(K):
            vk = col[y == k]; vk = vk[~np.isnan(vk)]; h, _ = np.histogram(vk, edges)
            Pb[k] = (h + alpha) / (h.sum() + alpha * len(h))
        tabs[c] = dict(edges=edges, logdens=np.log(Pb / width), pbin=Pb, mids=mids)
    for c in SEROLOGY:
        col = X[c].values.astype(float); Pb = np.zeros((K, 2))
        for k in range(K):
            vk = col[y == k]; vk = vk[~np.isnan(vk)]; p = (vk.sum() + alpha) / (len(vk) + 2 * alpha); Pb[k] = [1 - p, p]
        tabs[c] = dict(edges=None, logdens=np.log(Pb), pbin=Pb, mids=np.array([0.0, 1.0]))
    return dict(prior=pri, tables=tabs, K=K, alpha=alpha, sub=sub, prior_kind=prior)
def _bin_index(t, v):
    if t["edges"] is None: return v.astype(int)
    return np.clip(np.searchsorted(t["edges"], v, side="right") - 1, 0, t["logdens"].shape[1] - 1)
def bayes_proba(B, X):
    S = np.tile(B["prior"], (len(X), 1))
    for c, t in B["tables"].items():
        v = np.asarray(X[c].values, dtype=float); m = ~np.isnan(v)
        if m.any(): S[m] += t["logdens"][:, _bin_index(t, v[m])].T
    return _softmax(S)

# ---- one predictor for every model type ----
def member_proba(bundle, n, F):
    if n == "bayes": return bayes_proba(bundle["bayes"], F)
    return bundle["members"][n].predict_proba(member_input(n, F))
def model_predict_proba(bundle, F):
    F = F[bundle["feature_cols"]]
    if bundle["kind"] == "bayes": p = bayes_proba(bundle["bayes"], F)
    else:
        Z = np.hstack([np.log(np.clip(member_proba(bundle, n, F), 1e-6, 1.0)) for n in bundle["member_names"]])
        p = bundle["meta"].predict_proba(Z)
    return _softmax(np.log(np.clip(p, 1e-9, 1.0)) / bundle["temperature"])
def raw_predict_proba(bundle, raw_array):
    """Model as a function of the 14 REAL inputs (used for explanations and test recommendations)."""
    return model_predict_proba(bundle, engineer(pd.DataFrame(np.asarray(raw_array, dtype=float), columns=BASE_COLS)))

# ---- exact explanation for the Bayes model: likelihood ratio of every measured test ----
def bayes_evidence(B, X1, k, r):
    """For ONE patient: how much each measured test favours disease k over disease r.
    log P(k|x)/P(r|x) = prior term + sum of these values (exact, additive)."""
    out = []
    for c in TESTS:
        v = X1[c].iloc[0]
        if pd.isna(v): continue
        t = B["tables"][c]; i = int(_bin_index(t, np.array([float(v)]))[0])
        out.append((c, float(t["logdens"][k, i] - t["logdens"][r, i])))
    return out

# ---- next best test: which missing test would most likely turn this into a confident answer ----
def recommend_tests(bundle, X1, top=3):
    """X1: one-row numeric DataFrame (BASE_COLS). For each missing test, every possible result is weighed by how
    likely it is for THIS patient (sum over diseases of P(disease|current tests) x P(result|disease)), the model
    is re-run with that result, and we average the new confidence."""
    B = bundle["bayes"]; tau = bundle.get("confidence_threshold", 0.7)
    p0 = raw_predict_proba(bundle, X1[BASE_COLS].values)[0]; rows, meta = [], []
    for c in TESTS:
        if not pd.isna(X1[c].iloc[0]): continue
        t = B["tables"][c]; w = p0 @ t["pbin"]; keep = w > 1e-4
        vals, w = t["mids"][keep], w[keep] / w[keep].sum()
        for v, wi in zip(vals, w):
            r = X1[BASE_COLS].values[0].astype(float).copy(); r[BASE_COLS.index(c)] = v; rows.append(r); meta.append((c, wi))
    if not rows: return []
    conf = raw_predict_proba(bundle, np.array(rows)).max(1); agg = {}
    for (c, wi), cf in zip(meta, conf):
        a = agg.setdefault(c, [0.0, 0.0]); a[0] += wi * cf; a[1] += wi * float(cf >= tau)
    res = [dict(test=c, current_confidence=float(p0.max()), expected_confidence_after=float(a[0]),
                chance_of_confident_answer=float(a[1])) for c, a in agg.items()]
    res.sort(key=lambda d: (-d["chance_of_confident_answer"], -d["expected_confidence_after"]))
    return res if top is None else res[:top]
'''
exec(SHARED_CODE)


# %% [markdown]
# ## Cell 4 – Load and validate the dataset
# Finds the Excel file, checks that all 15 required columns exist, removes duplicate rows, encodes every value and reports any value it could not read. Saves class counts and missing-value rates.

# %%
# ==============================================================================================
# CELL 04 – LOAD AND VALIDATE THE DATASET
# What it does: Finds the Excel file, checks that all 15 required columns exist, removes duplicate
# rows, encodes every value and reports any value it could not read. Saves class counts and
# missing-value rates.
# Output: Number of patients and classes, duplicates removed, unreadable values (0 for this
# dataset).
# ==============================================================================================
def find_dataset():
    p = os.environ.get("DATASET_PATH")
    if p and os.path.exists(p): return p
    hits = []
    for pat in ["/kaggle/input/**/*.xlsx", "/kaggle/input/**/*.xls", "/kaggle/input/**/*.csv", "./*.xlsx", "./*.csv"]:
        hits += glob.glob(pat, recursive=True)
    if not hits: raise FileNotFoundError("No dataset found. In Kaggle click 'Add Data' and attach your Excel file.")
    hits.sort(key=lambda h: ("rheumatic" not in h.lower(), h))
    return hits[0]

DATA_PATH = find_dataset(); log(f"Pipeline {SCRIPT_VERSION} | dataset: {DATA_PATH} | FAST_MODE={FAST_MODE} | CPU cores {N_JOBS}")
df = pd.read_csv(DATA_PATH) if DATA_PATH.lower().endswith(".csv") else pd.read_excel(DATA_PATH)
df.columns = [str(c).strip() for c in df.columns]
need = BASE_COLS + ["Disease"]
lower = {c.lower(): c for c in df.columns}
missing_cols = [c for c in need if c.lower() not in lower]
if missing_cols: raise ValueError(f"Dataset is missing columns: {missing_cols}. Found: {list(df.columns)}")
df = df.rename(columns={lower[c.lower()]: c for c in need})[need]
df["Disease"] = df["Disease"].astype(str).str.strip()
n_dups = int(df.duplicated().sum()); df = df.drop_duplicates().reset_index(drop=True)
class_names = sorted(df["Disease"].unique()); K = len(class_names)
y_all = df["Disease"].map({c: i for i, c in enumerate(class_names)}).values
SHORT = {"Ankylosing Spondylitis": "AS", "Normal": "Normal", "Psoriatic Arthritis": "PsA", "Reactive Arthritis": "ReA",
         "Rheumatoid Arthritis": "RA", "Sjögren's Syndrome": "Sjogren", "Systemic Lupus Erythematosus": "SLE"}
short = [SHORT.get(c, c) for c in class_names]
X_all, parse_issues = encode_raw(df, return_issues=True)
flat_issues = [m for row in parse_issues for m in row]
pd.Series(flat_issues, dtype=str).value_counts().rename("count").to_csv(f"{OUT}/paper_results/data_audit_unparsed_values.csv")
if flat_issues: log(f"WARNING: {len(flat_issues)} values could not be read (treated as 'not tested'):\n" + pd.Series(flat_issues).value_counts().head(10).to_string())
else: log("Input check: every non-blank value in the file was understood (0 unreadable values)")
log(f"{len(df)} patients, {K} classes, duplicates removed: {n_dups}")
(df.groupby("Disease")[BASE_COLS].apply(lambda g: g.isna().mean().round(3)).to_csv(f"{OUT}/paper_results/data_audit_missingness_by_class.csv"))
pd.Series(df["Disease"].value_counts()).to_csv(f"{OUT}/paper_results/data_audit_class_counts.csv")


# %% [markdown]
# ## Cell 5 – Stratified train/test split and data audit
# Locks away 15% of patients (1,813) as the final test set, keeping each disease's share equal in both parts. On training patients only, it measures the value range of every lab test per disease, the share of positive antibody results, and the correlations between tests within each disease.

# %%
# ==============================================================================================
# CELL 05 – STRATIFIED TRAIN/TEST SPLIT AND DATA AUDIT
# What it does: Locks away 15% of patients (1,813) as the final test set, keeping each disease's
# share equal in both parts. On training patients only, it measures the value range of every lab
# test per disease, the share of positive antibody results, and the correlations between tests
# within each disease.
# Output: Split sizes, the per-disease lab ranges table and the independence check.
# ==============================================================================================
Xtr_b, Xte_b, ytr, yte = train_test_split(X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=SEED)
log(f"Frozen split: {len(Xtr_b)} train / {len(Xte_b)} test (test is untouched until the final exam)")
skf_tune = StratifiedKFold(3, shuffle=True, random_state=SEED)
skf_oof = StratifiedKFold(3 if FAST_MODE else 5, shuffle=True, random_state=SEED + 1)

# data audits (TRAINING patients only)
prof_rows = {}
for k, c in enumerate(class_names):
    sub = Xtr_b[ytr == k]
    prof_rows[c] = {**{l: float(sub[l].median()) for l in LABS}, **{s: float(sub[s].mean()) for s in SEROLOGY}}
prof_df = pd.DataFrame(prof_rows).T; prof_df.round(3).to_csv(f"{OUT}/paper_results/data_audit_marker_profiles_train.csv")
rng_rows = {c: {l: f"{Xtr_b[ytr == k][l].min():g}-{Xtr_b[ytr == k][l].max():g}" for l in LABS} for k, c in enumerate(class_names)}
pd.DataFrame(rng_rows).T.to_csv(f"{OUT}/paper_results/data_audit_lab_ranges_by_class_train.csv")
ind_rows = []
for k, c in enumerate(class_names):
    cm_ = Xtr_b[ytr == k][["Age"] + TESTS].corr(method="spearman").values.copy(); np.fill_diagonal(cm_, 0)
    ind_rows.append(dict(disease=c, max_abs_within_disease_correlation=float(np.nanmax(np.abs(cm_)))))
mm_ = Xtr_b[TESTS].isna().astype(float).corr().values.copy(); np.fill_diagonal(mm_, 0)
ind_df = pd.DataFrame(ind_rows); ind_df.round(4).to_csv(f"{OUT}/paper_results/data_audit_independence.csv", index=False)
log("Data pattern check (training patients): largest correlation between any two markers WITHIN a disease = "
    f"{ind_df.max_abs_within_disease_correlation.max():.3f}; between 'missing' flags = {np.nanmax(np.abs(mm_)):.3f}\n"
    "   -> markers behave as independent evidence; lab ranges per disease:\n" + pd.DataFrame(rng_rows).T.to_string())


# %% [markdown]
# ## Cell 6 – Feature-set comparison
# Compares four input sets for the tree models (raw 14 inputs; without age and sex; with 15 engineered features; both) using 3-fold cross-validation on training data and keeps the best by log-loss.

# %%
# ==============================================================================================
# CELL 06 – FEATURE-SET COMPARISON
# What it does: Compares four input sets for the tree models (raw 14 inputs; without age and sex;
# with 15 engineered features; both) using 3-fold cross-validation on training data and keeps the
# best by log-loss.
# Output: Table of macro-F1 and log-loss per feature set; the raw 14 inputs are chosen.
# ==============================================================================================
Ftr_all, Fte_all = engineer(Xtr_b), engineer(Xte_b)
eng_cols = list(Ftr_all.columns); no_demo = [c for c in eng_cols if c not in ("Age", "Gender")]
FEATURE_SETS = {"A_raw_14": BASE_COLS, "B_raw_no_age_gender": [c for c in BASE_COLS if c not in ("Age", "Gender")],
                "C_engineered_all": eng_cols, "D_engineered_no_age_gender": no_demo}
rows = []
for nm, cols in FEATURE_SETS.items():
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.06, num_leaves=24, class_weight="balanced",
                           random_state=SEED, n_jobs=N_JOBS, verbosity=-1, deterministic=True, force_col_wise=True)
    p = cross_val_predict(m, Ftr_all[cols], ytr, cv=skf_tune, method="predict_proba")
    rows.append(dict(feature_set=nm, n_features=len(cols), macro_f1=f1_score(ytr, p.argmax(1), average="macro"), log_loss=log_loss(ytr, p)))
fs_table = pd.DataFrame(rows).sort_values("log_loss"); fs_table.to_csv(f"{OUT}/paper_results/ablation_feature_sets.csv", index=False)
BEST_FS = fs_table.iloc[0]["feature_set"]; FEATS = FEATURE_SETS[BEST_FS]
log(f"Tree feature sets compared ({skf_tune.get_n_splits()}-fold CV, training data only):\n{fs_table.round(4).to_string(index=False)}\n   -> trees use {BEST_FS}")
Ftr, Fte = Ftr_all[FEATS], Fte_all[FEATS]


# %% [markdown]
# ## Cell 7 – Tree models and automatic hyperparameter tuning
# Defines Random Forest, LightGBM, XGBoost and CatBoost and tunes each with Optuna (TPE search) on 3-fold cross-validation. The objective rewards both correct answers and honest confidence (macro-F1 minus 0.25 x log-loss). After each round it builds the stacking ensemble from out-of-fold predictions and stops when gains plateau.

# %%
# ==============================================================================================
# CELL 07 – TREE MODELS AND AUTOMATIC HYPERPARAMETER TUNING
# What it does: Defines Random Forest, LightGBM, XGBoost and CatBoost and tunes each with Optuna
# (TPE search) on 3-fold cross-validation. The objective rewards both correct answers and honest
# confidence (macro-F1 minus 0.25 x log-loss). After each round it builds the stacking ensemble
# from out-of-fold predictions and stops when gains plateau.
# Output: Best tuning score per model per round, and the ensemble's cross-validated macro-F1.
# ==============================================================================================
MEMBERS = ["rf", "lgb", "xgb", "cat"]
NICE = dict(rf="Random Forest", lgb="LightGBM", xgb="XGBoost", cat="CatBoost")
def make_model(n, p):
    if n == "rf":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                      max_features=p["max_features"], class_weight="balanced_subsample" if p["balanced"] else None,
                                      n_jobs=N_JOBS, random_state=SEED)
    if n == "lgb":
        return lgb.LGBMClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                                  min_child_samples=p["min_child_samples"], subsample=p["subsample"], subsample_freq=1,
                                  colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"], reg_alpha=p["reg_alpha"],
                                  class_weight="balanced" if p["balanced"] else None, random_state=SEED, n_jobs=N_JOBS, verbosity=-1,
                                  deterministic=True, force_col_wise=True)
    if n == "xgb":
        return xgb.XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                                 subsample=p["subsample"], colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
                                 reg_lambda=p["reg_lambda"], tree_method="hist", objective="multi:softprob", eval_metric="mlogloss",
                                 n_jobs=N_JOBS, random_state=SEED, verbosity=0)
    return CatBoostClassifier(iterations=p["iterations"], depth=p["depth"], learning_rate=p["learning_rate"], l2_leaf_reg=p["l2_leaf_reg"],
                              auto_class_weights="Balanced" if p["balanced"] else None, loss_function="MultiClass", verbose=0,
                              random_seed=SEED, thread_count=N_JOBS, allow_writing_files=False)
def fit_model(n, p, F, y):
    m = make_model(n, p)
    if n == "xgb" and p["balanced"]: m.fit(member_input(n, F), y, sample_weight=compute_sample_weight("balanced", y))
    else: m.fit(member_input(n, F), y)
    return m
def suggest(n, t):
    if n == "rf":
        return dict(n_estimators=t.suggest_int("n_estimators", 200, 500, step=100), max_depth=t.suggest_int("max_depth", 6, 24),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 8), max_features=t.suggest_float("max_features", 0.3, 0.9),
                    balanced=t.suggest_categorical("balanced", [True, False]))
    if n == "lgb":
        return dict(n_estimators=t.suggest_int("n_estimators", 150, 600), learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                    num_leaves=t.suggest_int("num_leaves", 8, 64), min_child_samples=t.suggest_int("min_child_samples", 5, 60),
                    subsample=t.suggest_float("subsample", 0.6, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True), reg_alpha=t.suggest_float("reg_alpha", 1e-3, 5, log=True),
                    balanced=t.suggest_categorical("balanced", [True, False]))
    if n == "xgb":
        return dict(n_estimators=t.suggest_int("n_estimators", 150, 500), max_depth=t.suggest_int("max_depth", 3, 9),
                    learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True), subsample=t.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0), min_child_weight=t.suggest_int("min_child_weight", 1, 10),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-2, 10, log=True), balanced=t.suggest_categorical("balanced", [True, False]))
    return dict(iterations=t.suggest_int("iterations", 150, 400), depth=t.suggest_int("depth", 4, 8),
                learning_rate=t.suggest_float("learning_rate", 0.03, 0.2, log=True), l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 10),
                balanced=t.suggest_categorical("balanced", [True, False]))
def objective_for(n):
    def obj(trial):
        p = suggest(n, trial); sc = []
        for a, b in skf_tune.split(Ftr, ytr):
            m = fit_model(n, p, Ftr.iloc[a], ytr[a]); pr = m.predict_proba(member_input(n, Ftr.iloc[b]))
            sc.append(f1_score(ytr[b], pr.argmax(1), average="macro") - 0.25 * log_loss(ytr[b], pr, labels=list(range(K))))
        return float(np.mean(sc))
    return obj
def oof_trees(n, p, F, y, cv):
    oof = np.zeros((len(y), K))
    for a, b in cv.split(F, y):
        oof[b] = fit_model(n, p, F.iloc[a], y[a]).predict_proba(member_input(n, F.iloc[b]))
    return oof
def macro_f1(y, p): return f1_score(y, p.argmax(1), average="macro")
def stack_Z(probs, names): return np.hstack([np.log(np.clip(probs[n], 1e-6, 1)) for n in names])
def build_stack(oofs, names, y, C_grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    Z = stack_Z(oofs, names); best = None
    for C in C_grid:
        pr = cross_val_predict(LogisticRegression(C=C, max_iter=3000), Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 2), method="predict_proba", n_jobs=N_JOBS)
        ll = log_loss(y, pr, labels=list(range(K)))
        if best is None or ll < best[0]: best = (ll, C, pr)
    return Z, best[1], best[2]

studies = {n: optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)) for n in MEMBERS}
state, prev = None, -1
log("Tree tuning loop starts (fixed trial counts per round)")
for rnd in range(1, MAX_ROUNDS + 1):
    for n in MEMBERS:
        before = len(studies[n].trials)
        studies[n].optimize(objective_for(n), n_trials=TRIALS_PER_ROUND[n], timeout=TUNE_TIMEOUT_CAP)
        if len(studies[n].trials) - before < TRIALS_PER_ROUND[n]: log(f"  NOTE: {n} hit the safety time cap")
        log(f"  round {rnd} | {n:3s} | best tuning score {studies[n].best_value:.4f} | trials so far {len(studies[n].trials)}")
    best_params = {n: dict(studies[n].best_params) for n in MEMBERS}
    oofs = {n: oof_trees(n, best_params[n], Ftr, ytr, skf_oof) for n in MEMBERS}
    _, _, meta_oof = build_stack(oofs, MEMBERS, ytr); score = macro_f1(ytr, meta_oof)
    log(f"  round {rnd} | tree-stack macro-F1 on training folds = {score:.4f}")
    if state is None or score > state["score"]: state = dict(score=score, params=best_params, oofs=oofs)
    if score >= TARGET_OOF_MACRO_F1: log("  target reached -> stop"); break
    if rnd > 1 and score - prev < 0.002: log("  no further gain (plateau) -> stop"); break
    prev = score
best_params, oofs = state["params"], state["oofs"]
json.dump(best_params, open(f"{OUT}/paper_results/best_hyperparameters.json", "w"), indent=2, default=str)


# %% [markdown]
# ## Cell 8 – Range-aware Bayes model tuning
# Compares five settings of the range-aware Bayes model (bin layout and smoothing) with 3-fold cross-validation on training data, keeps the one with the lowest log-loss, and computes its out-of-fold predictions.

# %%
# ==============================================================================================
# CELL 08 – RANGE-AWARE BAYES MODEL TUNING
# What it does: Compares five settings of the range-aware Bayes model (bin layout and smoothing)
# with 3-fold cross-validation on training data, keeps the one with the lowest log-loss, and
# computes its out-of-fold predictions.
# Output: Table of the five settings; the chosen setting (range edges only, alpha = 0.1).
# ==============================================================================================
SUB10 = {"ESR": 10, "CRP": 3, "RF": 10, "Anti-CCP": 10, "C3": 40, "C4": 10}
BAYES_GRID = [dict(sub=None, alpha=a) for a in (0.03, 0.1, 0.5)] + [dict(sub=SUB10, alpha=a) for a in (0.1, 0.5)]
def bayes_oof(X, y, g, cv, prior="empirical"):
    P = np.zeros((len(y), K))
    for a, b in cv.split(X, y): P[b] = bayes_proba(bayes_fit(X.iloc[a], y[a], K, prior=prior, **g), X.iloc[b])
    return P
def select_bayes(X, y, cv):
    rows_, best = [], None
    for i, g in enumerate(BAYES_GRID):
        P = bayes_oof(X, y, g, cv); ll = log_loss(y, P, labels=list(range(K)))
        rows_.append(dict(setting=i, bins="range edges only" if g["sub"] is None else "range edges + sub-bins", alpha=g["alpha"],
                          accuracy=accuracy_score(y, P.argmax(1)), log_loss=ll))
        if best is None or ll < best[0]: best = (ll, g)
    return best[1], pd.DataFrame(rows_)
BAYES_CFG, bayes_grid_table = select_bayes(Xtr_b, ytr, skf_tune)
bayes_grid_table.round(4).to_csv(f"{OUT}/paper_results/bayes_settings_cv.csv", index=False)
log("Bayes model settings compared (training CV):\n" + bayes_grid_table.round(4).to_string(index=False)
    + f"\n   -> using {'range edges only' if BAYES_CFG['sub'] is None else 'range edges + sub-bins'}, alpha={BAYES_CFG['alpha']}")
oofs["bayes"] = bayes_oof(Xtr_b, ytr, BAYES_CFG, skf_oof)


# %% [markdown]
# ## Cell 9 – Model selection by cross-validation
# Builds the three candidates (tree stacking ensemble, range-aware Bayes model, hybrid stack), calibrates each with temperature scaling on out-of-fold predictions, and ranks them by 5-fold cross-validated log-loss. The test set is not used.

# %%
# ==============================================================================================
# CELL 09 – MODEL SELECTION BY CROSS-VALIDATION
# What it does: Builds the three candidates (tree stacking ensemble, range-aware Bayes model,
# hybrid stack), calibrates each with temperature scaling on out-of-fold predictions, and ranks
# them by 5-fold cross-validated log-loss. The test set is not used.
# Output: Model-selection table with accuracy, macro-F1, balanced accuracy, log-loss, ECE and
# top-2 accuracy; the winner.
# ==============================================================================================
def temp_scale(p, T): return _softmax(np.log(np.clip(p, 1e-9, 1)) / T)
def fit_T(y, p): return minimize_scalar(lambda T: log_loss(y, temp_scale(p, T), labels=list(range(K))), bounds=(0.3, 5.0), method="bounded").x
def ece_score(y, p, bins=15):
    conf, pred = p.max(1), p.argmax(1); ok = (pred == y).astype(float); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)
def top2(y, p): return float((np.argsort(-p, axis=1)[:, :2] == np.asarray(y)[:, None]).any(1).mean())
CAND = {}
Z4, C4, m4 = build_stack(oofs, MEMBERS, ytr)
CAND["Tree stacking ensemble (RF+LGB+XGB+CatBoost)"] = dict(kind="stack", names=MEMBERS, Z=Z4, C=C4, oof=m4)
CAND["Range-aware Bayes model"] = dict(kind="bayes", names=[], oof=oofs["bayes"])
Z5, C5, m5 = build_stack(oofs, MEMBERS + ["bayes"], ytr)
CAND["Hybrid stack (4 tree models + Bayes)"] = dict(kind="stack", names=MEMBERS + ["bayes"], Z=Z5, C=C5, oof=m5)
sel_rows = []
for nm, c in CAND.items():
    c["T"] = fit_T(ytr, c["oof"]); c["oof_cal"] = temp_scale(c["oof"], c["T"]); p = c["oof_cal"]
    sel_rows.append(dict(candidate=nm, cv_accuracy=accuracy_score(ytr, p.argmax(1)), cv_macro_f1=macro_f1(ytr, p),
                         cv_balanced_accuracy=balanced_accuracy_score(ytr, p.argmax(1)), cv_log_loss=log_loss(ytr, p, labels=list(range(K))),
                         cv_ece=ece_score(ytr, p), cv_top2=top2(ytr, p), temperature=c["T"]))
sel_df = pd.DataFrame(sel_rows).sort_values("cv_log_loss"); sel_df.round(4).to_csv(f"{OUT}/paper_results/model_selection_cv.csv", index=False)
SELECTED = sel_df.iloc[0]["candidate"]
log("MODEL SELECTION on training cross-validation (lowest log-loss wins; the test set is NOT used):\n"
    + sel_df.round(4).to_string(index=False) + f"\n   -> CV WINNER: {SELECTED} (deployment is confirmed by the robustness check next)")


# %% [markdown]
# ## Cell 10 – Robustness check and deployment decision
# Disturbs lab values on held-out training folds (random error, analyser shifts, wrong antibody results, extra missing tests) and applies a rule fixed in advance: keep the winner unless another candidate is at least 1.0 accuracy point better under 5% random lab error.

# %%
# ==============================================================================================
# CELL 10 – ROBUSTNESS CHECK AND DEPLOYMENT DECISION
# What it does: Disturbs lab values on held-out training folds (random error, analyser shifts,
# wrong antibody results, extra missing tests) and applies a rule fixed in advance: keep the
# winner unless another candidate is at least 1.0 accuracy point better under 5% random lab error.
# Output: Robustness table and the decision (the range-aware Bayes model is kept).
# ==============================================================================================
# Rule fixed BEFORE the run: deploy the cross-validation winner unless another candidate is at least
# ROBUST_SWITCH_MARGIN more accurate when lab values carry ROBUST_PRIMARY noise. The test set is not used here.
CV_WINNER = SELECTED
PERTURB = ["clean", "noise_5pct", "noise_10pct", "shift_plus_5pct", "shift_minus_5pct", "serology_flip_5pct", "two_more_missing"]
DETERMINISTIC = {"clean", "shift_plus_5pct", "shift_minus_5pct"}
PERTURB_TEXT = {"clean": "no change", "noise_5pct": "random lab error, sd 5%", "noise_10pct": "random lab error, sd 10%",
                "shift_plus_5pct": "all labs read 5% high (different analyser)", "shift_minus_5pct": "all labs read 5% low",
                "serology_flip_5pct": "5% of antibody results wrong", "two_more_missing": "2 extra tests missing per patient"}
def perturb(X, kind, rs):
    """Return a disturbed copy of a raw numeric table (BASE_COLS). Blanks stay blank; lab values never go below 0."""
    X = X.copy()
    if kind == "clean": return X
    if kind in ("noise_5pct", "noise_10pct", "shift_plus_5pct", "shift_minus_5pct"):
        L = X[LABS].values.astype(float)
        if kind == "noise_5pct": L = L * (1 + rs.normal(0, 0.05, L.shape))
        elif kind == "noise_10pct": L = L * (1 + rs.normal(0, 0.10, L.shape))
        elif kind == "shift_plus_5pct": L = L * 1.05
        else: L = L * 0.95
        X[LABS] = np.where(np.isnan(L), np.nan, np.clip(L, 0, None))
    elif kind == "serology_flip_5pct":
        S = X[SEROLOGY].values.astype(float).copy(); m = ~np.isnan(S) & (rs.random(S.shape) < 0.05); S[m] = 1 - S[m]; X[SEROLOGY] = S
    elif kind == "two_more_missing":
        T = X[TESTS].values.astype(float).copy()
        for i in range(len(T)):
            meas = np.where(~np.isnan(T[i]))[0]; k_ = min(2, max(0, len(meas) - MIN_MEASURED_TESTS))
            if k_: T[i, rs.choice(meas, k_, replace=False)] = np.nan
        X[TESTS] = T
    else: raise ValueError(kind)
    return X
log(f"Robustness check on training folds ({skf_oof.get_n_splits()} folds, {ROBUST_REPEATS} repeat(s) per random disturbance) ...")
ROB_MODELS = list(CAND) + ["XGBoost (single model, reference)"]
rob_ok = {nm: {k: np.zeros(len(ytr)) for k in PERTURB} for nm in ROB_MODELS}
for fi, (a, b) in enumerate(skf_oof.split(Xtr_b, ytr)):
    mem = {n: fit_model(n, best_params[n], Ftr.iloc[a], ytr[a]) for n in MEMBERS}
    bay = bayes_fit(Xtr_b.iloc[a], ytr[a], K, **BAYES_CFG)
    # stack meta-learners refit on the clean out-of-fold predictions of the other folds
    metas = {nm: LogisticRegression(C=c["C"], max_iter=3000).fit(stack_Z(oofs, c["names"])[a], ytr[a]) for nm, c in CAND.items() if c["kind"] == "stack"}
    for kind in PERTURB:
        nrep = 1 if kind in DETERMINISTIC else ROBUST_REPEATS
        for r in range(nrep):
            Xv = perturb(Xtr_b.iloc[b], kind, np.random.default_rng(SEED + 100 * fi + r)); Fv = engineer(Xv)[FEATS]
            pm = {n: mem[n].predict_proba(member_input(n, Fv)) for n in MEMBERS}; pm["bayes"] = bayes_proba(bay, Fv)
            for nm, c in CAND.items():
                p = pm["bayes"] if c["kind"] == "bayes" else metas[nm].predict_proba(stack_Z(pm, c["names"]))
                rob_ok[nm][kind][b] += (p.argmax(1) == ytr[b]) / nrep
            rob_ok["XGBoost (single model, reference)"][kind][b] += (pm["xgb"].argmax(1) == ytr[b]) / nrep
    log(f"  fold {fi+1}/{skf_oof.get_n_splits()} done")
rob_tab = pd.DataFrame({nm: {k: float(rob_ok[nm][k].mean()) for k in PERTURB} for nm in ROB_MODELS}).T[PERTURB]
rob_drop = rob_tab.sub(rob_tab["clean"], axis=0)
rob_tab.round(4).to_csv(f"{OUT}/paper_results/robustness_training_cv_accuracy.csv")
rob_drop.round(4).to_csv(f"{OUT}/paper_results/robustness_training_cv_change_vs_clean.csv")
prim = rob_tab[ROBUST_PRIMARY]; alts = prim.drop([CV_WINNER, "XGBoost (single model, reference)"])
best_alt = alts.idxmax() if len(alts) else None
if best_alt is not None and prim[best_alt] >= prim[CV_WINNER] + ROBUST_SWITCH_MARGIN:
    SELECTED = best_alt
    ROBUST_DECISION = (f"SWITCHED: under '{ROBUST_PRIMARY}' the CV winner ({CV_WINNER}) scored {prim[CV_WINNER]*100:.2f}% but "
                       f"{best_alt} scored {prim[best_alt]*100:.2f}% (>= {ROBUST_SWITCH_MARGIN*100:.1f} points better) -> deploying {best_alt}")
else:
    ROBUST_DECISION = (f"KEPT: under '{ROBUST_PRIMARY}' the CV winner ({CV_WINNER}) scored {prim[CV_WINNER]*100:.2f}%"
                       + (f" vs best alternative {best_alt} {prim[best_alt]*100:.2f}%" if best_alt is not None else "")
                       + f" (switch needs >= {ROBUST_SWITCH_MARGIN*100:.1f} points) -> deploying {CV_WINNER}")
log("Robustness on held-out training folds (accuracy %):\n" + (rob_tab * 100).round(2).to_string() + f"\n   {ROBUST_DECISION}")
ROB_MD = f"""
## Robustness check (held-out training folds; this decides which model is deployed)
Accuracy (%) when inputs are disturbed. Disturbances: {"; ".join(f"{k} = {v}" for k, v in PERTURB_TEXT.items())}.
Rule fixed before the run: deploy the CV winner unless another candidate is at least {ROBUST_SWITCH_MARGIN*100:.1f} points more accurate under '{ROBUST_PRIMARY}'.

{(rob_tab * 100).round(2).to_markdown()}

Decision: {ROBUST_DECISION}
"""


# %% [markdown]
# ## Cell 11 – Final training on the full training set
# Retrains all candidates on all 10,272 training patients and packages the selected model into one bundle. Checks that the bundle reproduces the predictions exactly.

# %%
# ==============================================================================================
# CELL 11 – FINAL TRAINING ON THE FULL TRAINING SET
# What it does: Retrains all candidates on all 10,272 training patients and packages the selected
# model into one bundle. Checks that the bundle reproduces the predictions exactly.
# Output: A short progress message.
# ==============================================================================================
log("Refitting all candidates on the full training set ...")
assert all(c in FEATS for c in TESTS)
final_members = {n: fit_model(n, best_params[n], Ftr, ytr) for n in MEMBERS}
bayes_final = bayes_fit(Xtr_b, ytr, K, **BAYES_CFG)
for nm, c in CAND.items():
    c["meta"] = LogisticRegression(C=c["C"], max_iter=3000).fit(c["Z"], ytr) if c["kind"] == "stack" else None
COMMON = dict(version=SCRIPT_VERSION, classes=class_names, short_names=short, feature_set=BEST_FS, base_cols=BASE_COLS, feature_cols=FEATS,
              members=final_members, bayes=bayes_final, seed=SEED, min_measured_tests=MIN_MEASURED_TESTS,
              shap_permutations=SHAP_PERMUTATIONS, input_ranges={c: [float(Xtr_b[c].min()), float(Xtr_b[c].max())] for c in ["Age"] + LABS})
def cand_bundle(nm):
    c = CAND[nm]; return {**COMMON, "kind": c["kind"], "member_names": c["names"], "meta": c["meta"], "temperature": float(c["T"]), "model_name": nm}
P_cand = {nm: model_predict_proba(cand_bundle(nm), Fte) for nm in CAND}
P_members = {n: final_members[n].predict_proba(member_input(n, Fte)) for n in MEMBERS}
bundle = cand_bundle(SELECTED)
if bundle["kind"] == "bayes": bundle["members"] = {}      # the Bayes model needs no tree libraries -> light package
P_final = P_cand[SELECTED]
assert np.allclose(P_final, model_predict_proba(bundle, Fte), atol=1e-8)
assert np.allclose(P_final, raw_predict_proba(bundle, Xte_b.values), atol=1e-8)


# %% [markdown]
# ## Cell 12 – Test-set evaluation and model comparison
# Scores the 1,813 frozen test patients once: all metrics with bootstrap 95% confidence intervals, per-disease results, the confusion matrix, the most common mistakes, a comparison with 10 other models and McNemar significance tests.

# %%
# ==============================================================================================
# CELL 12 – TEST-SET EVALUATION AND MODEL COMPARISON
# What it does: Scores the 1,813 frozen test patients once: all metrics with bootstrap 95%
# confidence intervals, per-disease results, the confusion matrix, the most common mistakes, a
# comparison with 10 other models and McNemar significance tests.
# Output: Final test metrics, mistakes table, model-comparison table and McNemar tests.
# ==============================================================================================
def all_metrics(y, p):
    p = p / p.sum(1, keepdims=True); pr = p.argmax(1)
    return dict(accuracy=accuracy_score(y, pr), macro_f1=f1_score(y, pr, average="macro"), weighted_f1=f1_score(y, pr, average="weighted"),
                balanced_accuracy=balanced_accuracy_score(y, pr), mcc=matthews_corrcoef(y, pr), kappa=cohen_kappa_score(y, pr),
                auc_ovr=roc_auc_score(y, p, multi_class="ovr", average="macro"), log_loss=log_loss(y, p, labels=list(range(K))),
                brier=float(np.mean(np.sum((p - np.eye(K)[y]) ** 2, axis=1))), ece=ece_score(y, p), top2_accuracy=top2(y, p))
main = all_metrics(yte, P_final)
log(f"FINAL TEST RESULTS - {SELECTED} ({len(yte)} untouched patients): " + ", ".join(f"{k}={v:.4f}" for k, v in main.items()))
rng = np.random.default_rng(SEED); boots = {k: [] for k in ["accuracy", "macro_f1", "balanced_accuracy", "auc_ovr", "top2_accuracy"]}
for _ in range(N_BOOT):
    i = rng.integers(0, len(yte), len(yte)); yb, pb = yte[i], P_final[i]
    if len(np.unique(yb)) < K: continue
    boots["accuracy"].append(accuracy_score(yb, pb.argmax(1))); boots["macro_f1"].append(f1_score(yb, pb.argmax(1), average="macro"))
    boots["balanced_accuracy"].append(balanced_accuracy_score(yb, pb.argmax(1)))
    boots["auc_ovr"].append(roc_auc_score(yb, pb / pb.sum(1, keepdims=True), multi_class="ovr")); boots["top2_accuracy"].append(top2(yb, pb))
ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boots.items()}
pr_, rc_, f1_, sup_ = precision_recall_fscore_support(yte, P_final.argmax(1), labels=range(K), zero_division=0)
cm = confusion_matrix(yte, P_final.argmax(1), labels=range(K))
per_class = pd.DataFrame(dict(disease=class_names, precision=pr_, recall=rc_, f1=f1_, support=sup_,
                              auc=[roc_auc_score((yte == k).astype(int), P_final[:, k]) for k in range(K)],
                              specificity=[(cm.sum() - cm[k].sum() - cm[:, k].sum() + cm[k, k]) / (cm.sum() - cm[k].sum()) for k in range(K)]))
per_class.round(4).to_csv(f"{OUT}/paper_results/per_class_metrics.csv", index=False)
sle_f1 = float(per_class.loc[per_class.disease == "Systemic Lupus Erythematosus", "f1"].iloc[0]) if "Systemic Lupus Erythematosus" in class_names else np.nan
conf_rows = [dict(true=short[i], predicted=short[j], patients=int(cm[i, j]), share_of_true_class=round(cm[i, j] / cm[i].sum(), 4))
             for i in range(K) for j in range(K) if i != j and cm[i, j] > 0]
top_conf = pd.DataFrame(conf_rows).sort_values("patients", ascending=False); top_conf.to_csv(f"{OUT}/paper_results/top_confusions.csv", index=False)
log("Most common mistakes:\n" + top_conf.head(8).to_string(index=False))

comp = {}
lr_pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
lr_pipe.fit(Ftr, ytr); comp["Logistic Regression (baseline)"] = lr_pipe.predict_proba(Fte)
med = make_pipeline(SimpleImputer(strategy="median"), make_model("lgb", best_params["lgb"])); med.fit(Ftr, ytr)
comp["LightGBM + median imputation (previous approach)"] = med.predict_proba(Fte)
for n in MEMBERS: comp[NICE[n]] = P_members[n]
comp["Soft-voting average (4 tree models)"] = np.mean([P_members[n] for n in MEMBERS], axis=0)
comp["Range-aware Bayes, equal disease weights"] = bayes_proba(bayes_fit(Xtr_b, ytr, K, prior="balanced", **BAYES_CFG), Fte)
for nm in CAND: comp[("PROPOSED: " if nm == SELECTED else "") + nm] = P_cand[nm]
PROP = "PROPOSED: " + SELECTED
comp_df = pd.DataFrame({k: all_metrics(yte, v) for k, v in comp.items()}).T
comp_df.round(4).to_csv(f"{OUT}/paper_results/model_comparison.csv")
log("Model comparison (same untouched test set):\n" + comp_df[["accuracy", "macro_f1", "balanced_accuracy", "log_loss", "ece", "top2_accuracy"]].round(4).to_string())
ok_final = P_final.argmax(1) == yte; sig_rows = []
for nm, P in comp.items():
    if nm == PROP: continue
    ok_o = P.argmax(1) == yte; b_ = int((ok_final & ~ok_o).sum()); c_ = int((~ok_final & ok_o).sum())
    pval = binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ > 0 else 1.0
    sig_rows.append(dict(compared_with=nm, only_proposed_right=b_, only_other_right=c_, net_gain_patients=b_ - c_,
                         p_value=round(float(pval), 4), significant_at_0_05=bool(pval < 0.05)))
sig_df = pd.DataFrame(sig_rows); sig_df.to_csv(f"{OUT}/paper_results/significance_mcnemar.csv", index=False)
log("McNemar tests (proposed vs. each alternative, same test patients):\n" + sig_df.to_string(index=False))
P_ens = P_cand["Tree stacking ensemble (RF+LGB+XGB+CatBoost)"]
P_final_raw = temp_scale(P_final, 1.0 / bundle["temperature"])     # undo the temperature step exactly
cal_df = pd.DataFrame({"before_temperature": all_metrics(yte, P_final_raw),
                       "after_temperature": all_metrics(yte, P_final)}).T[["ece", "brier", "log_loss"]]
cal_df.round(4).to_csv(f"{OUT}/paper_results/calibration_before_after.csv")

N_TESTS = len(TESTS); n_missing = Xte_b[TESTS].isna().sum(axis=1).values
bins = [(0, 1, "0-1 missing"), (2, 3, "2-3 missing"), (4, 5, "4-5 missing"), (6, 99, "6+ missing")]
comp_rows = [dict(group=lab, patients=int(((n_missing >= a) & (n_missing <= b)).sum()),
                  accuracy=float((P_final.argmax(1)[(n_missing >= a) & (n_missing <= b)] == yte[(n_missing >= a) & (n_missing <= b)]).mean()))
             for a, b, lab in bins if ((n_missing >= a) & (n_missing <= b)).any()]
pd.DataFrame(comp_rows).round(4).to_csv(f"{OUT}/paper_results/accuracy_by_missing_tests.csv", index=False)


# %% [markdown]
# ## Cell 13 – Safe mode (confidence gating)
# Chooses, on training out-of-fold predictions, the confidence threshold above which answered patients reach 95% accuracy, then measures coverage and accuracy on the test set. Also records accuracy by number of missing tests.

# %%
# ==============================================================================================
# CELL 13 – SAFE MODE (CONFIDENCE GATING)
# What it does: Chooses, on training out-of-fold predictions, the confidence threshold above which
# answered patients reach 95% accuracy, then measures coverage and accuracy on the test set. Also
# records accuracy by number of missing tests.
# Output: Threshold 0.667: share of patients answered and their accuracy.
# ==============================================================================================
# safe mode (threshold chosen on TRAINING folds only)
def selective(y, p):
    conf = p.max(1); o = np.argsort(-conf); ok = (p.argmax(1)[o] == y[o]); n = np.arange(1, len(y) + 1)
    return conf[o], n / len(y), np.cumsum(ok) / n
def pick_tau(y, p, target):
    conf, cov, acc = selective(y, p); idx = np.where((acc >= target) & (cov >= 0.05))[0]
    return float(conf[idx.max()]) if len(idx) else float(conf[0])
oof_sel = CAND[SELECTED]["oof_cal"]
tau = pick_tau(ytr, oof_sel, SELECTIVE_TARGET_ACC); bundle["confidence_threshold"] = tau
keep = P_final.max(1) >= tau
sel_acc = float((P_final.argmax(1)[keep] == yte[keep]).mean()) if keep.any() else float("nan"); sel_cov = float(keep.mean())
flag_acc = float((P_final.argmax(1)[~keep] == yte[~keep]).mean()) if (~keep).any() else float("nan")
sel_tbl = []
for tgt in [0.90, 0.95, 0.97]:
    t_ = pick_tau(ytr, oof_sel, tgt); kk = P_final.max(1) >= t_
    sel_tbl.append(dict(target_accuracy=tgt, confidence_threshold=t_, test_coverage=float(kk.mean()),
                        test_accuracy=float((P_final.argmax(1)[kk] == yte[kk]).mean()) if kk.any() else np.nan,
                        flagged_patients_accuracy=float((P_final.argmax(1)[~kk] == yte[~kk]).mean()) if (~kk).any() else np.nan))
pd.DataFrame(sel_tbl).round(4).to_csv(f"{OUT}/paper_results/confidence_gated_performance.csv", index=False)
log(f"Safe mode: confidence >= {tau:.3f} -> {sel_cov*100:.1f}% answered at {sel_acc*100:.2f}% accuracy; flagged would have been {flag_acc*100:.1f}% accurate")


# %% [markdown]
# ## Cell 14 – Next-best-test analysis
# For every flagged test patient with missing tests, ranks the missing tests by the expected chance of reaching a confident answer and compares the recommended test with a random missing test.

# %%
# ==============================================================================================
# CELL 14 – NEXT-BEST-TEST ANALYSIS
# What it does: For every flagged test patient with missing tests, ranks the missing tests by the
# expected chance of reaching a confident answer and compares the recommended test with a random
# missing test.
# Output: Chance of a confident answer after the recommended vs a random test.
# ==============================================================================================
# next best test for flagged patients (model-based expectation; the real missing values are unknown)
nbt_rows = []
for i in np.where(~keep)[0]:
    X1 = Xte_b.iloc[[i]]; recs = recommend_tests(bundle, X1, top=None)
    if not recs: continue
    nbt_rows.append(dict(current=recs[0]["current_confidence"], best_test=recs[0]["test"],
                         best_expected_conf=recs[0]["expected_confidence_after"], best_chance_confident=recs[0]["chance_of_confident_answer"],
                         random_expected_conf=float(np.mean([r["expected_confidence_after"] for r in recs])),
                         random_chance_confident=float(np.mean([r["chance_of_confident_answer"] for r in recs]))))
nbt = pd.DataFrame(nbt_rows)
if len(nbt):
    nbt_sum = pd.DataFrame([dict(flagged_patients_with_missing_tests=len(nbt), mean_current_confidence=nbt.current.mean(),
                                 expected_confidence_after_recommended_test=nbt.best_expected_conf.mean(),
                                 expected_confidence_after_random_missing_test=nbt.random_expected_conf.mean(),
                                 chance_confident_after_recommended_test=nbt.best_chance_confident.mean(),
                                 chance_confident_after_random_missing_test=nbt.random_chance_confident.mean())])
    nbt_sum.round(4).to_csv(f"{OUT}/paper_results/next_best_test_summary.csv", index=False)
    nbt.best_test.value_counts().rename("times_recommended").to_csv(f"{OUT}/paper_results/next_best_test_most_recommended.csv")
    log(f"Next-best-test ({len(nbt)} flagged patients with missing tests): chance of a confident answer after the RECOMMENDED test "
        f"{nbt.best_chance_confident.mean()*100:.1f}% vs a RANDOM missing test {nbt.random_chance_confident.mean()*100:.1f}%")


# %% [markdown]
# ## Cell 15 – Repeated-split validation
# Repeats training and testing on 10 new random stratified splits to measure stability (tree settings fixed; Bayes settings re-chosen per split).

# %%
# ==============================================================================================
# CELL 15 – REPEATED-SPLIT VALIDATION
# What it does: Repeats training and testing on 10 new random stratified splits to measure
# stability (tree settings fixed; Bayes settings re-chosen per split).
# Output: Accuracy of each model on every split and the mean +/- standard deviation.
# ==============================================================================================
log(f"Repeated-split check: {N_REPEATS} new random train/test splits (tree settings fixed from the main run; Bayes settings re-chosen per split)")
def per_class_scores(y, p):
    _, rc, f1c, _ = precision_recall_fscore_support(y, p.argmax(1), labels=range(K), zero_division=0)
    return {**{f"recall_{short[k]}": float(rc[k]) for k in range(K)}, **{f"f1_{short[k]}": float(f1c[k]) for k in range(K)}}
def one_split(seed):
    Xa, Xb, ya, yb = train_test_split(X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=seed)
    Fa, Fb = engineer(Xa)[FEATS], engineer(Xb)[FEATS]; cv = StratifiedKFold(3 if FAST_MODE else 5, shuffle=True, random_state=seed)
    g, _ = select_bayes(Xa, ya, StratifiedKFold(3, shuffle=True, random_state=seed))
    oo = {n: oof_trees(n, best_params[n], Fa, ya, cv) for n in MEMBERS}; oo["bayes"] = bayes_oof(Xa, ya, g, cv)
    pm = {n: fit_model(n, best_params[n], Fa, ya).predict_proba(member_input(n, Fb)) for n in MEMBERS}
    pm["bayes"] = bayes_proba(bayes_fit(Xa, ya, K, **g), Fb)
    res = {}
    for nm, c in CAND.items():
        if c["kind"] == "bayes": T = fit_T(ya, oo["bayes"]); res[nm] = temp_scale(pm["bayes"], T); continue
        Za = stack_Z(oo, c["names"]); mo = cross_val_predict(LogisticRegression(C=c["C"], max_iter=3000), Za, ya,
                                                            cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba", n_jobs=N_JOBS)
        T = fit_T(ya, mo); meta_ = LogisticRegression(C=c["C"], max_iter=3000).fit(Za, ya); res[nm] = temp_scale(meta_.predict_proba(stack_Z(pm, c["names"])), T)
    res["XGBoost"] = pm["xgb"]
    return {nm: {**all_metrics(yb, P), **per_class_scores(yb, P)} for nm, P in res.items()}
rep_rows = []
for r in range(N_REPEATS):
    s = 1000 + r; out = one_split(s)
    for nm, m in out.items(): rep_rows.append(dict(split_seed=s, model=nm, **m))
    log(f"  split {r+1}/{N_REPEATS}: " + " | ".join(f"{nm.split(' (')[0]}: {m['accuracy']*100:.2f}%" for nm, m in out.items()))
rep = pd.DataFrame(rep_rows); rep.round(4).to_csv(f"{OUT}/paper_results/repeated_splits_all.csv", index=False)
rep_sum = rep.groupby("model")[["accuracy", "macro_f1", "balanced_accuracy", "log_loss", "ece", "top2_accuracy"]].agg(["mean", "std"])
rep_sum.columns = [f"{a}_{b}" for a, b in rep_sum.columns]; rep_sum = rep_sum.sort_values("accuracy_mean", ascending=False)
rep_sum.round(4).to_csv(f"{OUT}/paper_results/repeated_splits_summary.csv")
ENS_NAME = "Tree stacking ensemble (RF+LGB+XGB+CatBoost)"
piv = rep.pivot(index="split_seed", columns="model", values="accuracy")
wins = int((piv[SELECTED] > piv[ENS_NAME]).sum()) if SELECTED != ENS_NAME else None
diff = (piv[SELECTED] - piv[ENS_NAME]) if SELECTED != ENS_NAME else None
log("Repeated-split summary (mean +/- sd over splits):\n" + "\n".join(
    f"   {m:48s} acc {rep_sum.loc[m,'accuracy_mean']*100:.2f} +/- {rep_sum.loc[m,'accuracy_std']*100:.2f} | macro-F1 {rep_sum.loc[m,'macro_f1_mean']:.4f} | "
    f"log-loss {rep_sum.loc[m,'log_loss_mean']:.4f} | top-2 {rep_sum.loc[m,'top2_accuracy_mean']*100:.2f}" for m in rep_sum.index)
    + (f"\n   {SELECTED} beat the tree stacking ensemble on {wins}/{N_REPEATS} splits (mean gain {diff.mean()*100:+.2f} points)" if wins is not None else ""))


# %% [markdown]
# ## Cell 16 – Accuracy ceiling estimate
# Simulates 400,000 patients from the learned disease ranges, antibody rates and missing-test rates and measures the accuracy of the best possible rule.

# %%
# ==============================================================================================
# CELL 16 – ACCURACY CEILING ESTIMATE
# What it does: Simulates 400,000 patients from the learned disease ranges, antibody rates and
# missing-test rates and measures the accuracy of the best possible rule.
# Output: The ceiling (about 86.2%) and the share of it our model reaches.
# ==============================================================================================
def simulate(B, n, rs):
    """Draw synthetic patients from the learned pattern: disease -> independent test results -> random missing tests."""
    yk = rs.choice(B["K"], n, p=np.exp(B["prior"])); S = pd.DataFrame(index=range(n))
    for c in LABS:
        t = B["tables"][c]; e = t["edges"]; out = np.empty(n); w_ok = np.ones(len(e) - 1); w_ok[[0, -1]] = 0
        for k in range(B["K"]):
            idx = np.where(yk == k)[0]; pk = t["pbin"][k] * w_ok; pk /= pk.sum(); b = rs.choice(len(pk), len(idx), p=pk)
            out[idx] = rs.uniform(e[b], e[b + 1])
        S[c] = out
    for c in SEROLOGY: S[c] = (rs.random(n) < B["tables"][c]["pbin"][yk, 1]).astype(float)
    for c in TESTS: S.loc[rs.random(n) < Xtr_b[c].isna().mean(), c] = np.nan
    return S, yk
Bsim = bayes_fit(Xtr_b, ytr, K, **BAYES_CFG); Ssim, ysim = simulate(Bsim, CEILING_SIM, np.random.default_rng(SEED))
psim = bayes_proba(Bsim, Ssim).argmax(1)
ceiling = dict(ceiling_accuracy=accuracy_score(ysim, psim), ceiling_macro_f1=f1_score(ysim, psim, average="macro"),
               ceiling_balanced_accuracy=balanced_accuracy_score(ysim, psim), simulated_patients=CEILING_SIM)
pd.DataFrame([ceiling]).round(4).to_csv(f"{OUT}/paper_results/accuracy_ceiling.csv", index=False)
rep_mean_sel = float(rep_sum.loc[SELECTED, "accuracy_mean"])
log(f"Accuracy ceiling (best possible model if the data follows the independent-markers pattern): {ceiling['ceiling_accuracy']*100:.2f}% | "
    f"final model reaches {rep_mean_sel*100:.2f}% on average = {rep_mean_sel/ceiling['ceiling_accuracy']*100:.1f}% of the ceiling")


# %% [markdown]
# ## Cell 17 – Result figures
# Draws the confusion matrix, ROC curves, reliability diagram, per-disease F1, model comparison, confidence-gated accuracy, accuracy by missing tests and repeated-split accuracy.

# %%
# ==============================================================================================
# CELL 17 – RESULT FIGURES
# What it does: Draws the confusion matrix, ROC curves, reliability diagram, per-disease F1, model
# comparison, confidence-gated accuracy, accuracy by missing tests and repeated-split accuracy.
# Output: PNG files in outputs/figures.
# ==============================================================================================
def save(fig, name): fig.tight_layout(); fig.savefig(f"{OUT}/figures/{name}.png", dpi=200); plt.close(fig)
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(8, 6.5)); sns.heatmap(cmn, annot=np.array([[f"{cmn[i,j]:.2f}\n({cm[i,j]})" for j in range(K)] for i in range(K)]),
    fmt="", cmap="Blues", xticklabels=short, yticklabels=short, ax=ax, cbar=False, vmin=0, vmax=1)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (frozen test set)"); save(fig, "confusion_matrix")
fig, ax = plt.subplots(figsize=(7.5, 6))
for k in range(K):
    f_, t_, _ = roc_curve((yte == k).astype(int), P_final[:, k]); ax.plot(f_, t_, lw=2, label=f"{short[k]} (AUC {auc(f_, t_):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("One-vs-rest ROC curves"); save(fig, "roc_curves")
conf_t, ok_t = P_final.max(1), (P_final.argmax(1) == yte).astype(float); edges_ = np.linspace(0, 1, 11); xs, ys = [], []
for lo, hi in zip(edges_[:-1], edges_[1:]):
    m = (conf_t > lo) & (conf_t <= hi)
    if m.sum() >= 5: xs.append(conf_t[m].mean()); ys.append(ok_t[m].mean())
fig, ax = plt.subplots(figsize=(5.5, 5.5)); ax.plot([0, 1], [0, 1], "k--"); ax.plot(xs, ys, "o-", color="#1f77b4")
ax.set_xlabel("Confidence shown by the AI"); ax.set_ylabel("How often it was actually right"); ax.set_title(f"Reliability diagram (ECE = {main['ece']:.3f})"); save(fig, "calibration_curve")
fig, ax = plt.subplots(figsize=(7, 4)); sns.barplot(x=short, y=per_class["f1"].values, ax=ax, color="#4c78a8"); ax.set_ylim(0, 1.05)
for i, v in enumerate(per_class["f1"].values): ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("F1-score"); ax.set_title("F1-score per disease"); save(fig, "per_class_f1")
fig, ax = plt.subplots(figsize=(9, 5.5)); cd = comp_df[["accuracy", "macro_f1"]].copy(); cd.index = [i.replace("PROPOSED: ", "* ") for i in cd.index]
cd.plot.barh(ax=ax); ax.set_xlim(0.7, 0.9); ax.axvline(ceiling["ceiling_accuracy"], color="r", ls="--", lw=1, label="estimated accuracy ceiling")
ax.legend(fontsize=8, loc="lower right"); ax.set_title("Model comparison on the same untouched test set (* = proposed)"); ax.invert_yaxis(); save(fig, "model_comparison")
c_, cov_, acc_ = selective(yte, P_final); fig, ax = plt.subplots(figsize=(6.5, 4.5)); ax.plot(cov_, acc_, lw=2)
ax.axhline(SELECTIVE_TARGET_ACC, color="g", ls="--"); ax.scatter([sel_cov], [sel_acc], color="r", zorder=5, label=f"Safe mode: {sel_cov*100:.0f}% coverage, {sel_acc*100:.1f}% accuracy")
ax.set_xlabel("Share of patients the AI answers (most confident first)"); ax.set_ylabel("Accuracy on those patients"); ax.legend(); ax.set_title("Confidence-gated accuracy"); save(fig, "confidence_gated_accuracy")
if comp_rows:
    cr = pd.DataFrame(comp_rows); fig, ax = plt.subplots(figsize=(6, 4)); sns.barplot(data=cr, x="group", y="accuracy", ax=ax, color="#59a14f"); ax.set_ylim(0.5, 1.02)
    for i, (a_, n_) in enumerate(zip(cr["accuracy"], cr["patients"])): ax.text(i, a_ + 0.01, f"{a_*100:.1f}%\nn={n_}", ha="center", fontsize=8)
    ax.set_xlabel(f"Number of tests missing (of {N_TESTS})"); ax.set_title("Accuracy depends on how complete the work-up is"); save(fig, "accuracy_by_missing_tests")
fig, ax = plt.subplots(figsize=(8, 4.5)); order_m = list(rep_sum.index)
sns.boxplot(data=rep, y="model", x="accuracy", order=order_m, ax=ax, color="#9ecae1"); sns.stripplot(data=rep, y="model", x="accuracy", order=order_m, ax=ax, color="k", size=3)
ax.axvline(ceiling["ceiling_accuracy"], color="r", ls="--", lw=1, label=f"estimated ceiling {ceiling['ceiling_accuracy']*100:.1f}%"); ax.legend(fontsize=8)
ax.set_ylabel(""); ax.set_title(f"Accuracy over {N_REPEATS} different train/test splits"); save(fig, "repeated_splits_accuracy")


# %% [markdown]
# ## Cell 18 – Explainability (SHAP)
# Computes seeded permutation SHAP values on the 14 real inputs of the final model to show which tests drive predictions overall and for AS, ReA, SLE and RA.

# %%
# ==============================================================================================
# CELL 18 – EXPLAINABILITY (SHAP)
# What it does: Computes seeded permutation SHAP values on the 14 real inputs of the final model
# to show which tests drive predictions overall and for AS, ReA, SLE and RA.
# Output: SHAP importance table and figures.
# ==============================================================================================
log(f"Global explanations (SHAP, {SHAP_PERMUTATIONS} seeded permutations) on the 14 real inputs of the final model ...")
bundle["shap_background"] = Xtr_b.sample(n=min(SHAP_BACKGROUND, len(Xtr_b)), random_state=SEED).reset_index(drop=True)
Xs_raw = Xte_b.sample(n=min(SHAP_SAMPLES, len(Xte_b)), random_state=SEED); shap_ok = False
try:
    f_ = lambda a: raw_predict_proba(bundle, a)
    explainer = shap.explainers.Permutation(f_, shap.maskers.Independent(bundle["shap_background"][BASE_COLS].values, max_samples=len(bundle["shap_background"])),
                                           feature_names=BASE_COLS, seed=SEED)
    sv = np.asarray(explainer(Xs_raw.values, max_evals=SHAP_PERMUTATIONS * (2 * len(BASE_COLS) + 1), silent=True).values)
    imp = np.abs(sv).mean(axis=0); pd.DataFrame(imp, index=BASE_COLS, columns=short).round(5).to_csv(f"{OUT}/paper_results/shap_importance_by_class.csv")
    order = np.argsort(-imp.sum(1))[::-1]; fig, ax = plt.subplots(figsize=(8, 6)); left = np.zeros(len(order))
    for k in range(K): ax.barh([BASE_COLS[i] for i in order], imp[order, k], left=left, label=short[k]); left += imp[order, k]
    ax.legend(fontsize=8); ax.set_xlabel("Average impact on the prediction (SHAP)"); ax.set_title("Clinical markers driving the final model"); save(fig, "shap_global_importance")
    for k in [i for i, c in enumerate(class_names) if c in ("Ankylosing Spondylitis", "Reactive Arthritis", "Systemic Lupus Erythematosus", "Rheumatoid Arthritis")]:
        plt.figure(figsize=(8, 5.5)); shap.summary_plot(sv[:, :, k], Xs_raw.values, feature_names=BASE_COLS, show=False, max_display=14)
        plt.title(f"What pushes a patient toward {short[k]}"); plt.tight_layout(); plt.savefig(f"{OUT}/figures/shap_beeswarm_{short[k]}.png", dpi=200); plt.close()
    shap_ok = True
except Exception as e:
    log(f"  SHAP plots skipped ({type(e).__name__}: {e}). The model itself is unaffected.")


# %% [markdown]
# ## Cell 19 – Sensitivity mode for rare diseases
# Chooses, on training cross-validation, how strongly to favour rare diseases (strength 0.55) while costing at most 1.0 point of accuracy, then compares default and sensitive modes on the test set.

# %%
# ==============================================================================================
# CELL 19 – SENSITIVITY MODE FOR RARE DISEASES
# What it does: Chooses, on training cross-validation, how strongly to favour rare diseases
# (strength 0.55) while costing at most 1.0 point of accuracy, then compares default and sensitive
# modes on the test set.
# Output: Default vs sensitive mode table.
# ==============================================================================================
EXTRA_MD = ""
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
def recalls(y, pr): return precision_recall_fscore_support(y, pr, labels=range(K), zero_division=0)[1]
log(f"Sensitivity mode: how strongly to favour rare diseases (training CV only; overall accuracy may drop at most {SENS_MAX_ACC_DROP*100:.1f} points)")
prior_tr = np.bincount(ytr, minlength=K) / len(ytr)
oofp = CAND[SELECTED]["oof_cal"]; base_acc = accuracy_score(ytr, oofp.argmax(1)); best_bal = balanced_accuracy_score(ytr, oofp.argmax(1))
SENS_LAMBDA, sens_rows = 0.0, []
for lam in np.round(np.arange(0, 1.0001, 0.05), 2):
    pr = np.argmax(oofp * (1.0 / prior_tr) ** lam, axis=1); rc = recalls(ytr, pr)
    acc_, bal_ = accuracy_score(ytr, pr), balanced_accuracy_score(ytr, pr)
    sens_rows.append(dict(strength=float(lam), cv_accuracy=acc_, cv_balanced_accuracy=bal_, **{f"recall_{short[k]}": rc[k] for k in range(K)}))
    if base_acc - acc_ <= SENS_MAX_ACC_DROP + 1e-12 and bal_ > best_bal + 1e-12: SENS_LAMBDA, best_bal = float(lam), bal_
sens_df = pd.DataFrame(sens_rows); sens_df.round(4).to_csv(f"{OUT}/paper_results/sensitivity_mode_tradeoff_cv.csv", index=False)
w_sens = (1.0 / prior_tr) ** SENS_LAMBDA; w_sens = w_sens / w_sens.mean()
bundle["sensitivity_weights"] = [float(v) for v in w_sens]; bundle["sensitivity_lambda"] = SENS_LAMBDA
def mode_row(nm, pr):
    rc = recalls(yte, pr)
    return dict(mode=nm, accuracy=accuracy_score(yte, pr), balanced_accuracy=balanced_accuracy_score(yte, pr),
                macro_f1=f1_score(yte, pr, average="macro"), **{f"recall_{short[k]}": rc[k] for k in range(K)})
pred_sen = np.argmax(P_final * w_sens, axis=1)
sens_test = pd.DataFrame([mode_row("default", P_final.argmax(1)), mode_row(f"sensitive (strength {SENS_LAMBDA:.2f})", pred_sen)])
sens_test.round(4).to_csv(f"{OUT}/paper_results/sensitivity_mode_test.csv", index=False)
if SENS_LAMBDA == 0: log("  no strength met the rule -> sensitivity mode behaves exactly like the default mode")
log(f"  chosen strength {SENS_LAMBDA:.2f}. Frozen test set:\n" + sens_test.round(4).to_string(index=False))
EXTRA_MD += f"""
## Sensitivity mode (optional switch for rare diseases)
Probabilities are re-weighted by (1/disease frequency)^strength. The strength was chosen on training CV as the one with the highest
balanced accuracy that costs at most {SENS_MAX_ACC_DROP*100:.1f} points of overall accuracy. Chosen strength: {SENS_LAMBDA:.2f}.
Frozen test set:

{sens_test.round(4).to_markdown(index=False)}
"""


# %% [markdown]
# ## Cell 20 – Robustness on the test set
# Applies the same disturbances to the test patients for reporting only.

# %%
# ==============================================================================================
# CELL 20 – ROBUSTNESS ON THE TEST SET
# What it does: Applies the same disturbances to the test patients for reporting only.
# Output: Accuracy of each model under each disturbance.
# ==============================================================================================
try:
    log("Robustness on the frozen test set (report only; the deployment decision was already made on training data) ...")
    rt = {}
    for kind in PERTURB:
        nrep = 1 if kind in DETERMINISTIC else ROBUST_REPEATS; acc_ = {nm: [] for nm in ROB_MODELS}
        for r in range(nrep):
            Fv = engineer(perturb(Xte_b, kind, np.random.default_rng(SEED + 7 + r)))
            for nm in CAND: acc_[nm].append(accuracy_score(yte, model_predict_proba(cand_bundle(nm), Fv).argmax(1)))
            acc_["XGBoost (single model, reference)"].append(accuracy_score(yte, final_members["xgb"].predict_proba(member_input("xgb", Fv[FEATS])).argmax(1)))
        rt[kind] = {nm: float(np.mean(v)) for nm, v in acc_.items()}
    rob_test = pd.DataFrame(rt)[PERTURB]; rob_test.round(4).to_csv(f"{OUT}/paper_results/robustness_test_accuracy.csv")
    log("  accuracy (%):\n" + (rob_test * 100).round(2).to_string())
    EXTRA_MD += f"\n## Robustness on the frozen test set (report only)\nAccuracy (%):\n\n{(rob_test * 100).round(2).to_markdown()}\n"
    fig, ax = plt.subplots(figsize=(10, 4.8)); (rob_test.T * 100).plot.bar(ax=ax); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, float(rob_test.values.min() * 100) - 5), 90); ax.set_title("Accuracy when lab results are disturbed (frozen test set)")
    ax.set_xticklabels([PERTURB_TEXT[k] for k in PERTURB], rotation=25, ha="right", fontsize=8); ax.legend(fontsize=7); save(fig, "robustness_test")
except Exception as e: log(f"  robustness-on-test skipped ({type(e).__name__}: {e})")


# %% [markdown]
# ## Cell 21 – Per-disease repeated-split results and paired tests
# Summarises per-disease recall over the 10 splits and runs paired t-, Wilcoxon and sign tests between the final model and each alternative.

# %%
# ==============================================================================================
# CELL 21 – PER-DISEASE REPEATED-SPLIT RESULTS AND PAIRED TESTS
# What it does: Summarises per-disease recall over the 10 splits and runs paired t-, Wilcoxon and
# sign tests between the final model and each alternative.
# Output: Per-disease recall table and paired-test table.
# ==============================================================================================
try:
    rc_cols = [f"recall_{s}" for s in short]
    pc_rep = rep.groupby("model")[rc_cols].agg(["mean", "std"]); pc_rep.columns = [f"{a}_{b}" for a, b in pc_rep.columns]
    pc_rep.round(4).to_csv(f"{OUT}/paper_results/repeated_splits_per_disease_recall.csv")
    pc_show = pd.DataFrame({s: [f"{pc_rep.loc[m, f'recall_{s}_mean']*100:.1f} +/- {pc_rep.loc[m, f'recall_{s}_std']*100:.1f}" for m in pc_rep.index] for s in short}, index=pc_rep.index)
    from scipy.stats import ttest_rel, wilcoxon
    pt_rows = []
    for m in piv.columns:
        if m == SELECTED: continue
        d_ = piv[SELECTED] - piv[m]
        try: pw = float(wilcoxon(piv[SELECTED], piv[m]).pvalue)
        except Exception: pw = np.nan
        pt_rows.append(dict(compared_with=m, mean_gain_points=d_.mean() * 100, wins=int((d_ > 0).sum()), losses=int((d_ < 0).sum()),
                            paired_t_p=float(ttest_rel(piv[SELECTED], piv[m]).pvalue) if d_.std() > 0 else np.nan, wilcoxon_p=pw,
                            sign_test_p=float(binomtest(int((d_ > 0).sum()), int((d_ != 0).sum()), 0.5).pvalue) if (d_ != 0).any() else np.nan))
    paired_df = pd.DataFrame(pt_rows); paired_df.round(4).to_csv(f"{OUT}/paper_results/repeated_splits_paired_tests.csv", index=False)
    log(f"Per-disease recall over {N_REPEATS} splits (mean +/- sd, %):\n" + pc_show.to_string() + "\nPaired tests (deployed model vs others):\n" + paired_df.round(4).to_string(index=False))
    EXTRA_MD += f"""
## Per-disease recall over the {N_REPEATS} repeated splits (mean +/- sd, %)
{pc_show.to_markdown()}

## Paired tests over the repeated splits (deployed model vs each alternative, accuracy)
{paired_df.round(4).to_markdown(index=False)}
"""
except Exception as e: log(f"  per-disease repeated-split summary skipped ({type(e).__name__}: {e})")


# %% [markdown]
# ## Cell 22 – Subgroup analysis
# Measures test accuracy by sex, age band, serostatus (two definitions) and work-up completeness, with Wilson 95% intervals and chi-square tests.

# %%
# ==============================================================================================
# CELL 22 – SUBGROUP ANALYSIS
# What it does: Measures test accuracy by sex, age band, serostatus (two definitions) and work-up
# completeness, with Wilson 95% intervals and chi-square tests.
# Output: Subgroup table and figure.
# ==============================================================================================
try:
    from scipy.stats import chi2_contingency
    Xt = Xte_b.reset_index(drop=True); ok_dep = P_final.argmax(1) == yte; ok_v2 = P_ens.argmax(1) == yte
    both = (Xt["RF"].notna() & Xt["Anti-CCP"].notna()).values
    neg20 = ((Xt["RF"] <= 20) & (Xt["Anti-CCP"] <= 20)).values; negclin = ((Xt["RF"] < 14) & (Xt["Anti-CCP"] < 17)).values
    G = [("Sex", "Female", (Xt.Gender == 0).values), ("Sex", "Male", (Xt.Gender == 1).values),
         ("Age", "20-39", (Xt.Age < 40).values), ("Age", "40-59", ((Xt.Age >= 40) & (Xt.Age < 60)).values), ("Age", "60+", (Xt.Age >= 60).values),
         ("Serostatus, dataset cut-off (RF and anti-CCP <= 20)", "seronegative", both & neg20), ("Serostatus, dataset cut-off (RF and anti-CCP <= 20)", "seropositive", both & ~neg20),
         ("Serostatus, common lab cut-offs (RF < 14, anti-CCP < 17)", "seronegative", both & negclin), ("Serostatus, common lab cut-offs (RF < 14, anti-CCP < 17)", "seropositive", both & ~negclin),
         ("Serostatus", "RF or anti-CCP not tested", ~both),
         ("Work-up", "0-1 tests missing", n_missing <= 1), ("Work-up", "2-3 tests missing", (n_missing >= 2) & (n_missing <= 3)), ("Work-up", "4+ tests missing", n_missing >= 4)]
    sg_rows = []
    for grp, lvl, m in G:
        n_ = int(m.sum())
        if n_ == 0: continue
        k_ = int(ok_dep[m].sum()); lo, hi = wilson(k_, n_); labs_ = np.unique(yte[m])
        sg_rows.append(dict(grouping=grp, subgroup=lvl, patients=n_, accuracy=k_ / n_, ci_low=lo, ci_high=hi, ensemble_accuracy=float(ok_v2[m].mean()),
                            macro_f1=f1_score(yte[m], P_final.argmax(1)[m], labels=labs_, average="macro"),
                            ece=ece_score(yte[m], P_final[m]) if n_ >= 100 else np.nan, note="small group (n<50)" if n_ < 50 else ""))
    sg = pd.DataFrame(sg_rows); pvals = {}
    for grp in sg.grouping.unique():
        lv = [(lvl, m) for g_, lvl, m in G if g_ == grp and m.sum() > 0]
        if len(lv) < 2: continue
        tab = np.array([[int(ok_dep[m].sum()), int((~ok_dep[m]).sum())] for _, m in lv])
        try: pvals[grp] = float(chi2_contingency(tab)[1])
        except Exception: pvals[grp] = np.nan
    sg["difference_between_levels_p"] = sg.grouping.map(pvals)
    sg.round(4).to_csv(f"{OUT}/paper_results/subgroup_analysis_test.csv", index=False)
    log("Subgroup analysis (frozen test set):\n" + sg[["grouping", "subgroup", "patients", "accuracy", "ci_low", "ci_high", "ensemble_accuracy", "difference_between_levels_p"]].round(4).to_string(index=False))
    EXTRA_MD += f"""
## Subgroup analysis (frozen test set; 95% Wilson CI; p = chi-square test for any accuracy difference between levels of a grouping)
{sg.round(4).to_markdown(index=False)}
"""
    fig, ax = plt.subplots(figsize=(8, 6)); lab_ = [f"{r.grouping.split(' (')[0]}: {r.subgroup} (n={r.patients})" for r in sg.itertuples()]
    ax.errorbar(sg.accuracy * 100, range(len(sg)), xerr=[(sg.accuracy - sg.ci_low) * 100, (sg.ci_high - sg.accuracy) * 100], fmt="o", capsize=3)
    ax.axvline(main["accuracy"] * 100, color="r", ls="--", lw=1, label="overall"); ax.set_yticks(range(len(sg))); ax.set_yticklabels(lab_, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("Accuracy (%) with 95% CI"); ax.legend(fontsize=8); ax.set_title("Accuracy by patient subgroup (frozen test set)"); save(fig, "subgroup_accuracy")
except Exception as e: log(f"  subgroup analysis skipped ({type(e).__name__}: {e})")


# %% [markdown]
# ## Cell 23 – Comparison with published methods
# Re-runs the protocols of the two published studies on their exact splits (80/20 with MICE; 70/30 with mean/mode imputation) with our model on the same patients, and checks the effect of oversampling before vs after the split.

# %%
# ==============================================================================================
# CELL 23 – COMPARISON WITH PUBLISHED METHODS
# What it does: Re-runs the protocols of the two published studies on their exact splits (80/20
# with MICE; 70/30 with mean/mode imputation) with our model on the same patients, and checks the
# effect of oversampling before vs after the split.
# Output: Head-to-head table and McNemar tests.
# ==============================================================================================
h2h_rows = []
def h2h_add(protocol, model, yv, P_or_pred, note=""):
    pr = P_or_pred if np.ndim(P_or_pred) == 1 else np.asarray(P_or_pred).argmax(1); rc = recalls(yv, pr)
    h2h_rows.append(dict(protocol=protocol, model=model, test_patients=len(yv), accuracy=accuracy_score(yv, pr), macro_f1=f1_score(yv, pr, average="macro"),
                         balanced_accuracy=balanced_accuracy_score(yv, pr), recall_AS=rc[short.index("AS")] if "AS" in short else np.nan,
                         recall_ReA=rc[short.index("ReA")] if "ReA" in short else np.nan, note=note))
    return pr
def ours_bayes(Xa, ya, Xb, seed=SEED):
    g, _ = select_bayes(Xa, ya, StratifiedKFold(3, shuffle=True, random_state=seed)); return bayes_proba(bayes_fit(Xa, ya, K, **g), Xb)
def mcnemar_p(y, pa, pb):
    a_, b_ = (pa == y), (pb == y); n1, n2 = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
    return (n1, n2, float(binomtest(n1, n1 + n2, 0.5).pvalue) if n1 + n2 else 1.0)
H2H_SIG = []
# 19a. BMC Rheumatology protocol: 80/20 stratified split, seed 42, MICE + standard scaling, fixed hyper-parameters
try:
    log("Head-to-head 1/3: BMC Rheumatology protocol (80/20 split, MICE imputation, their fixed settings) ...")
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    P1 = "BMC 2026 protocol (80/20, seed 42)"
    Xa, Xb, ya, yb = train_test_split(X_all, y_all, test_size=0.20, stratify=y_all, random_state=42)
    imp = IterativeImputer(max_iter=10, random_state=42).fit(Xa); sc = StandardScaler().fit(imp.transform(Xa))
    Za, Zb = sc.transform(imp.transform(Xa)), sc.transform(imp.transform(Xb))
    def pick_trees(make):
        best = None
        for n_ in H2H_TREE_GRID:
            s_ = accuracy_score(ya, cross_val_predict(make(n_), Za, ya, cv=StratifiedKFold(3, shuffle=True, random_state=42)))
            if best is None or s_ > best[0]: best = (s_, n_)
        return best[1]
    mk_x = lambda n_: xgb.XGBClassifier(n_estimators=n_, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                        objective="multi:softprob", eval_metric="mlogloss", n_jobs=N_JOBS, random_state=42, verbosity=0)
    mk_l = lambda n_: lgb.LGBMClassifier(n_estimators=n_, num_leaves=31, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=42,
                                         n_jobs=N_JOBS, verbosity=-1, deterministic=True, force_col_wise=True)
    nx, nl = pick_trees(mk_x), pick_trees(mk_l)
    preds = {}
    preds["Random Forest (their settings)"] = h2h_add(P1, "Random Forest (their settings, re-implemented)", yb,
        RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=N_JOBS).fit(Za, ya).predict_proba(Zb))
    preds["XGBoost (their settings)"] = h2h_add(P1, "XGBoost (their settings, re-implemented)", yb, mk_x(nx).fit(Za, ya).predict_proba(Zb), f"trees not reported; {nx} chosen by training CV")
    preds["LightGBM (their settings)"] = h2h_add(P1, "LightGBM (their settings, re-implemented)", yb, mk_l(nl).fit(Za, ya).predict_proba(Zb), f"trees not reported; {nl} chosen by training CV")
    imp_all = IterativeImputer(max_iter=10, random_state=42).fit(X_all); sc_all = StandardScaler().fit(imp_all.transform(X_all))
    leakX = pd.DataFrame(sc_all.transform(imp_all.transform(X_all)), index=X_all.index)
    h2h_add(P1, "XGBoost, imputation fitted on ALL data (as described in the paper)", yb, mk_x(nx).fit(leakX.loc[Xa.index].values, ya).predict_proba(leakX.loc[Xb.index].values), "leaky variant")
    pb_ = h2h_add(P1, "OURS: Range-aware Bayes (no imputation)", yb, ours_bayes(Xa, ya, Xb))
    h2h_rows.append(dict(protocol=P1, model="Published: XGBoost (paper, Table 2)", test_patients=2417, accuracy=0.8548, note="as reported"))
    h2h_rows.append(dict(protocol=P1, model="Published: Random Forest (paper, Table 2)", test_patients=2417, accuracy=0.8378, note="as reported (their confusion matrix implies 84.24%)"))
    bn = max(preds, key=lambda k: accuracy_score(yb, preds[k])); n1, n2, pv = mcnemar_p(yb, pb_, preds[bn]); H2H_SIG.append(dict(protocol=P1, ours_vs=bn, only_ours_right=n1, only_theirs_right=n2, p_value=pv))
except Exception as e: log(f"  BMC protocol skipped ({type(e).__name__}: {e})")
# 19b. PeerJ protocol, honest version (no oversampling): 70/30 split, mean/mode imputation, default-style settings
NUMC, CATC = ["Age"] + LABS, ["Gender"] + SEROLOGY
def mean_mode_fit(Xa): return {**{c: float(Xa[c].mean()) for c in NUMC}, **{c: float(Xa[c].mode().iloc[0]) for c in CATC}}
try:
    log("Head-to-head 2/3: PeerJ protocol without oversampling (70/30 split, mean/mode imputation, their settings) ...")
    from sklearn.ensemble import GradientBoostingClassifier
    P2 = "PeerJ 2025 protocol, no oversampling (70/30, seed 42)"
    Xa2, Xb2, ya2, yb2 = train_test_split(X_all, y_all, test_size=0.30, stratify=y_all, random_state=42)
    fill = mean_mode_fit(Xa2); Fa2, Fb2 = Xa2.fillna(fill).values, Xb2.fillna(fill).values
    pg = h2h_add(P2, "Gradient Boosting (their settings, re-implemented)", yb2, GradientBoostingClassifier(n_estimators=GB_TREES, learning_rate=0.1, max_depth=3, random_state=42).fit(Fa2, ya2).predict_proba(Fb2))
    prf_ = h2h_add(P2, "Random Forest (their settings, re-implemented)", yb2, RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS).fit(Fa2, ya2).predict_proba(Fb2))
    pb2 = h2h_add(P2, "OURS: Range-aware Bayes (no imputation)", yb2, ours_bayes(Xa2, ya2, Xb2))
    h2h_rows.append(dict(protocol=P2, model="Published: Gradient Boosting, unbalanced data (paper, Table 11)", test_patients=3626, accuracy=0.8392, note="as reported"))
    best2 = pg if accuracy_score(yb2, pg) >= accuracy_score(yb2, prf_) else prf_; n1, n2, pv = mcnemar_p(yb2, pb2, best2)
    H2H_SIG.append(dict(protocol=P2, ours_vs="best re-implemented model", only_ours_right=n1, only_theirs_right=n2, p_value=pv))
except Exception as e: log(f"  PeerJ protocol skipped ({type(e).__name__}: {e})")
# 19c. Why the PeerJ headline (86.89%) is not comparable: oversampling BEFORE vs AFTER the split
if HAVE_IMBLEARN:
    try:
        log("Head-to-head 3/3: effect of oversampling Reactive Arthritis before vs after the split (ADASYN) ...")
        from imblearn.over_sampling import ADASYN
        from sklearn.ensemble import GradientBoostingClassifier
        rea = class_names.index("Reactive Arthritis"); target_total = 2946; P3 = "PeerJ oversampling check (70/30)"
        full_fill = X_all.fillna(mean_mode_fit(X_all)).values
        Xr, yr = ADASYN(sampling_strategy={rea: target_total}, n_neighbors=5, random_state=42).fit_resample(full_fill, y_all)
        syn = np.r_[np.zeros(len(y_all), bool), np.ones(len(yr) - len(y_all), bool)]
        Ra, Rb, ra, rb, sa, sb = train_test_split(Xr, yr, syn, test_size=0.30, stratify=yr, random_state=42)
        pl = GradientBoostingClassifier(n_estimators=GB_TREES, learning_rate=0.1, max_depth=3, random_state=42).fit(Ra, ra).predict(Rb)
        h2h_add(P3, "Gradient Boosting, oversampled BEFORE split (as in the paper)", rb, pl, f"{int(sb.sum())} of {int((rb == rea).sum())} ReA test cases are synthetic")
        h2h_add(P3, "  same model, scored on the REAL test patients only", rb[~sb], pl[~sb])
        tgt_tr = int(round(target_total / (y_all == rea).sum() * (ya2 == rea).sum()))
        Xo, yo = ADASYN(sampling_strategy={rea: tgt_tr}, n_neighbors=5, random_state=42).fit_resample(Fa2, ya2)
        h2h_add(P3, "Gradient Boosting, oversampled AFTER split (training only)", yb2,
                GradientBoostingClassifier(n_estimators=GB_TREES, learning_rate=0.1, max_depth=3, random_state=42).fit(Xo, yo).predict(Fb2), "honest version")
        h2h_rows.append(dict(protocol=P3, model="Published: Gradient Boosting, balanced data (paper, Table 13)", test_patients=4355, accuracy=0.8689, note="as reported; test includes synthetic patients"))
    except Exception as e: log(f"  oversampling check skipped ({type(e).__name__}: {e})")
else: log("Head-to-head 3/3 skipped: package 'imbalanced-learn' is not available in this environment")
if h2h_rows:
    h2h = pd.DataFrame(h2h_rows); h2h.round(4).to_csv(f"{OUT}/paper_results/head_to_head_published_protocols.csv", index=False)
    h2h_sig = pd.DataFrame(H2H_SIG); h2h_sig.round(4).to_csv(f"{OUT}/paper_results/head_to_head_mcnemar.csv", index=False)
    log("Head-to-head results:\n" + h2h[["protocol", "model", "accuracy", "macro_f1", "recall_AS", "recall_ReA", "note"]].round(4).to_string(index=False)
        + ("\nMcNemar (ours vs best re-implemented model on the same split):\n" + h2h_sig.round(4).to_string(index=False) if len(h2h_sig) else ""))
    EXTRA_MD += f"""
## Head-to-head on the published protocols (same splits, same patients; 'Published' rows are the papers' own numbers)
{h2h.round(4).to_markdown(index=False)}

McNemar test, ours vs the best re-implemented model on the identical split:
{h2h_sig.round(4).to_markdown(index=False) if len(h2h_sig) else "not available"}
"""
    try:
        hp = h2h[~h2h.model.str.startswith("  ")].copy(); fig, ax = plt.subplots(figsize=(9, 0.42 * len(hp) + 1.5))
        cols_ = ["#d62728" if m.startswith("OURS") else ("#999999" if m.startswith("Published") else "#4c78a8") for m in hp.model]
        ax.barh(range(len(hp)), hp.accuracy * 100, color=cols_); ax.set_yticks(range(len(hp)))
        ax.set_yticklabels([f"[{p.split(' (')[0]}] {m}" for p, m in zip(hp.protocol, hp.model)], fontsize=7); ax.invert_yaxis()
        ax.set_xlim(75, 90); ax.set_xlabel("Accuracy (%)"); ax.set_title("Head-to-head on the published protocols (red = ours, grey = published numbers)"); save(fig, "head_to_head")
    except Exception as e: log(f"  head-to-head figure skipped ({e})")

# sensitivity trade-off figure
try:
    fig, ax = plt.subplots(figsize=(6.5, 4.2)); ax.plot(sens_df.strength, sens_df.cv_accuracy * 100, label="overall accuracy")
    ax.plot(sens_df.strength, sens_df.cv_balanced_accuracy * 100, label="balanced accuracy")
    for s in ("ReA", "AS"):
        if f"recall_{s}" in sens_df: ax.plot(sens_df.strength, sens_df[f"recall_{s}"] * 100, ls=":", label=f"{s} recall")
    ax.axvline(SENS_LAMBDA, color="k", ls="--", lw=1, label=f"chosen ({SENS_LAMBDA:.2f})"); ax.set_xlabel("Strength of rare-disease weighting")
    ax.set_ylabel("% (training cross-validation)"); ax.legend(fontsize=7); ax.set_title("Sensitivity mode trade-off"); save(fig, "sensitivity_mode_tradeoff")
    fig, ax = plt.subplots(figsize=(8, 4)); (rob_drop.drop(columns="clean") * 100).T.plot.bar(ax=ax); ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Accuracy change vs clean (points)"); ax.set_xticklabels([PERTURB_TEXT[k] for k in PERTURB if k != "clean"], rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=7); ax.set_title("Robustness on held-out training folds (used for the deployment decision)"); save(fig, "robustness_training_cv")
except Exception as e: log(f"  extra figures skipped ({e})")


# %% [markdown]
# ## Cell 24 – Deployment package
# Saves the trained model, inference.py, api.py (FastAPI), requirements.txt with exact versions, disease profiles and sample patients.

# %%
# ==============================================================================================
# CELL 24 – DEPLOYMENT PACKAGE
# What it does: Saves the trained model, inference.py, api.py (FastAPI), requirements.txt with
# exact versions, disease profiles and sample patients.
# Output: Files in outputs/artifacts.
# ==============================================================================================
json.dump(dict(classes=class_names, short=short, source="training patients only",
               profiles={c: {k: float(v) for k, v in prof_rows[c].items()} for c in class_names},
               lab_ranges={l: [float(Xtr_b[l].min()), float(Xtr_b[l].max())] for l in LABS}),
          open(f"{OUT}/artifacts/class_profiles.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
try:
    import dill; dill.extend(False)
except Exception: pass
joblib.dump(bundle, f"{OUT}/artifacts/model_bundle.pkl", compress=3)

INFERENCE = SHARED_CODE + r'''
import os, joblib
_B = None
def load(path=None):
    global _B
    if _B is None: _B = joblib.load(path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_bundle.pkl"))
    return _B
def _frame(records):
    recs = records if isinstance(records, list) else [records]
    df = pd.DataFrame(recs)
    for c in BASE_COLS:
        if c not in df.columns: df[c] = None
    return df
def _range_warnings(b, x):
    w = []
    for c, (lo, hi) in b.get("input_ranges", {}).items():
        v = x[c]
        if not pd.isna(v) and (v < lo or v > hi): w.append(f"{c} = {v:g} is outside the training range {lo:g}-{hi:g}; treat the result with extra care")
    return w
def predict(records, path=None, recommend=True, mode="default"):
    """records: dict or list of dicts using the ORIGINAL column names. Unmeasured tests: leave out or use None.
    status: 'confident' | 'needs_review' | 'insufficient_data'. 'needs_review' answers include recommended_tests.
    mode: 'default' (most accurate overall) or 'sensitive' (favours rare diseases; probabilities stay the calibrated ones)."""
    if mode not in ("default", "sensitive"): raise ValueError("mode must be 'default' or 'sensitive'")
    b = load(path); raw = _frame(records); X, issues = encode_raw(raw, return_issues=True)
    P = model_predict_proba(b, engineer(X)); out = []
    w = np.asarray(b.get("sensitivity_weights") or np.ones(P.shape[1]), dtype=float) if mode == "sensitive" else np.ones(P.shape[1])
    for i in range(len(P)):
        p = P[i]; j0 = int(p.argmax()); j = int(np.argmax(p * w)); miss = [c for c in TESTS if pd.isna(X.iloc[i][c])]
        measured = len(TESTS) - len(miss); warn = issues[i] + _range_warnings(b, X.iloc[i])
        if measured < b.get("min_measured_tests", 3):
            out.append({"predicted_disease": None, "confidence": None, "probabilities": None, "status": "insufficient_data",
                        "message": f"Only {measured} of {len(TESTS)} tests were measured; at least {b.get('min_measured_tests', 3)} are needed.",
                        "missing_tests": miss, "n_missing_tests": len(miss), "warnings": warn, "recommended_tests": []})
            continue
        order = np.argsort(-p)
        r = {"predicted_disease": b["classes"][j], "confidence": float(p[j]), "runner_up": b["classes"][int([o for o in order if o != j][0])],
             "probabilities": {b["classes"][k]: float(p[k]) for k in range(len(p))},
             "status": "confident" if (j == j0 and p[j] >= b["confidence_threshold"]) else "needs_review", "mode": mode,
             "missing_tests": miss, "n_missing_tests": len(miss), "warnings": warn, "recommended_tests": []}
        if recommend and r["status"] == "needs_review" and miss: r["recommended_tests"] = recommend_tests(b, X.iloc[[i]], top=3)
        out.append(r)
    return out
def recommend(record, top=3, path=None):
    """Which missing tests would most likely give a confident answer for this patient."""
    b = load(path); X = encode_raw(_frame(record)); return recommend_tests(b, X.iloc[[0]], top=top)
def _display(col, v):
    if pd.isna(v): return "not tested"
    if col in SEROLOGY: return "Positive" if v == 1 else "Negative"
    if col == "Gender": return "Male" if v == 1 else "Female"
    return float(v)
def explain(record, top_k=8, target_class=None, compare_with=None, path=None):
    """Bayes model: EXACT explanation. Each measured test gets a likelihood ratio = how many times more likely this
    result is under the predicted disease than under the runner-up (or compare_with). Stack models: seeded SHAP."""
    b = load(path); X = encode_raw(_frame(record))[BASE_COLS]; p = raw_predict_proba(b, X.values)[0]
    k = b["classes"].index(target_class) if target_class else int(p.argmax())
    if b["kind"] == "bayes":
        r = b["classes"].index(compare_with) if compare_with else int([i for i in np.argsort(-p) if i != k][0])
        ev = sorted(bayes_evidence(b["bayes"], X, k, r), key=lambda t: -abs(t[1]))[:top_k]
        items = [{"feature": c, "value": _display(c, X[c].iloc[0]), "impact": w, "likelihood_ratio": float(np.exp(w))} for c, w in ev]
        return {"method": "exact likelihood ratios", "explained_class": b["classes"][k], "compared_with": b["classes"][r],
                "probability": float(p[k]), "probability_compared": float(p[r]),
                "not_tested": [c for c in TESTS if pd.isna(X[c].iloc[0])],
                "pushes_toward": [x for x in items if x["impact"] > 0], "pushes_against": [x for x in items if x["impact"] < 0]}
    import shap
    f = lambda a: raw_predict_proba(b, a); bgd = b["shap_background"][BASE_COLS].values
    ex = shap.explainers.Permutation(f, shap.maskers.Independent(bgd, max_samples=len(bgd)), feature_names=BASE_COLS, seed=b["seed"])
    s = np.asarray(ex(X.values, max_evals=b.get("shap_permutations", 8) * (2 * len(BASE_COLS) + 1), silent=True).values)[0][:, k]
    items = [{"feature": BASE_COLS[i], "value": _display(BASE_COLS[i], X.iloc[0, i]), "impact": float(s[i])} for i in np.argsort(-np.abs(s))[:top_k]]
    return {"method": "SHAP (seeded permutation)", "explained_class": b["classes"][k], "probability": float(p[k]),
            "pushes_toward": [x for x in items if x["impact"] > 0], "pushes_against": [x for x in items if x["impact"] < 0]}
'''
open(f"{OUT}/artifacts/inference.py", "w", encoding="utf-8").write(INFERENCE)
open(f"{OUT}/artifacts/api.py", "w", encoding="utf-8").write(textwrap.dedent('''
    # Run:  pip install -r requirements.txt
    #       uvicorn api:app --reload --port 8000
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import inference
    app = FastAPI(title="Autoimmune Diagnosis CDSS")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    @app.get("/health")
    def health():
        b = inference.load()
        return {"status": "ok", "model_version": b.get("version"), "model": b.get("model_name"), "classes": b["classes"],
                "confidence_threshold": b.get("confidence_threshold"), "min_measured_tests": b.get("min_measured_tests"),
                "sensitivity_strength": b.get("sensitivity_lambda")}
    @app.post("/predict")
    def predict(patient: dict, mode: str = "default"): return inference.predict(patient, mode=mode)[0]
    @app.post("/explain")
    def explain(patient: dict, top_k: int = 8, compare_with: str = None): return inference.explain(patient, top_k=top_k, compare_with=compare_with)
    @app.post("/recommend")
    def recommend(patient: dict, top: int = 3): return inference.recommend(patient, top=top)
    @app.get("/profiles")
    def profiles():
        import json, os
        return json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_profiles.json"), encoding="utf-8"))
'''))
from importlib.metadata import version as _ver, PackageNotFoundError
pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "openpyxl"] + (["lightgbm", "xgboost", "catboost", "shap"] if bundle["kind"] == "stack" else [])
req = []
for pkg in pkgs:
    try: req.append(f"{pkg}=={_ver(pkg)}")
    except PackageNotFoundError: req.append(pkg)
req += ["fastapi", "uvicorn"]
open(f"{OUT}/artifacts/requirements.txt", "w").write(f"# Exact versions used to train the model (Python {sys.version.split()[0]}). Final model: {SELECTED}\n" + "\n".join(req) + "\n")
samples = []
for k in range(K):
    for i in np.where(yte == k)[0][:2]:
        r = df.loc[Xte_b.index[i], BASE_COLS].to_dict(); samples.append({"true_disease": class_names[k], "patient": {a: (None if pd.isna(b) else (b.item() if hasattr(b, "item") else b)) for a, b in r.items()}})
json.dump(samples, open(f"{OUT}/artifacts/sample_patients.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


# %% [markdown]
# ## Cell 25 – Automatic self-tests
# Runs six checks on the saved package: exact reproduction, loading in a fresh process, input handling, repeatable explanations, test recommendations and sensitivity mode.

# %%
# ==============================================================================================
# CELL 25 – AUTOMATIC SELF-TESTS
# What it does: Runs six checks on the saved package: exact reproduction, loading in a fresh
# process, input handling, repeatable explanations, test recommendations and sensitivity mode.
# Output: One 'passed' line per test.
# ==============================================================================================
# self-test 1: saved package reproduces the in-memory predictions
import importlib.util
sys.dont_write_bytecode = True          # no __pycache__ inside the artifacts folder
spec = importlib.util.spec_from_file_location("inference", f"{OUT}/artifacts/inference.py"); inf = importlib.util.module_from_spec(spec); spec.loader.exec_module(inf)
chk = df.loc[Xte_b.index[:40], BASE_COLS]; res = inf.predict(chk.to_dict("records"), path=f"{OUT}/artifacts/model_bundle.pkl")
pkg_diff = max(abs(r["probabilities"][class_names[j]] - P_final[i, j]) for i, r in enumerate(res) if r["probabilities"] for j in range(K))
assert pkg_diff < 1e-6, f"packaging mismatch {pkg_diff}"; log(f"Self-test 1 passed: saved package reproduces predictions (max difference {pkg_diff:.1e})")
# self-test 2: loads in a fresh process without 'dill' (and, for the Bayes model, without any tree library)
block = "['dill']" if bundle["kind"] == "stack" else "['dill','lightgbm','xgboost','catboost','shap']"
code = (f"import sys, json\nfor m in {block}: sys.modules[m] = None\nsys.path.insert(0, {f'{OUT}/artifacts'!r}); import inference\n"
        f"s = json.load(open({f'{OUT}/artifacts/sample_patients.json'!r}, encoding='utf-8')); print(inference.predict(s[0]['patient'])[0]['status'])")
r2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
if r2.returncode == 0: log(f"Self-test 2 passed: package loads in a fresh process without {block}")
else:
    open(f"{OUT}/artifacts/requirements.txt", "a").write("dill\n"); log("Self-test 2 FAILED (dill added to requirements). Error: " + r2.stderr.strip()[-300:])
# self-test 3: input handling
t3 = inf.predict([{"Age": 40, "Gender": "Female"}, {"Age": 40, "Gender": "Female", "ESR": 30, "CRP": "12", "HLA-B27": "weak positive", "ANA": "equivocal"}],
                 path=f"{OUT}/artifacts/model_bundle.pkl")
assert t3[0]["status"] == "insufficient_data" and t3[1]["status"] != "insufficient_data" and "HLA-B27" not in t3[1]["missing_tests"] and t3[1]["warnings"]
log(f"Self-test 3 passed: empty patient refused; 'weak positive' read as positive; unreadable value reported")
# self-test 4: explanations are repeatable (tolerance 1e-9: multi-threaded maths can differ in the 16th decimal)
e1 = inf.explain(chk.iloc[0].to_dict(), path=f"{OUT}/artifacts/model_bundle.pkl"); e2 = inf.explain(chk.iloc[0].to_dict(), path=f"{OUT}/artifacts/model_bundle.pkl")
v1_ = [x["impact"] for x in e1["pushes_toward"] + e1["pushes_against"]]; v2_ = [x["impact"] for x in e2["pushes_toward"] + e2["pushes_against"]]
same = len(v1_) == len(v2_) and np.allclose(v1_, v2_, atol=1e-9, rtol=0)
log(f"Self-test 4 {'passed' if same else 'FAILED'}: same patient explained twice -> {'identical' if same else 'different'} ({e1['method']}; "
    + ", ".join(f"{x['feature']} {x.get('likelihood_ratio', x['impact']):.2g}" for x in e1['pushes_toward'][:3]) + ")")
# self-test 5: next-best-test returns a ranked list for a 'needs review' patient
nr = [i for i, r in enumerate(res) if r["status"] == "needs_review" and r["missing_tests"]]
if nr: log(f"Self-test 5 passed: a 'needs review' patient gets recommendations -> {[(t['test'], round(t['chance_of_confident_answer'], 2)) for t in res[nr[0]]['recommended_tests']]}")
else: log("Self-test 5: no 'needs review' patient with missing tests among the first 40 (nothing to check)")

# self-test 6: sensitivity mode
t6 = inf.predict(chk.to_dict("records"), path=f"{OUT}/artifacts/model_bundle.pkl", mode="sensitive")
assert all(r["status"] == "insufficient_data" or (r["predicted_disease"] in class_names and r["mode"] == "sensitive") for r in t6)
ch6 = sum(a_["predicted_disease"] != b_["predicted_disease"] for a_, b_ in zip(res, t6))
log(f"Self-test 6 passed: sensitivity mode works (strength {bundle['sensitivity_lambda']:.2f}; {ch6} of {len(t6)} test patients get a different answer)")


# %% [markdown]
# ## Cell 26 – Results summary and download
# Writes RESULTS_SUMMARY.md and results.json and zips all outputs into project_artifacts.zip for download.

# %%
# ==============================================================================================
# CELL 26 – RESULTS SUMMARY AND DOWNLOAD
# What it does: Writes RESULTS_SUMMARY.md and results.json and zips all outputs into
# project_artifacts.zip for download.
# Output: The full summary printed, and the download path.
# ==============================================================================================
vals = {**main, "SLE_f1": sle_f1, "selective_accuracy": sel_acc}
tgt_rows = [f"| {k} | {TARGETS[k]} | {vals[k]:.4f} | {'PASS' if ((vals[k] <= TARGETS[k]) if k == 'ece' else (vals[k] >= TARGETS[k])) else 'below target'} |" for k in TARGETS]
rep_tbl = rep_sum[["accuracy_mean", "accuracy_std", "macro_f1_mean", "balanced_accuracy_mean", "log_loss_mean", "ece_mean", "top2_accuracy_mean"]].round(4)
summary = f"""# Results summary (auto-generated, pipeline {SCRIPT_VERSION})
Dataset: {len(df)} patients, {K} classes | Train {len(ytr)} / frozen test {len(yte)}
Chosen on training cross-validation: {CV_WINNER} | DEPLOYED after the robustness check: **{SELECTED}** (temperature {bundle['temperature']:.3f})\nSensitivity mode strength: {bundle['sensitivity_lambda']:.2f}
Unreadable input values in the dataset: {len(flat_issues)}

## Model selection (training cross-validation; the test set played no part)
{sel_df.round(4).to_markdown(index=False)}

## Headline numbers (frozen test set, 95% bootstrap CI)
- Accuracy {main['accuracy']:.4f}  {ci['accuracy'][0]:.3f}-{ci['accuracy'][1]:.3f}
- Macro F1 {main['macro_f1']:.4f}  {ci['macro_f1'][0]:.3f}-{ci['macro_f1'][1]:.3f}
- Balanced accuracy {main['balanced_accuracy']:.4f}  {ci['balanced_accuracy'][0]:.3f}-{ci['balanced_accuracy'][1]:.3f}
- ROC-AUC (one-vs-rest) {main['auc_ovr']:.4f}  {ci['auc_ovr'][0]:.4f}-{ci['auc_ovr'][1]:.4f}
- Top-2 accuracy {main['top2_accuracy']:.4f}  {ci['top2_accuracy'][0]:.3f}-{ci['top2_accuracy'][1]:.3f}
- Log-loss {main['log_loss']:.4f} | Brier {main['brier']:.4f} | ECE {main['ece']:.4f} | MCC {main['mcc']:.4f} | Kappa {main['kappa']:.4f}
- Safe mode: confidence >= {tau:.3f} -> {sel_cov*100:.1f}% answered at {sel_acc*100:.2f}% accuracy; the other {100-sel_cov*100:.1f}% are flagged (they would have been {flag_acc*100:.1f}% accurate)

## Repeated-split check ({N_REPEATS} different train/test splits, mean and sd)
{rep_tbl.to_markdown()}
""" + (f"\n{SELECTED} beat the tree stacking ensemble on {wins}/{N_REPEATS} splits (mean gain {diff.mean()*100:+.2f} accuracy points).\n" if wins is not None else "") + f"""
## Accuracy ceiling
If the data follows the independent-markers pattern found in the audit, the best possible accuracy is about {ceiling['ceiling_accuracy']*100:.2f}%
(macro-F1 {ceiling['ceiling_macro_f1']:.4f}). The final model averages {rep_mean_sel*100:.2f}% over the repeated splits = {rep_mean_sel/ceiling['ceiling_accuracy']*100:.1f}% of that ceiling.

## Targets (frozen test set)
| metric | target | achieved | status |
|---|---|---|---|
""" + "\n".join(tgt_rows) + f"""

## Per-disease (frozen test set)
{per_class.round(3).to_markdown(index=False)}

## Most common mistakes
{top_conf.head(8).to_markdown(index=False)}

## Model comparison (same frozen test set)
{comp_df[['accuracy','macro_f1','balanced_accuracy','auc_ovr','log_loss','ece','top2_accuracy']].round(4).to_markdown()}

## Is the proposed model's accuracy significantly different? (McNemar exact test, frozen test set)
{sig_df.to_markdown(index=False)}
""" + ROB_MD + EXTRA_MD
open(f"{OUT}/RESULTS_SUMMARY.md", "w", encoding="utf-8").write(summary)
json.dump(dict(version=SCRIPT_VERSION, final_model=SELECTED, main=main, ci=ci, per_class=per_class.round(4).to_dict("records"),
               safe_mode=dict(threshold=tau, coverage=sel_cov, accuracy=sel_acc, flagged_accuracy=flag_acc),
               repeated_splits={m: {k: float(v) for k, v in rep_sum.loc[m].items()} for m in rep_sum.index},
               ceiling=ceiling, bayes_settings=dict(sub=BAYES_CFG["sub"], alpha=BAYES_CFG["alpha"]), tree_feature_set=BEST_FS,
               temperature=float(bundle["temperature"]), unreadable_values=len(flat_issues), shap_ok=shap_ok,
               cv_winner=CV_WINNER, deployed=SELECTED, robustness_decision=ROBUST_DECISION, sensitivity_lambda=float(bundle["sensitivity_lambda"]),
               runtime_min=(time.time() - T0) / 60, fast_mode=FAST_MODE),
          open(f"{OUT}/paper_results/results.json", "w"), indent=2, default=float)
print("\n" + summary)
base = "/kaggle/working/project_artifacts" if os.path.isdir("/kaggle/working") else os.path.join(os.path.dirname(OUT), "project_artifacts")
shutil.rmtree(f"{OUT}/artifacts/__pycache__", ignore_errors=True); shutil.make_archive(base, "zip", OUT)
log(f"DONE. Download: {base}.zip   (total {(time.time()-T0)/60:.1f} min)")

