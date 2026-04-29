"""
Global configuration for the Fashion AI Engine project.

All paths, constants and global parameters live here so they're
imported once instead of being hardcoded across the codebase.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

CONFIGS_DIR = PROJECT_ROOT / "configs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# ---------------------------------------------------------------------------
# H&M raw file names
# ---------------------------------------------------------------------------
ARTICLES_FILE = RAW_DIR / "articles.csv"
CUSTOMERS_FILE = RAW_DIR / "customers.csv"
TRANSACTIONS_FILE = RAW_DIR / "transactions_train.csv"

# ---------------------------------------------------------------------------
# Temporal split (the H&M dataset spans 2018-09-20 to 2020-09-22)
# ---------------------------------------------------------------------------
TRAIN_END = "2020-06-22"   # 21 months for training
VAL_END = "2020-08-22"     # 2 months for validation
TEST_END = "2020-09-22"    # 1 month for test (final business simulation)

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Industry benchmarks for return rates by category (sources cited in README)
# Used to simulate realistic return labels since H&M dataset doesn't include them.
# References: Narvar 2023 Returns Report, Optoro Returns Industry Report.
# ---------------------------------------------------------------------------
RETURN_RATES_BY_CATEGORY = {
    "Dresses":             0.35,
    "Trousers":            0.30,
    "Jeans":               0.30,
    "Shirts":              0.25,
    "Tops":                0.22,
    "Knitwear":            0.20,
    "Skirts":              0.25,
    "Outdoor":             0.20,
    "Shoes":               0.25,
    "Bags":                0.08,
    "Accessories":         0.05,
    "Underwear":           0.10,
    "Swimwear":            0.20,
    "Nightwear":           0.10,
    "Socks & Tights":      0.05,
    "Default":             0.18,  # fallback for uncategorized items
}

# ---------------------------------------------------------------------------
# Economic assumptions (cited from H&M and Inditex annual reports)
# ---------------------------------------------------------------------------
GROSS_MARGIN_PCT = 0.53            # H&M reports ~52-55% gross margin
RETURN_HANDLING_COST_EUR = 18.0    # Optoro/Narvar benchmark per returned item
MARKDOWN_DISCOUNT_PCT = 0.40       # average markdown applied to unsold stock
UNSOLD_DESTRUCTION_RATE = 0.15     # % of unsold stock that ends up destroyed/donated


# ---------------------------------------------------------------------------
# Helper to ensure required dirs exist on import
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create all required directories if missing."""
    for d in (
        RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR,
        FIGURES_DIR, RESULTS_DIR, CONFIGS_DIR, MLRUNS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
