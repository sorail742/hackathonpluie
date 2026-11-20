"""Central configuration and constants for the IndabaX Guinea 2026 Weather Prediction Challenge."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
MODELS_DIR = BASE_DIR / "models_checkpoints"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
VARIABLE_DEFINITIONS_PATH = DATA_DIR / "VariableDefinitions.csv"

# Optional external files (flagged as requested by the user)
STATIONS_FILE = DATA_DIR / "stations.csv"
REGIONAL_DAILY_FILE = DATA_DIR / "regional_daily.csv"
CLIMAT_NATIONAL_FILE = DATA_DIR / "climat_national.csv"

# Flag to use extra files if present, fallback cleanly if absent
USE_EXTRA_FILES = all(p.exists() for p in [STATIONS_FILE, REGIONAL_DAILY_FILE, CLIMAT_NATIONAL_FILE])

# Target classes
CLASSES = ["aucune", "faible", "moderee", "forte"]
TARGET_MAP = {name: i for i, name in enumerate(CLASSES)}
REV_TARGET_MAP = {i: name for i, name in enumerate(CLASSES)}

# Reproducibility
RANDOM_SEED = 42
SEEDS = [42, 1337, 2026]

# Canonical 20 Guinean cities
# 18 in train, 2 unseen in test (Labé and Siguiri)
CANONICAL_TRAIN_CITIES = [
    "Boké", "Conakry", "Dabola", "Dalaba", "Faranah",
    "Forécariah", "Fria", "Guéckédou", "Kamsar", "Kankan",
    "Kindia", "Kissidougou", "Kouroussa", "Macenta", "Mamou",
    "Nzérékoré", "Pita", "Télimélé"
]

CANONICAL_TEST_UNSEEN_CITIES = ["Labé", "Siguiri"]

CANONICAL_CITIES = sorted(CANONICAL_TRAIN_CITIES + CANONICAL_TEST_UNSEEN_CITIES)

# Explicit normalization alias dictionary
CITY_ALIAS_MAP = {
    # Boké
    "boke": "Boké",
    "boké": "Boké",
    # Conakry
    "conakry": "Conakry",
    "conakry ": "Conakry",
    # Dabola
    "dabola": "Dabola",
    # Dalaba
    "dalaba": "Dalaba",
    # Faranah
    "faranah": "Faranah",
    # Forécariah
    "forecariah": "Forécariah",
    "forécariah": "Forécariah",
    # Fria
    "fria": "Fria",
    # Guéckédou
    "gueckedou": "Guéckédou",
    "gueckédou": "Guéckédou",
    "guéckédou": "Guéckédou",
    # Kamsar
    "kamsar": "Kamsar",
    # Kankan
    "kankan": "Kankan",
    "kankan ": "Kankan",
    # Kindia
    "kindia": "Kindia",
    # Kissidougou
    "kissi": "Kissidougou",
    "kissidougou": "Kissidougou",
    # Kouroussa
    "kouroussa": "Kouroussa",
    # Labé (test unseen)
    "labe": "Labé",
    "labé": "Labé",
    "labe ": "Labé",
    # Macenta
    "macenta": "Macenta",
    # Mamou
    "mamou": "Mamou",
    # Nzérékoré
    "n'zérékoré": "Nzérékoré",
    "nzerekore": "Nzérékoré",
    "nzérékoré": "Nzérékoré",
    # Pita
    "pita": "Pita",
    # Siguiri (test unseen)
    "siguiri": "Siguiri",
    # Télimélé
    "telimele": "Télimélé",
    "télimélé": "Télimélé",
}
