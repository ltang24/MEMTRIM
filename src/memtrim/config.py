import os

MEMTRIM_BASE_URL = os.environ.get("MEMTRIM_BASE_URL", "http://localhost:8000/v1")
MEMTRIM_MODEL = os.environ.get("MEMTRIM_MODEL", "qwen")
MEMTRIM_MODEL_HF_NAME = os.environ.get("MEMTRIM_MODEL_HF_NAME", "Qwen/Qwen2.5-7B-Instruct")
MEMTRIM_ENCODER_DEVICE = os.environ.get("MEMTRIM_ENCODER_DEVICE", "cpu")
ENCODER_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RIPPLEEDITS_DIR = os.path.join(PROJECT_ROOT, "RippleEdits", "data", "benchmark")
CACHE_DB_PATH = os.path.join(DATA_DIR, "llm_cache.sqlite3")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
MEM0_STORE_DIR = os.path.join(DATA_DIR, "mem0_stores")
FAKE_API_KEY = "sk-memtrim"
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
ERROR_LOG_PATH = os.path.join(LOGS_DIR, "errors.jsonl")

DEFAULT_SEED = 42
DEFAULT_N_PER_CONDITION = 200
DEFAULT_TOP_K = 5
DEFAULT_MAX_STORE_SIZE = 20
DEFAULT_WORKERS = 16
MAX_UNCHANGED_FACTS = 3

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MEM0_STORE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
