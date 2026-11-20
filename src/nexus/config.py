# config.py
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT_DIR / "config" / "models.yaml"

with open(CONFIG_PATH, encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

# LLM setup
LLM_MODEL = CONFIG["llm"]["model"]
MODEL_TEMPERATURE = CONFIG["llm"]["temperature"]
MAX_TOKEN = CONFIG["llm"]["max_tokens"]
TIMEOUT = CONFIG["llm"]["timeout"]
MAX_RETRIES = CONFIG["llm"]["max_retries"]
THINKING_LEVEL = CONFIG["llm"]["thinking_level"]

# Embedding setup
EMBEDDING_MODEL = CONFIG["embedding"]["model"]

# Reranker
RERANKER_MODEL = CONFIG["reranker"]["model"]
