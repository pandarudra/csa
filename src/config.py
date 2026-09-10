"""Central configuration: paths, environment, and tunable constants.

Every other module reads settings from here instead of touching `os.environ`
or building paths ad hoc, so the whole pipeline can be reconfigured (a
different model, a different sample size, a different machine's paths)
from one place.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, here, at import time. This is a local file read, not a
# network/API call, so it is safe to run even without GEMINI_API_KEY set --
# modules that actually need the key check for it lazily (see llm_client.py).
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Target brand -----------------------------------------------------------
# The brand handle as it appears in the TWCS `author_id` column for brand
# (non-inbound) rows. Fixed for this assignment, not meant to vary per run.
BRAND_HANDLE = "SpotifyCares"

# --- Data paths --------------------------------------------------------------
DATA_PATH = Path(
    os.environ.get(
        "DATA_PATH", REPO_ROOT / "dataset/customer-support-on-twitter/twcs/twcs.csv"
    )
)
PROCESSED_DATA_PATH = Path(
    os.environ.get("PROCESSED_DATA_PATH", REPO_ROOT / "data/processed/spotify_threads.jsonl")
)
INTENTS_PATH = Path(os.environ.get("INTENTS_PATH", REPO_ROOT / "intents.yaml"))
GOLDEN_SET_PATH = Path(os.environ.get("GOLDEN_SET_PATH", REPO_ROOT / "golden/golden_set.csv"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results"))

# --- NVIDIA NIM (OpenAI-compatible endpoint) ---------------------------------
# Switched from Gemini after discovering its free-tier generateContent quota
# is 5 requests/minute for this account -- unworkable for an eval harness
# that needs a few hundred chat calls (see DECISIONS.md). NVIDIA's build.nvidia.com
# NIM catalog exposes hosted open models through an OpenAI-compatible API with
# a much more generous free tier (measured: 8+ sequential calls with no
# throttling; moderate concurrency is safe, high concurrency (~8 parallel)
# starts producing 429s/timeouts).
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Pinned to specific model names (not a "-latest" alias) for reproducibility.
# Verified live against the configured API key on 2026-09-10 (see
# DECISIONS.md) -- most of the ~80 models in NIM's public catalog return 404
# "not found for account" for this particular free-tier key; these two are
# confirmed working.
NVIDIA_CHAT_MODEL = os.environ.get("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NVIDIA_EMBEDDING_MODEL = os.environ.get("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
NVIDIA_EMBEDDING_DIM = int(os.environ.get("NVIDIA_EMBEDDING_DIM", "2048"))
# Concurrency (not a per-minute item count, unlike the old Gemini limiter):
# empirically this account tolerates a handful of simultaneous requests
# fine but starts throwing 429s/timeouts well before 8-way concurrency.
NVIDIA_MAX_CONCURRENT_REQUESTS = int(os.environ.get("NVIDIA_MAX_CONCURRENT_REQUESTS", "4"))

# --- Reproducibility ----------------------------------------------------------
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

# --- Data prep -----------------------------------------------------------------
# The full TWCS export contains ~3M rows; we only need a subsample of
# SpotifyCares-adjacent threads for a runnable, gradeable pipeline (the
# assignment explicitly expects and encourages subsampling).
TARGET_THREAD_SAMPLE_SIZE = int(os.environ.get("TARGET_THREAD_SAMPLE_SIZE", "4000"))
MAX_THREAD_WALK_DEPTH = int(os.environ.get("MAX_THREAD_WALK_DEPTH", "20"))
# A handful of "threads" balloon to hundreds of turns because their root is
# a viral broadcast tweet that many unrelated customers replied to -- not a
# real 1:1 support conversation. Root-grouping can't distinguish that case
# from genuine multi-turn support, so we cap thread size and drop outliers
# instead (measured: 99.8% of reconstructed threads have <= 12 turns).
MAX_THREAD_TURNS = int(os.environ.get("MAX_THREAD_TURNS", "12"))

# --- Retrieval -----------------------------------------------------------------
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "4"))

# --- Golden set ------------------------------------------------------------------
GOLDEN_SET_TARGET_SIZE = int(os.environ.get("GOLDEN_SET_TARGET_SIZE", "200"))
GOLDEN_SEED_LABEL_COUNT = int(os.environ.get("GOLDEN_SEED_LABEL_COUNT", "45"))
GOLDEN_DEV_FRACTION = float(os.environ.get("GOLDEN_DEV_FRACTION", "0.25"))
