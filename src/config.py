import os

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# LLM PROVIDER
# ============================================================

# Supported:
#   groq
#   ollama
#
# Groq is the primary provider.
# Ollama is the automatic fallback.
#
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "groq",
).lower()


# ============================================================
# MODELS
# ============================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.8-27b",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)


# ============================================================
# OLLAMA
# ============================================================

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)


# ============================================================
# GENERATION SETTINGS
# ============================================================

LLM_TEMPERATURE = float(
    os.getenv("LLM_TEMPERATURE", "0.1")
)

LLM_MAX_TOKENS = int(
    os.getenv("LLM_MAX_TOKENS", "512")
)

LLM_TIMEOUT = int(
    os.getenv("LLM_TIMEOUT", "120")
)


# ============================================================
# PROVIDER FALLBACK
# ============================================================

PROVIDER_ORDER = {
    "groq": [
        "groq",
        "ollama",
    ],
    "ollama": [
        "ollama",
    ],
}


def get_provider_order():
    """
    Return the provider order based on the configured
    primary provider.
    """

    if LLM_PROVIDER not in PROVIDER_ORDER:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. "
            f"Choose from: "
            f"{', '.join(PROVIDER_ORDER.keys())}"
        )

    return PROVIDER_ORDER[LLM_PROVIDER]