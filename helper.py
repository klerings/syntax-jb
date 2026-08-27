from functools import lru_cache
import json
from pathlib import Path

META_FILE = Path(__file__).parent / "meta_info.json"


@lru_cache
def get_variants() -> list[str]:
    with open(META_FILE) as f:
        return json.load(f)["syntax_variants"]


@lru_cache
def get_coefficients() -> list[str]:
    with open(META_FILE) as f:
        return json.load(f)["steering_coefficients"]


@lru_cache
def get_boundary_variant() -> list[str]:
    with open(META_FILE) as f:
        return json.load(f)["boundary_prompts_variant"]


@lru_cache
def get_valid_models() -> list[str]:
    with open(META_FILE) as f:
        return json.load(f)["valid_models_for_steering"]
