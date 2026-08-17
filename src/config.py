"""Small helper so every script loads config.yaml the same way."""
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    """Turn a project-relative path from config.yaml into an absolute path."""
    return PROJECT_ROOT / path_str
