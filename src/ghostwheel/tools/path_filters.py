from pathlib import PurePath


NOISE_DIRECTORY_NAMES = frozenset({".git", ".venv", "__pycache__", "node_modules"})


def glob_matches(path: PurePath, pattern: str) -> bool:
    """Match both an exact path glob and the same glob at any depth."""

    return path.full_match(pattern) or path.full_match(f"**/{pattern}")
