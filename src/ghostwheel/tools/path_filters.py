from dataclasses import dataclass
import glob as glob_module
from pathlib import PurePath, PurePosixPath
import re


NOISE_DIRECTORY_NAMES = frozenset({".git", ".venv", "__pycache__", "node_modules"})


def glob_matches(path: PurePath, pattern: str) -> bool:
    """Match both an exact path glob and the same glob at any depth."""

    return path.full_match(pattern) or path.full_match(f"**/{pattern}")


@dataclass(frozen=True, slots=True)
class CompiledPathGlob:
    patterns: tuple[re.Pattern[str], ...]

    @classmethod
    def compile(cls, pattern: str) -> "CompiledPathGlob":
        normalized = PurePosixPath(pattern).as_posix()
        variants = (normalized, f"**/{normalized}")
        return cls(
            tuple(
                re.compile(
                    glob_module.translate(
                        variant,
                        recursive=True,
                        include_hidden=True,
                        seps="/",
                    )
                )
                for variant in variants
            )
        )

    def matches(self, path: PurePath | str) -> bool:
        value = path if isinstance(path, str) else path.as_posix()
        return any(pattern.fullmatch(value) is not None for pattern in self.patterns)
