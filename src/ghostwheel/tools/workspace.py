import os
import stat as stat_module
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

_HAS_OPENAT = os.open in os.supports_dir_fd


def _require_secure_posix() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    missing = [name for name in required_flags if not hasattr(os, name)]
    if os.name != "posix" or missing or not _HAS_OPENAT:
        details = f"; missing: {', '.join(missing)}" if missing else ""
        raise RuntimeError(
            f"Secure workspace access requires POSIX openat/O_NOFOLLOW support{details}"
        )


def _canonical_root(
    path: Path | str,
    *,
    relative_to: Path | None = None,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and relative_to is not None:
        candidate = relative_to / candidate
    return candidate.resolve(strict=True)


def _lexical_absolute(path: Path | str, *, relative_to: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    return Path(os.path.abspath(candidate))


def _open_absolute_directory(path: Path) -> int:
    """Open an absolute directory without following any component symlink."""
    current_fd = os.open(
        path.anchor,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        for part in path.parts[1:]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
    except BaseException:
        os.close(current_fd)
        raise
    return current_fd


def _pin_root(
    path: Path | str,
    *,
    relative_to: Path | None = None,
) -> tuple[Path, int]:
    """Resolve and pin a root, rejecting changes during initialization."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and relative_to is not None:
        candidate = relative_to / candidate
    before = candidate.resolve(strict=True)
    descriptor = _open_absolute_directory(before)
    try:
        after = candidate.resolve(strict=True)
        verification_descriptor = _open_absolute_directory(after)
        try:
            first_stat = os.fstat(descriptor)
            second_stat = os.fstat(verification_descriptor)
            if before != after or (
                first_stat.st_dev,
                first_stat.st_ino,
            ) != (
                second_stat.st_dev,
                second_stat.st_ino,
            ):
                raise RuntimeError(f"Workspace root changed during setup: {candidate}")
        finally:
            os.close(verification_descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return before, descriptor


@dataclass(frozen=True, slots=True)
class WorkspacePath:
    root: Path
    relative: Path
    absolute: Path


@dataclass(frozen=True, slots=True)
class OpenedWorkspaceFile:
    file: BinaryIO
    path: WorkspacePath
    stat: os.stat_result


@dataclass(frozen=True, slots=True)
class OpenedWorkspaceDirectory:
    fd: int
    path: WorkspacePath
    stat: os.stat_result


@dataclass(frozen=True, slots=True)
class Workspace:
    """Descriptor-pinned filesystem policy shared by file-oriented tools.

    Allowed roots are opened once without traversing symlinks. Operations then
    duplicate that descriptor and open each relative component with ``O_NOFOLLOW``.
    The unrestricted shell is intentionally outside this policy.
    """

    cwd: Path
    filesystem_roots: tuple[Path, ...] = ()
    root_descriptors: tuple[tuple[Path, int], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    descriptor_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        cwd: Path | str,
        filesystem_roots: Iterable[Path | str] = (),
    ) -> None:
        _require_secure_posix()
        descriptors: list[tuple[Path, int]] = []
        try:
            requested_roots = tuple(filesystem_roots)
            if requested_roots:
                canonical_cwd = _canonical_root(cwd)
                for requested_root in requested_roots:
                    root, descriptor = _pin_root(
                        requested_root,
                        relative_to=canonical_cwd,
                    )
                    if any(existing_root == root for existing_root, _fd in descriptors):
                        os.close(descriptor)
                        continue
                    descriptors.append((root, descriptor))
            else:
                canonical_cwd, descriptor = _pin_root(cwd)
                descriptors.append((canonical_cwd, descriptor))
        except BaseException:
            for _root, descriptor in descriptors:
                os.close(descriptor)
            raise

        object.__setattr__(self, "cwd", canonical_cwd)
        object.__setattr__(
            self,
            "filesystem_roots",
            tuple(root for root, _descriptor in descriptors),
        )
        object.__setattr__(self, "root_descriptors", tuple(descriptors))
        object.__setattr__(self, "descriptor_lock", threading.Lock())

    def locate(self, path: Path | str) -> WorkspacePath:
        absolute = _lexical_absolute(path, relative_to=self.cwd)
        matching_roots = [
            root for root in self.filesystem_roots if absolute.is_relative_to(root)
        ]
        if not matching_roots:
            raise ValueError(
                f"Path {absolute} is outside allowed roots (filesystem_roots)"
            )
        root = max(matching_roots, key=lambda item: len(item.parts))
        return WorkspacePath(
            root=root,
            relative=absolute.relative_to(root),
            absolute=absolute,
        )

    def display_path(self, path: Path | str) -> str:
        located = self.locate(path)
        try:
            return str(located.absolute.relative_to(self.cwd))
        except ValueError:
            return str(located.absolute)

    def _duplicate_root(self, located: WorkspacePath) -> int:
        with self.descriptor_lock:
            if not self.root_descriptors:
                raise RuntimeError("Workspace is closed")
            return os.dup(
                next(
                    descriptor
                    for root, descriptor in self.root_descriptors
                    if root == located.root
                )
            )

    def _open_parent_fd(self, located: WorkspacePath) -> int:
        current_fd = self._duplicate_root(located)
        parts = located.relative.parts
        try:
            for part in parts[:-1]:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current_fd,
                )
                os.close(current_fd)
                current_fd = next_fd
        except BaseException:
            os.close(current_fd)
            raise
        return current_fd

    def _open_fd(self, located: WorkspacePath, *, directory: bool) -> int:
        parts = located.relative.parts
        if not parts:
            if directory:
                return self._duplicate_root(located)
            raise IsADirectoryError(str(located.absolute))

        current_fd = self._open_parent_fd(located)
        try:
            final_flags = os.O_RDONLY | os.O_NOFOLLOW
            if directory:
                final_flags |= os.O_DIRECTORY
            else:
                final_flags |= os.O_NONBLOCK
            final_fd = os.open(parts[-1], final_flags, dir_fd=current_fd)
        finally:
            os.close(current_fd)
        return final_fd

    def stat_path(self, path: Path | str) -> os.stat_result:
        located = self.locate(path)
        if not located.relative.parts:
            descriptor = self._duplicate_root(located)
            try:
                return os.fstat(descriptor)
            finally:
                os.close(descriptor)
        parent_fd = self._open_parent_fd(located)
        try:
            return os.stat(
                located.relative.parts[-1],
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(parent_fd)

    @contextmanager
    def open_file(self, path: Path | str) -> Iterator[OpenedWorkspaceFile]:
        located = self.locate(path)
        fd = self._open_fd(located, directory=False)
        file_stat = os.fstat(fd)
        if not stat_module.S_ISREG(file_stat.st_mode):
            os.close(fd)
            raise ValueError(f"Not a regular file: {located.absolute}")
        file = os.fdopen(fd, "rb")
        try:
            yield OpenedWorkspaceFile(file=file, path=located, stat=file_stat)
        finally:
            file.close()

    @contextmanager
    def open_directory(
        self,
        path: Path | str,
    ) -> Iterator[OpenedWorkspaceDirectory]:
        located = self.locate(path)
        fd = self._open_fd(located, directory=True)
        directory_stat = os.fstat(fd)
        try:
            yield OpenedWorkspaceDirectory(
                fd=fd,
                path=located,
                stat=directory_stat,
            )
        finally:
            os.close(fd)

    def is_file(self, path: Path | str) -> bool:
        try:
            return stat_module.S_ISREG(self.stat_path(path).st_mode)
        except OSError:
            return False

    def is_directory(self, path: Path | str) -> bool:
        try:
            return stat_module.S_ISDIR(self.stat_path(path).st_mode)
        except OSError:
            return False

    def close(self) -> None:
        lock = getattr(self, "descriptor_lock", None)
        if lock is None:
            return
        with lock:
            descriptors = getattr(self, "root_descriptors", ())
            object.__setattr__(self, "root_descriptors", ())
        for _root, descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def __del__(self) -> None:
        self.close()

    def __copy__(self) -> "Workspace":
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> "Workspace":
        return self

    def __reduce_ex__(self, _protocol: int):
        raise TypeError(
            "Workspace instances own live descriptors and cannot be pickled"
        )
