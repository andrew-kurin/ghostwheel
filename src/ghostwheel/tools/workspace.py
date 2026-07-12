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
) -> tuple[Path, int, os.stat_result]:
    """Resolve and pin a root, rejecting changes during initialization."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and relative_to is not None:
        candidate = relative_to / candidate
    initial_stat = os.stat(candidate)
    before = candidate.resolve(strict=True)
    expected_stat = os.stat(before, follow_symlinks=False)
    if (initial_stat.st_dev, initial_stat.st_ino) != (
        expected_stat.st_dev,
        expected_stat.st_ino,
    ):
        raise RuntimeError(f"Workspace root changed during setup: {candidate}")
    descriptor = _open_absolute_directory(before)
    try:
        first_stat = os.fstat(descriptor)
        if (expected_stat.st_dev, expected_stat.st_ino) != (
            first_stat.st_dev,
            first_stat.st_ino,
        ):
            raise RuntimeError(f"Workspace root changed during setup: {candidate}")
        after = candidate.resolve(strict=True)
        verification_descriptor = _open_absolute_directory(after)
        try:
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
    return before, descriptor, first_stat


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
class _RootIdentity:
    path: Path
    fd: int
    stat: os.stat_result


@dataclass(slots=True)
class _RootLease:
    identity: _RootIdentity
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: bool = True


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    workspace_token: object
    fd: int
    path: WorkspacePath
    stat: os.stat_result


@dataclass(slots=True)
class _DirectoryLease:
    identity: _DirectoryIdentity
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: bool = True


@dataclass(frozen=True, slots=True, eq=False)
class OpenedWorkspaceDirectory:
    _lease: _DirectoryLease = field(repr=False, compare=False)

    @property
    def path(self) -> WorkspacePath:
        return self._lease.identity.path

    @property
    def stat(self) -> os.stat_result:
        return self._lease.identity.stat


@dataclass(frozen=True, slots=True)
class Workspace:
    """Descriptor-pinned filesystem policy shared by file-oriented tools.

    Allowed roots are opened once without traversing symlinks. Operations then
    duplicate that descriptor and open each relative component with ``O_NOFOLLOW``.
    The unrestricted shell is intentionally outside this policy.
    """

    cwd: Path
    filesystem_roots: tuple[Path, ...] = ()
    _root_leases: tuple[_RootLease, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    _directory_token: object = field(
        default_factory=object,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        cwd: Path | str,
        filesystem_roots: Iterable[Path | str] = (),
    ) -> None:
        _require_secure_posix()
        root_leases: list[_RootLease] = []
        try:
            requested_roots = tuple(filesystem_roots)
            if requested_roots:
                canonical_cwd = _canonical_root(cwd)
                for requested_root in requested_roots:
                    root, descriptor, root_stat = _pin_root(
                        requested_root,
                        relative_to=canonical_cwd,
                    )
                    if any(lease.identity.path == root for lease in root_leases):
                        os.close(descriptor)
                        continue
                    root_leases.append(
                        _RootLease(
                            _RootIdentity(
                                path=root,
                                fd=descriptor,
                                stat=root_stat,
                            )
                        )
                    )
            else:
                canonical_cwd, descriptor, root_stat = _pin_root(cwd)
                root_leases.append(
                    _RootLease(
                        _RootIdentity(
                            path=canonical_cwd,
                            fd=descriptor,
                            stat=root_stat,
                        )
                    )
                )
        except BaseException:
            for lease in root_leases:
                os.close(lease.identity.fd)
            raise

        object.__setattr__(self, "cwd", canonical_cwd)
        object.__setattr__(
            self,
            "filesystem_roots",
            tuple(lease.identity.path for lease in root_leases),
        )
        object.__setattr__(self, "_root_leases", tuple(root_leases))
        object.__setattr__(self, "_directory_token", object())

    @property
    def is_closed(self) -> bool:
        leases = getattr(self, "_root_leases", ())
        for lease in leases:
            with lease.lock:
                if lease.active:
                    return False
        return True

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
        lease = next(
            (
                candidate
                for candidate in self._root_leases
                if candidate.identity.path == located.root
            ),
            None,
        )
        if lease is None:
            raise ValueError("Path root is not owned by this Workspace")

        identity = lease.identity
        with lease.lock:
            if not lease.active:
                raise RuntimeError("Workspace is closed")
            try:
                owned_stat = os.fstat(identity.fd)
            except OSError as exc:
                raise RuntimeError(
                    "Workspace root descriptor identity changed"
                ) from exc
            if not self._same_directory(identity.stat, owned_stat):
                raise RuntimeError("Workspace root descriptor identity changed")
            descriptor = os.dup(identity.fd)
            try:
                duplicate_stat = os.fstat(descriptor)
                if not self._same_directory(identity.stat, duplicate_stat):
                    raise RuntimeError("Workspace root descriptor identity changed")
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor

    @staticmethod
    def _same_directory(
        expected: os.stat_result,
        actual: os.stat_result,
    ) -> bool:
        return stat_module.S_ISDIR(actual.st_mode) and (
            actual.st_dev,
            actual.st_ino,
        ) == (
            expected.st_dev,
            expected.st_ino,
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

    def _child_path(
        self,
        parent: _DirectoryIdentity,
        child_name: str,
    ) -> WorkspacePath:
        """Locate one direct child of an already-opened workspace directory."""

        if Path(child_name).parts != (child_name,) or child_name in {".", ".."}:
            raise ValueError(f"Invalid child name: {child_name!r}")
        if parent.path.root not in self.filesystem_roots:
            raise ValueError("Directory handle root is not owned by this Workspace")
        return WorkspacePath(
            root=parent.path.root,
            relative=parent.path.relative / child_name,
            absolute=parent.path.absolute / child_name,
        )

    def _duplicate_directory_fd(
        self,
        directory: OpenedWorkspaceDirectory,
    ) -> tuple[int, _DirectoryIdentity]:
        """Duplicate and verify one live directory lease for an operation."""

        lease = directory._lease
        identity = lease.identity
        if identity.workspace_token is not self._directory_token:
            raise ValueError("Directory handle belongs to a different Workspace")
        with lease.lock:
            if not lease.active:
                raise RuntimeError("Directory handle is no longer active")
            descriptor = os.dup(identity.fd)
        try:
            current_stat = os.fstat(descriptor)
            if not stat_module.S_ISDIR(current_stat.st_mode) or (
                current_stat.st_dev,
                current_stat.st_ino,
            ) != (
                identity.stat.st_dev,
                identity.stat.st_ino,
            ):
                raise RuntimeError("Directory handle identity changed")
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor, identity

    @staticmethod
    def _close_directory(
        directory: OpenedWorkspaceDirectory,
    ) -> None:
        lease = directory._lease
        identity = lease.identity
        with lease.lock:
            if not lease.active:
                return
            lease.active = False
            try:
                current_stat = os.fstat(identity.fd)
            except OSError:
                return
            if not stat_module.S_ISDIR(current_stat.st_mode) or (
                current_stat.st_dev,
                current_stat.st_ino,
            ) != (
                identity.stat.st_dev,
                identity.stat.st_ino,
            ):
                return
            os.close(identity.fd)

    def _open_child_fd(
        self,
        parent: OpenedWorkspaceDirectory,
        child_name: str,
        *,
        directory: bool,
    ) -> tuple[WorkspacePath, int, os.stat_result]:
        parent_fd, parent_identity = self._duplicate_directory_fd(parent)
        try:
            located = self._child_path(parent_identity, child_name)
            flags = os.O_RDONLY | os.O_NOFOLLOW
            if directory:
                flags |= os.O_DIRECTORY
            else:
                flags |= os.O_NONBLOCK
            descriptor = os.open(child_name, flags, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        try:
            child_stat = os.fstat(descriptor)
            expected_type = (
                stat_module.S_ISDIR(child_stat.st_mode)
                if directory
                else stat_module.S_ISREG(child_stat.st_mode)
            )
            if not expected_type:
                kind = "directory" if directory else "regular file"
                raise ValueError(f"Not a {kind}: {located.absolute}")
        except BaseException:
            os.close(descriptor)
            raise
        return located, descriptor, child_stat

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
        try:
            file_stat = os.fstat(fd)
            if not stat_module.S_ISREG(file_stat.st_mode):
                raise ValueError(f"Not a regular file: {located.absolute}")
            file = os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise
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
        try:
            directory_stat = os.fstat(fd)
            opened = OpenedWorkspaceDirectory(
                _lease=_DirectoryLease(
                    _DirectoryIdentity(
                        workspace_token=self._directory_token,
                        fd=fd,
                        path=located,
                        stat=directory_stat,
                    )
                ),
            )
        except BaseException:
            os.close(fd)
            raise
        try:
            yield opened
        finally:
            self._close_directory(opened)

    @contextmanager
    def scan_directory(
        self,
        directory: OpenedWorkspaceDirectory,
    ) -> Iterator[Iterator[os.DirEntry[str]]]:
        """Iterate children of an opened directory without resolving pathnames."""

        descriptor, _identity = self._duplicate_directory_fd(directory)
        try:
            with os.scandir(descriptor) as entries:
                yield entries
        finally:
            os.close(descriptor)

    @contextmanager
    def open_child_file(
        self,
        parent: OpenedWorkspaceDirectory,
        child_name: str,
    ) -> Iterator[OpenedWorkspaceFile]:
        """Open one direct regular-file child without following symlinks."""

        located, descriptor, file_stat = self._open_child_fd(
            parent,
            child_name,
            directory=False,
        )
        try:
            file = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        try:
            yield OpenedWorkspaceFile(file=file, path=located, stat=file_stat)
        finally:
            file.close()

    @contextmanager
    def open_child_directory(
        self,
        parent: OpenedWorkspaceDirectory,
        child_name: str,
    ) -> Iterator[OpenedWorkspaceDirectory]:
        """Open one direct directory child without following symlinks."""

        located, descriptor, directory_stat = self._open_child_fd(
            parent,
            child_name,
            directory=True,
        )
        try:
            opened = OpenedWorkspaceDirectory(
                _lease=_DirectoryLease(
                    _DirectoryIdentity(
                        workspace_token=self._directory_token,
                        fd=descriptor,
                        path=located,
                        stat=directory_stat,
                    )
                ),
            )
        except BaseException:
            os.close(descriptor)
            raise
        try:
            yield opened
        finally:
            self._close_directory(opened)

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
        for lease in getattr(self, "_root_leases", ()):
            identity = lease.identity
            with lease.lock:
                if not lease.active:
                    continue
                lease.active = False
                try:
                    current_stat = os.fstat(identity.fd)
                except OSError:
                    continue
                if not self._same_directory(identity.stat, current_stat):
                    continue
                os.close(identity.fd)

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
