import errno
import fcntl
import os
import platform
import posix
import secrets
import stat as stat_module
import struct
import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

_HAS_OPENAT = os.open in os.supports_dir_fd
_HAS_RENAMEAT = os.rename in os.supports_dir_fd


def _require_secure_posix() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    missing = [name for name in required_flags if not hasattr(os, name)]
    missing_capabilities = list(missing)
    if not _HAS_OPENAT:
        missing_capabilities.append("openat")
    if not _HAS_RENAMEAT:
        missing_capabilities.append("renameat")
    if os.name != "posix" or missing_capabilities:
        details = (
            f"; missing: {', '.join(missing_capabilities)}"
            if missing_capabilities
            else ""
        )
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
class AtomicRewriteResult:
    path: WorkspacePath
    bytes_before: int
    bytes_after: int
    changed: bool
    committed: bool
    durable: bool | None
    warning: str | None = None


class ConcurrentFileChange(RuntimeError):
    """Raised when a file changes while an atomic rewrite is being prepared."""


class AtomicRewriteCancelled(RuntimeError):
    """Raised when cancellation is requested before an atomic rewrite commits."""


_ATOMIC_REWRITE_CHUNK_BYTES = 64 * 1024
_ATOMIC_REWRITE_TEMP_ATTEMPTS = 32

_LINUX_FS_EXTENTS_FL = 0x00080000
_LINUX_FS_VERITY_FL = 0x00100000
_LINUX_LAYOUT_ONLY_FLAGS = (
    0x00040000  # FS_HUGE_FILE_FL
    | _LINUX_FS_EXTENTS_FL
    | 0x00200000  # FS_EA_INODE_FL
    | 0x00400000  # FS_EOFBLOCKS_FL
    | 0x10000000  # FS_INLINE_DATA_FL
)
_LINUX_SAFE_SECURITY_XATTRS = frozenset({"security.selinux"})

_LINUX_LEGACY_IOCTL_PREFIXES = ("alpha", "mips", "powerpc", "ppc", "sparc")
_LINUX_PARISC_IOCTL_PREFIXES = ("hppa", "parisc")
_LINUX_GENERIC_IOCTL_PREFIXES = (
    "aarch64",
    "arc",
    "arm",
    "csky",
    "hexagon",
    "ia64",
    "loongarch",
    "m68k",
    "microblaze",
    "nios2",
    "openrisc",
    "or1k",
    "riscv",
    "s390",
    "sh",
    "xtensa",
)
_LINUX_GENERIC_X86_MACHINES = frozenset(
    {"amd64", "i386", "i486", "i586", "i686", "x86", "x86_64"}
)


def _file_version(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat_module.S_IMODE(value.st_mode),
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        _platform_file_flags(value),
    )


def _platform_file_flags(value: os.stat_result) -> int:
    return int(getattr(value, "st_flags", 0))


def _linux_fs_ioc_getflags_request(machine: str | None = None) -> int:
    """Encode _IOR('f', 1, long) for a known native Linux ioctl ABI."""

    architecture = (machine if machine is not None else platform.machine()).lower()
    if architecture.startswith(_LINUX_LEGACY_IOCTL_PREFIXES):
        size_bits = 13
        read_direction = 2
    elif architecture.startswith(_LINUX_PARISC_IOCTL_PREFIXES):
        size_bits = 14
        read_direction = 1
    elif (
        architecture.startswith(_LINUX_GENERIC_IOCTL_PREFIXES)
        or architecture in _LINUX_GENERIC_X86_MACHINES
    ):
        size_bits = 14
        read_direction = 2
    else:
        display_architecture = architecture or "<empty>"
        raise RuntimeError(
            f"Atomic rewrite does not support Linux ioctl ABI {display_architecture!r}"
        )

    direction_shift = 16 + size_bits
    return (
        (read_direction << direction_shift)
        | (struct.calcsize("@l") << 16)
        | (ord("f") << 8)
        | 1
    )


def _linux_descriptor_file_flags(descriptor: int) -> int:
    buffer = bytearray(struct.calcsize("@l"))
    request = _linux_fs_ioc_getflags_request()
    try:
        fcntl.ioctl(descriptor, request, buffer, True)
    except OSError as exc:
        unsupported_errors = {errno.ENOTTY, errno.EOPNOTSUPP}
        if exc.errno in unsupported_errors:
            return 0
        raise
    int_size = struct.calcsize("@I")
    return int.from_bytes(buffer[:int_size], byteorder=sys.byteorder, signed=False)


def _descriptor_file_flags(
    descriptor: int,
    value: os.stat_result,
) -> int:
    if sys.platform.startswith("linux"):
        return _linux_descriptor_file_flags(descriptor)
    return _platform_file_flags(value)


def _unsupported_atomic_rewrite_flags(flags: int) -> int:
    if sys.platform.startswith("linux"):
        return _unsupported_linux_atomic_rewrite_flags(flags)
    return flags


def _unsupported_linux_atomic_rewrite_flags(flags: int) -> int:
    # These bits describe inode layout and are routinely set by the filesystem
    # itself. All semantic and unknown bits fail closed.
    return flags & ~_LINUX_LAYOUT_ONLY_FLAGS


def _same_file_identity(expected: os.stat_result, actual: os.stat_result) -> bool:
    return stat_module.S_ISREG(actual.st_mode) and (
        actual.st_dev,
        actual.st_ino,
    ) == (
        expected.st_dev,
        expected.st_ino,
    )


def _read_bounded_file(descriptor: int, *, max_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    retained = 0
    while retained <= max_bytes:
        chunk = os.read(
            descriptor,
            min(_ATOMIC_REWRITE_CHUNK_BYTES, max_bytes + 1 - retained),
        )
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        retained += len(chunk)
    raise ValueError(f"File exceeds configured file size limit of {max_bytes} bytes")


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("Atomic rewrite temporary file write made no progress")
        remaining = remaining[written:]


def _copy_darwin_file_metadata(source_fd: int, destination_fd: int) -> None:
    """Copy ACLs/xattrs without restoring stale size or timestamps."""

    fcopyfile = getattr(posix, "_fcopyfile", None)
    acl_flag = getattr(posix, "_COPYFILE_ACL", None)
    xattr_flag = getattr(posix, "_COPYFILE_XATTR", None)
    if fcopyfile is None or acl_flag is None or xattr_flag is None:
        raise RuntimeError(
            "Atomic rewrite cannot preserve macOS ACLs and extended attributes"
        )
    fcopyfile(source_fd, destination_fd, acl_flag | xattr_flag)


def _copy_linux_file_metadata(source_fd: int, destination_fd: int) -> None:
    """Mirror fd-based xattrs, including the POSIX ACL xattr when exposed."""

    operations = {
        name: getattr(os, name, None)
        for name in ("listxattr", "getxattr", "setxattr", "removexattr")
    }
    if any(operation is None for operation in operations.values()):
        raise RuntimeError("Atomic rewrite cannot preserve Linux extended attributes")
    listxattr = operations["listxattr"]
    getxattr = operations["getxattr"]
    setxattr = operations["setxattr"]
    removexattr = operations["removexattr"]
    assert listxattr is not None
    assert getxattr is not None
    assert setxattr is not None
    assert removexattr is not None

    source_names = set(listxattr(source_fd))
    rejected_names = sorted(
        name
        for name in source_names
        if name.startswith("trusted.")
        or (name.startswith("security.") and name not in _LINUX_SAFE_SECURITY_XATTRS)
    )
    if rejected_names:
        raise ValueError(
            "Atomic rewrite rejects privileged/integrity extended attributes: "
            + ", ".join(rejected_names)
        )
    destination_names = set(listxattr(destination_fd))
    for name in sorted(destination_names - source_names):
        removexattr(destination_fd, name)

    missing_attribute_errors = {errno.ENODATA}
    if hasattr(errno, "ENOATTR"):
        missing_attribute_errors.add(errno.ENOATTR)
    for name in sorted(source_names):
        source_value = getxattr(source_fd, name)
        try:
            destination_value = getxattr(destination_fd, name)
        except OSError as exc:
            if exc.errno not in missing_attribute_errors:
                raise
            destination_value = None
        if destination_value != source_value:
            setxattr(destination_fd, name, source_value)


def _copy_atomic_rewrite_metadata(source_fd: int, destination_fd: int) -> None:
    if sys.platform == "darwin":
        _copy_darwin_file_metadata(source_fd, destination_fd)
        return
    if sys.platform.startswith("linux"):
        _copy_linux_file_metadata(source_fd, destination_fd)
        return
    raise RuntimeError(
        f"Atomic rewrite metadata preservation is unsupported on {sys.platform}"
    )


def _create_atomic_rewrite_temp(parent_fd: int) -> tuple[str, int]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    for _attempt in range(_ATOMIC_REWRITE_TEMP_ATTEMPTS):
        name = f".ghostwheel-edit-{secrets.token_hex(16)}.tmp"
        try:
            return name, os.open(name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
    raise FileExistsError("Unable to allocate an atomic rewrite temporary file")


def _unlink_owned_temp(
    parent_fd: int,
    name: str,
    descriptor: int,
) -> None:
    try:
        descriptor_stat = os.fstat(descriptor)
        name_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not _same_file_identity(descriptor_stat, name_stat):
        return
    os.unlink(name, dir_fd=parent_fd)


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
    _mutation_lock: threading.Lock = field(
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
        object.__setattr__(self, "_mutation_lock", threading.Lock())

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

    def atomic_rewrite_regular_file(
        self,
        path: Path | str,
        transform: Callable[[bytes], bytes],
        *,
        max_bytes: int,
        dry_run: bool = False,
        cancelled: Callable[[], bool] | None = None,
    ) -> AtomicRewriteResult:
        """Atomically rewrite an existing regular file below an allowed root.

        The destination and temporary file are accessed relative to one securely
        opened parent directory. Content and identity checks reject changes seen
        before the final descriptor-relative rename. POSIX rename does not offer
        a portable compare-and-swap operation, so a non-cooperating writer can
        still race the final validation and rename. The replacement preserves
        ownership, ordinary mode bits, ACLs, and extended attributes. Hardlinked,
        setid, and platform-flagged files are rejected rather than weakened.
        """
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise ValueError("max_bytes must be a positive integer")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        located = self.locate(path)
        if not located.relative.parts:
            raise IsADirectoryError(str(located.absolute))

        with self._mutation_lock:
            return self._atomic_rewrite_regular_file(
                located,
                transform,
                max_bytes=max_bytes,
                dry_run=dry_run,
                cancelled=cancelled,
            )

    def _atomic_rewrite_regular_file(
        self,
        located: WorkspacePath,
        transform: Callable[[bytes], bytes],
        *,
        max_bytes: int,
        dry_run: bool,
        cancelled: Callable[[], bool] | None,
    ) -> AtomicRewriteResult:
        parent_fd = -1
        target_fd = -1
        temp_fd = -1
        temp_name: str | None = None
        temp_owned = False
        committed = False
        result: AtomicRewriteResult | None = None
        active_error: BaseException | None = None
        post_commit_warnings: list[str] = []
        try:
            parent_fd = self._open_parent_fd(located)
            parent_stat = os.fstat(parent_fd)
            name = located.relative.parts[-1]
            target_fd = os.open(
                name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
            initial_stat = os.fstat(target_fd)
            if not stat_module.S_ISREG(initial_stat.st_mode):
                raise ValueError(f"Not a regular file: {located.absolute}")
            initial_file_flags = _descriptor_file_flags(target_fd, initial_stat)
            if initial_stat.st_nlink != 1:
                raise ValueError(
                    f"Atomic rewrite requires one hard link: {located.absolute}"
                )
            if initial_stat.st_mode & (stat_module.S_ISUID | stat_module.S_ISGID):
                raise ValueError(
                    f"Atomic rewrite rejects setuid/setgid files: {located.absolute}"
                )
            unsupported_file_flags = _unsupported_atomic_rewrite_flags(
                initial_file_flags
            )
            if unsupported_file_flags:
                raise ValueError(
                    "Atomic rewrite rejects files with semantic platform flags "
                    f"({unsupported_file_flags:#x}): {located.absolute}"
                )
            if initial_stat.st_size > max_bytes:
                raise ValueError(
                    f"File exceeds configured file size limit of {max_bytes} bytes"
                )

            original = _read_bounded_file(target_fd, max_bytes=max_bytes)
            after_read_stat = os.fstat(target_fd)
            if _file_version(initial_stat) != _file_version(after_read_stat):
                raise ConcurrentFileChange(
                    f"File changed while it was read: {located.absolute}"
                )

            replacement = transform(original)
            if not isinstance(replacement, bytes):
                raise TypeError("Atomic rewrite transform must return bytes")
            if len(replacement) > max_bytes:
                raise ValueError(
                    "Replacement exceeds configured file size limit of "
                    f"{max_bytes} bytes"
                )
            if replacement == original:
                result = AtomicRewriteResult(
                    path=located,
                    bytes_before=len(original),
                    bytes_after=len(replacement),
                    changed=False,
                    committed=False,
                    durable=None,
                )
            elif dry_run:
                result = AtomicRewriteResult(
                    path=located,
                    bytes_before=len(original),
                    bytes_after=len(replacement),
                    changed=True,
                    committed=False,
                    durable=None,
                )
            else:
                if cancelled is not None and cancelled():
                    raise AtomicRewriteCancelled(
                        f"Atomic rewrite cancelled: {located.absolute}"
                    )
                temp_name, temp_fd = _create_atomic_rewrite_temp(parent_fd)
                temp_owned = True
                _write_all(temp_fd, replacement)
                temporary_stat = os.fstat(temp_fd)
                if (
                    temporary_stat.st_uid,
                    temporary_stat.st_gid,
                ) != (
                    initial_stat.st_uid,
                    initial_stat.st_gid,
                ):
                    os.fchown(temp_fd, initial_stat.st_uid, initial_stat.st_gid)
                _copy_atomic_rewrite_metadata(target_fd, temp_fd)
                os.fchmod(temp_fd, stat_module.S_IMODE(initial_stat.st_mode))
                os.fsync(temp_fd)
                staged_stat = os.fstat(temp_fd)

                try:
                    current = _read_bounded_file(target_fd, max_bytes=max_bytes)
                except ValueError as exc:
                    raise ConcurrentFileChange(
                        f"File changed before atomic rewrite: {located.absolute}"
                    ) from exc
                current_stat = os.fstat(target_fd)
                if current != original or _file_version(current_stat) != _file_version(
                    initial_stat
                ):
                    raise ConcurrentFileChange(
                        f"File changed before atomic rewrite: {located.absolute}"
                    )

                try:
                    name_stat = os.stat(
                        name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    staged_content = _read_bounded_file(
                        temp_fd,
                        max_bytes=max_bytes,
                    )
                    staged_current_stat = os.fstat(temp_fd)
                    temp_name_stat = os.stat(
                        temp_name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise ConcurrentFileChange(
                        f"File changed before atomic rewrite: {located.absolute}"
                    ) from exc
                if _file_version(name_stat) != _file_version(initial_stat):
                    raise ConcurrentFileChange(
                        "File pathname changed before atomic rewrite: "
                        f"{located.absolute}"
                    )
                if (
                    staged_content != replacement
                    or _file_version(staged_current_stat) != _file_version(staged_stat)
                    or _file_version(temp_name_stat) != _file_version(staged_stat)
                ):
                    raise ConcurrentFileChange(
                        "Temporary file changed before atomic rewrite: "
                        f"{located.absolute}"
                    )

                try:
                    verification_parent_fd = self._open_parent_fd(located)
                except OSError as exc:
                    raise ConcurrentFileChange(
                        "Parent directory changed before atomic rewrite: "
                        f"{located.absolute}"
                    ) from exc
                try:
                    current_parent_stat = os.fstat(parent_fd)
                    verification_parent_stat = os.fstat(verification_parent_fd)
                    if not self._same_directory(
                        parent_stat,
                        current_parent_stat,
                    ) or not self._same_directory(
                        parent_stat,
                        verification_parent_stat,
                    ):
                        raise ConcurrentFileChange(
                            "Parent directory changed before atomic rewrite: "
                            f"{located.absolute}"
                        )
                finally:
                    os.close(verification_parent_fd)

                precommit_stat = os.fstat(target_fd)
                precommit_file_flags = _descriptor_file_flags(
                    target_fd,
                    precommit_stat,
                )
                if precommit_file_flags != initial_file_flags:
                    raise ConcurrentFileChange(
                        f"File flags changed before atomic rewrite: {located.absolute}"
                    )
                if _file_version(precommit_stat) != _file_version(initial_stat):
                    raise ConcurrentFileChange(
                        f"File changed before atomic rewrite: {located.absolute}"
                    )
                if cancelled is not None and cancelled():
                    raise AtomicRewriteCancelled(
                        f"Atomic rewrite cancelled: {located.absolute}"
                    )
                os.rename(
                    temp_name,
                    name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                committed = True
                temp_owned = False
                try:
                    os.fsync(parent_fd)
                except OSError as exc:
                    post_commit_warnings.append(f"directory fsync failed: {exc}")
                result = AtomicRewriteResult(
                    path=located,
                    bytes_before=len(original),
                    bytes_after=len(replacement),
                    changed=True,
                    committed=True,
                    durable=not post_commit_warnings,
                    warning=None,
                )
        except BaseException as exc:
            active_error = exc

        cleanup_errors: list[OSError] = []
        if temp_owned and temp_name is not None and temp_fd >= 0:
            try:
                _unlink_owned_temp(parent_fd, temp_name, temp_fd)
            except OSError as exc:
                cleanup_errors.append(exc)
        for label, descriptor in (
            ("temporary file", temp_fd),
            ("original file", target_fd),
            ("parent directory", parent_fd),
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    cleanup_errors.append(OSError(f"closing {label} failed: {exc}"))

        if active_error is not None:
            for cleanup_error in cleanup_errors:
                active_error.add_note(f"Atomic rewrite cleanup failed: {cleanup_error}")
            raise active_error
        if committed:
            post_commit_warnings.extend(str(error) for error in cleanup_errors)
            if post_commit_warnings:
                assert result is not None
                return AtomicRewriteResult(
                    path=result.path,
                    bytes_before=result.bytes_before,
                    bytes_after=result.bytes_after,
                    changed=result.changed,
                    committed=True,
                    durable=False,
                    warning=(
                        "Atomic rewrite committed, but "
                        + "; ".join(post_commit_warnings)
                    ),
                )
        elif cleanup_errors:
            primary_cleanup_error = cleanup_errors[0]
            for cleanup_error in cleanup_errors[1:]:
                primary_cleanup_error.add_note(
                    f"Additional atomic rewrite cleanup failure: {cleanup_error}"
                )
            raise primary_cleanup_error

        if result is None:
            raise RuntimeError("Atomic rewrite completed without a result")
        return result

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
