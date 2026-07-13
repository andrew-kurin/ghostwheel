import copy
from dataclasses import replace
import errno
import os
from pathlib import Path
import stat as stat_module
import subprocess
import sys
from types import SimpleNamespace

import pytest
import ghostwheel.tools.search as search_module
import ghostwheel.tools.workspace as workspace_module

from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools.filesystem import ls, read
from ghostwheel.tools.search import GrepIncompleteReason, grep
from ghostwheel.tools.workspace import (
    AtomicRewriteCancelled,
    AtomicRewriteResult,
    ConcurrentFileChange,
    Workspace,
)

from .support import grep_metadata, listing_metadata, read_metadata, read_rows, tool_ctx


def test_relative_filesystem_roots_are_resolved_from_tool_cwd(tmp_path: Path) -> None:
    cwd = (tmp_path / "repo").resolve()
    shared = cwd / "shared"
    shared.mkdir(parents=True)
    (shared / "value.txt").write_text("value", encoding="utf-8")
    ctx = SimpleNamespace(deps=ToolDeps(cwd=cwd, filesystem_roots=["shared"]))

    result = read(ctx, str(shared / "value.txt"))

    assert read_metadata(result).path == "shared/value.txt"


def test_recursive_tools_support_nested_allowed_roots(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    (inner / "needle.txt").write_text("needle", encoding="utf-8")
    deps = ToolDeps(cwd=outer, filesystem_roots=[outer, inner])
    try:
        ctx = SimpleNamespace(deps=deps)

        listing = listing_metadata(ls(ctx, depth=2))
        search = grep_metadata(grep(ctx, "needle"))
    finally:
        deps.close()

    assert [entry.name for entry in listing.entries] == [
        "inner",
        "inner/needle.txt",
    ]
    assert listing.complete is True
    assert [match.file for match in search.matches] == ["inner/needle.txt"]
    assert search.complete is True


def test_filesystem_tools_reject_paths_outside_allowed_roots(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    ctx = tool_ctx(root)

    with pytest.raises(ValueError, match="outside allowed roots"):
        read(ctx, "../outside.txt")

    with pytest.raises(ValueError, match="outside allowed roots"):
        ls(ctx, "..")


def test_filesystem_tools_never_traverse_symlinked_parent_directories(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    outside = (tmp_path / "outside").resolve()
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    ctx = tool_ctx(root)

    with pytest.raises(OSError):
        read(ctx, "linked/secret.txt")
    with pytest.raises(OSError):
        ls(ctx, "linked")

    assert grep_metadata(grep(ctx, "SECRET")).matches == []


def test_read_is_safe_when_parent_is_swapped_after_descriptor_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "repo").resolve()
    safe = root / "safe"
    outside = (tmp_path / "outside").resolve()
    safe.mkdir(parents=True)
    outside.mkdir()
    (safe / "value.txt").write_text("SAFE", encoding="utf-8")
    (outside / "value.txt").write_text("SECRET", encoding="utf-8")
    original_open = os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "value.txt" and not swapped:
            swapped = True
            safe.rename(root / "original-safe")
            safe.symlink_to(outside, target_is_directory=True)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace_module.os, "open", racing_open)

    result = read(tool_ctx(root), "safe/value.txt")

    assert read_rows(result) == [(1, "SAFE")]
    assert "SECRET" not in result.return_value


def test_workspace_pins_allowed_root_across_ancestor_swap(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    root = parent / "repo"
    outside_parent = tmp_path / "outside-parent"
    outside_root = outside_parent / "repo"
    root.mkdir(parents=True)
    outside_root.mkdir(parents=True)
    (root / "value.txt").write_text("SAFE", encoding="utf-8")
    (outside_root / "value.txt").write_text("SECRET", encoding="utf-8")
    ctx = tool_ctx(root)
    parent.rename(tmp_path / "original-parent")
    parent.symlink_to(outside_parent, target_is_directory=True)

    result = read(ctx, "value.txt")

    assert read_rows(result) == [(1, "SAFE")]
    assert "SECRET" not in result.return_value


def test_workspace_initialization_rejects_ancestor_symlink_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "allowed-parent-race"
    root = parent / "repo"
    outside_parent = tmp_path / "outside-parent-race"
    (outside_parent / "repo").mkdir(parents=True)
    root.mkdir(parents=True)
    original_open = os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and path == parent.name:
            swapped = True
            parent.rename(tmp_path / "original-allowed-parent")
            parent.symlink_to(outside_parent, target_is_directory=True)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace_module.os, "open", racing_open)

    with pytest.raises(OSError):
        tool_ctx(root)


def test_workspace_initialization_rejects_real_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "allowed-parent-replacement"
    root = parent / "repo"
    replacement_parent = tmp_path / "replacement-parent"
    replacement_root = replacement_parent / "repo"
    root.mkdir(parents=True)
    replacement_root.mkdir(parents=True)
    (root / "value.txt").write_text("SAFE", encoding="utf-8")
    (replacement_root / "value.txt").write_text("SECRET", encoding="utf-8")
    original_open = os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and path == parent.name:
            swapped = True
            parent.rename(tmp_path / "original-allowed-parent")
            replacement_parent.rename(parent)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace_module.os, "open", racing_open)

    with pytest.raises(RuntimeError, match="Workspace root changed during setup"):
        Workspace(root)


def test_closed_workspace_never_reuses_or_recloses_descriptor_numbers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "value.txt").write_text("SAFE", encoding="utf-8")
    workspace = Workspace(root)
    workspace.close()
    outside_fd = os.open(outside, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RuntimeError, match="Workspace is closed"):
            with workspace.open_file("value.txt"):
                pass
        workspace.close()
        os.fstat(outside_fd)
    finally:
        os.close(outside_fd)


def test_workspace_root_descriptors_are_private_and_rebinding_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "value.txt").write_text("SAFE", encoding="utf-8")
    (outside / "value.txt").write_text("SECRET", encoding="utf-8")
    workspace = Workspace(root)

    assert not hasattr(workspace, "root_descriptors")
    assert not hasattr(workspace, "descriptor_lock")

    # Deliberately breach the private lease to simulate a corrupted descriptor.
    owned_fd = workspace._root_leases[0].identity.fd
    outside_fd = os.open(outside, os.O_RDONLY | os.O_DIRECTORY)
    os.dup2(outside_fd, owned_fd)
    try:
        with pytest.raises(RuntimeError, match="root descriptor identity changed"):
            with workspace.open_file("value.txt"):
                pass

        workspace.close()

        # Identity-checked cleanup must not close the foreign replacement.
        os.fstat(owned_fd)
    finally:
        os.close(owned_fd)
        os.close(outside_fd)


def test_workspace_rejects_a_rebound_duplicate_of_a_valid_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "value.txt").write_text("SAFE", encoding="utf-8")
    (outside / "value.txt").write_text("SECRET", encoding="utf-8")
    workspace = Workspace(root)
    owned_fd = workspace._root_leases[0].identity.fd
    outside_fd = os.open(outside, os.O_RDONLY | os.O_DIRECTORY)
    real_dup = os.dup
    foreign_duplicates: list[int] = []

    def replacing_dup(descriptor: int) -> int:
        if descriptor == owned_fd:
            duplicate = real_dup(outside_fd)
            foreign_duplicates.append(duplicate)
            return duplicate
        return real_dup(descriptor)

    try:
        monkeypatch.setattr(workspace_module.os, "dup", replacing_dup)

        with pytest.raises(RuntimeError, match="root descriptor identity changed"):
            with workspace.open_file("value.txt"):
                pass

        assert foreign_duplicates
        with pytest.raises(OSError):
            os.fstat(foreign_duplicates[-1])
    finally:
        os.close(outside_fd)
        workspace.close()


def test_workspace_copies_share_one_descriptor_owner(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    workspace = Workspace(root)

    assert copy.copy(workspace) is workspace
    assert copy.deepcopy(workspace) is workspace


def test_grep_is_safe_when_a_file_is_replaced_by_a_symlink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside.txt"
    root.mkdir()
    (root / "safe.txt").write_text("SAFE", encoding="utf-8")
    outside.write_text("SECRET", encoding="utf-8")
    ctx = tool_ctx(root)
    original_open = search_module.os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "safe.txt" and not swapped:
            swapped = True
            (root / "safe.txt").rename(root / "original-safe.txt")
            (root / "safe.txt").symlink_to(outside)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(search_module.os, "open", racing_open)

    search = grep_metadata(grep(ctx, "SECRET", file_glob="*.txt"))

    assert search.matches == []
    assert search.files_skipped == 1
    assert search.reasons == [GrepIncompleteReason.FILE_ERROR]


def test_grep_is_safe_when_a_directory_is_replaced_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    safe = root / "safe"
    outside = tmp_path / "outside"
    safe.mkdir(parents=True)
    outside.mkdir()
    (safe / "value.txt").write_text("SAFE", encoding="utf-8")
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    ctx = tool_ctx(root)
    original_open = search_module.os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "safe" and not swapped:
            swapped = True
            safe.rename(root / "original-safe")
            safe.symlink_to(outside, target_is_directory=True)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(search_module.os, "open", racing_open)

    search = grep_metadata(grep(ctx, "SECRET"))

    assert search.matches == []
    assert search.reasons == [GrepIncompleteReason.ENTRY_ERROR]


def test_grep_continues_after_an_unreadable_subtree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a-blocked").mkdir()
    (tmp_path / "z-visible").mkdir()
    (tmp_path / "z-visible" / "value.txt").write_text(
        "needle",
        encoding="utf-8",
    )
    ctx = tool_ctx(tmp_path)
    original_open = search_module.os.open

    def denying_open(path: object, *args: object, **kwargs: object) -> int:
        if path == "a-blocked":
            raise PermissionError("blocked by test")
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(search_module.os, "open", denying_open)

    search = grep_metadata(grep(ctx, "needle"))

    assert [match.file for match in search.matches] == ["z-visible/value.txt"]
    assert search.reasons == [GrepIncompleteReason.ENTRY_ERROR]


def test_workspace_child_primitives_scan_and_open_direct_children(
    tmp_path: Path,
) -> None:
    (tmp_path / "child").mkdir()
    (tmp_path / "value.txt").write_text("value", encoding="utf-8")
    workspace = Workspace(tmp_path)
    try:
        with workspace.open_directory(".") as root:
            assert not hasattr(root, "fd")
            with workspace.scan_directory(root) as entries:
                assert sorted(entry.name for entry in entries) == ["child", "value.txt"]
            with workspace.open_child_file(root, "value.txt") as opened_file:
                assert opened_file.file.read() == b"value"
                assert opened_file.path.absolute == tmp_path / "value.txt"
            with workspace.open_child_directory(root, "child") as opened_directory:
                assert opened_directory.path.absolute == tmp_path / "child"
    finally:
        workspace.close()


def test_workspace_child_primitives_reject_stale_reused_directory_handles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    (outside / "nested").mkdir()
    workspace = Workspace(root)
    try:
        with workspace.open_directory(".") as opened:
            stale = opened

        stale_fd = stale._lease.identity.fd
        replacement_fd = os.open(outside, os.O_RDONLY | os.O_DIRECTORY)
        if replacement_fd != stale_fd:
            os.dup2(replacement_fd, stale_fd)
            os.close(replacement_fd)
            replacement_fd = stale_fd
        try:
            with pytest.raises(RuntimeError, match="no longer active"):
                with workspace.scan_directory(stale) as entries:
                    list(entries)
            with pytest.raises(RuntimeError, match="no longer active"):
                with workspace.open_child_file(stale, "secret.txt"):
                    pass
            with pytest.raises(RuntimeError, match="no longer active"):
                with workspace.open_child_directory(stale, "nested"):
                    pass
        finally:
            os.close(replacement_fd)
    finally:
        workspace.close()


def test_workspace_child_primitives_reject_foreign_directory_handles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    (outside / "nested").mkdir()
    workspace = Workspace(root)
    foreign_workspace = Workspace(outside)
    try:
        with foreign_workspace.open_directory(".") as foreign:
            with pytest.raises(ValueError, match="different Workspace"):
                with workspace.scan_directory(foreign) as entries:
                    list(entries)
            with pytest.raises(ValueError, match="different Workspace"):
                with workspace.open_child_file(foreign, "secret.txt"):
                    pass
            with pytest.raises(ValueError, match="different Workspace"):
                with workspace.open_child_directory(foreign, "nested"):
                    pass
    finally:
        foreign_workspace.close()
        workspace.close()


def test_workspace_child_primitives_reject_a_rebound_live_descriptor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    workspace = Workspace(root)
    outside_fd = os.open(outside, os.O_RDONLY | os.O_DIRECTORY)
    rebound_fd = -1
    try:
        with workspace.open_directory(".") as opened:
            rebound_fd = opened._lease.identity.fd
            os.dup2(outside_fd, rebound_fd)

            with pytest.raises(RuntimeError, match="identity changed"):
                with workspace.scan_directory(opened) as entries:
                    list(entries)
            with pytest.raises(RuntimeError, match="identity changed"):
                with workspace.open_child_file(opened, "secret.txt"):
                    pass
    finally:
        if rebound_fd >= 0:
            os.close(rebound_fd)
        os.close(outside_fd)
        workspace.close()


def test_workspace_directory_identity_cannot_be_replaced_on_a_live_handle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "secret.txt").write_text("SAFE", encoding="utf-8")
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    workspace = Workspace(root)
    outside_fd = os.open(outside, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with workspace.open_directory(".") as opened:
            with pytest.raises(TypeError):
                replace(
                    opened,
                    fd=outside_fd,
                    stat=os.fstat(outside_fd),
                    path=workspace.locate("."),
                )

            with workspace.open_child_file(opened, "secret.txt") as child:
                assert child.file.read() == b"SAFE"
    finally:
        os.close(outside_fd)
        workspace.close()


def test_workspace_context_exit_does_not_close_a_reused_foreign_descriptor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    workspace = Workspace(root)
    replacement_fd = -1
    try:
        with workspace.open_directory(".") as opened:
            owned_fd = opened._lease.identity.fd
            os.close(owned_fd)
            replacement_fd = os.open(outside, os.O_RDONLY | os.O_DIRECTORY)
            if replacement_fd != owned_fd:
                os.dup2(replacement_fd, owned_fd)
                os.close(replacement_fd)
                replacement_fd = owned_fd

        os.fstat(replacement_fd)
        secret_fd = os.open("secret.txt", os.O_RDONLY, dir_fd=replacement_fd)
        try:
            assert os.read(secret_fd, 100) == b"SECRET"
        finally:
            os.close(secret_fd)
    finally:
        if replacement_fd >= 0:
            os.close(replacement_fd)
        workspace.close()


@pytest.mark.parametrize(
    ("opener", "path"),
    [("open_file", "value.txt"), ("open_directory", ".")],
)
def test_workspace_closes_descriptors_when_initial_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opener: str,
    path: str,
) -> None:
    (tmp_path / "value.txt").write_text("value", encoding="utf-8")
    workspace = Workspace(tmp_path)
    root_fds = {lease.identity.fd for lease in workspace._root_leases}
    real_fstat = os.fstat
    failed_fds: list[int] = []

    def failing_fstat(descriptor: int) -> os.stat_result:
        if descriptor not in root_fds:
            failed_fds.append(descriptor)
            raise OSError("simulated fstat failure")
        return real_fstat(descriptor)

    try:
        monkeypatch.setattr(workspace_module.os, "fstat", failing_fstat)
        with pytest.raises(OSError, match="simulated fstat failure"):
            with getattr(workspace, opener)(path):
                pass
        monkeypatch.setattr(workspace_module.os, "fstat", real_fstat)

        assert failed_fds
        with pytest.raises(OSError):
            real_fstat(failed_fds[-1])
    finally:
        workspace.close()


def test_workspace_closes_descriptor_when_fdopen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "value.txt").write_text("value", encoding="utf-8")
    workspace = Workspace(tmp_path)
    real_fdopen = os.fdopen
    failed_fds: list[int] = []

    def failing_fdopen(descriptor: int, *_args: object, **_kwargs: object):
        failed_fds.append(descriptor)
        raise OSError("simulated fdopen failure")

    try:
        monkeypatch.setattr(workspace_module.os, "fdopen", failing_fdopen)
        with pytest.raises(OSError, match="simulated fdopen failure"):
            with workspace.open_file("value.txt"):
                pass
        monkeypatch.setattr(workspace_module.os, "fdopen", real_fdopen)

        assert failed_fds
        with pytest.raises(OSError):
            os.fstat(failed_fds[-1])
    finally:
        workspace.close()


@pytest.mark.parametrize("child_name", ["", ".", "..", "nested/value.txt"])
def test_workspace_child_primitives_reject_non_child_names(
    tmp_path: Path,
    child_name: str,
) -> None:
    workspace = Workspace(tmp_path)
    try:
        with workspace.open_directory(".") as root:
            with pytest.raises(ValueError, match="Invalid child name"):
                with workspace.open_child_file(root, child_name):
                    pass
    finally:
        workspace.close()


def test_atomic_rewrite_commits_and_preserves_mode(tmp_path: Path) -> None:
    target = tmp_path / "script.sh"
    target.write_bytes(b"echo old\n")
    target.chmod(0o751)
    workspace = Workspace(tmp_path)
    try:
        result = workspace.atomic_rewrite_regular_file(
            "script.sh",
            lambda content: content.replace(b"old", b"new"),
            max_bytes=100,
        )
    finally:
        workspace.close()

    assert isinstance(result, AtomicRewriteResult)
    assert result.path.absolute == target
    assert result.bytes_before == len(b"echo old\n")
    assert result.bytes_after == len(b"echo new\n")
    assert result.changed is True
    assert result.committed is True
    assert result.durable is True
    assert result.warning is None
    assert target.read_bytes() == b"echo new\n"
    assert target.stat().st_mode & 0o7777 == 0o751
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_dry_run_never_creates_a_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)

    def unexpected_temp(_parent_fd: int) -> tuple[str, int]:
        raise AssertionError("dry run created a temporary file")

    try:
        monkeypatch.setattr(
            workspace_module,
            "_create_atomic_rewrite_temp",
            unexpected_temp,
        )
        result = workspace.atomic_rewrite_regular_file(
            "value.txt",
            lambda _content: b"new",
            max_bytes=100,
            dry_run=True,
        )
    finally:
        workspace.close()

    assert result.changed is True
    assert result.committed is False
    assert result.durable is None
    assert target.read_bytes() == b"old"


def test_atomic_rewrite_rejects_source_and_result_over_limit(
    tmp_path: Path,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"12345")
    workspace = Workspace(tmp_path)
    try:
        with pytest.raises(ValueError, match="File exceeds configured file size limit"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda content: content,
                max_bytes=4,
            )
        with pytest.raises(
            ValueError,
            match="Replacement exceeds configured file size limit",
        ):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"123456",
                max_bytes=5,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"12345"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_rejects_parent_symlink_swap_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    safe = root / "safe"
    moved_safe = root / "moved-safe"
    outside = tmp_path / "outside"
    safe.mkdir(parents=True)
    outside.mkdir()
    (safe / "value.txt").write_bytes(b"safe old")
    (outside / "value.txt").write_bytes(b"outside secret")
    workspace = Workspace(root)
    real_fsync = os.fsync
    swapped = False

    def racing_fsync(descriptor: int) -> None:
        nonlocal swapped
        real_fsync(descriptor)
        descriptor_stat = os.fstat(descriptor)
        if stat_module.S_ISREG(descriptor_stat.st_mode) and not swapped:
            swapped = True
            safe.rename(moved_safe)
            safe.symlink_to(outside, target_is_directory=True)

    try:
        monkeypatch.setattr(workspace_module.os, "fsync", racing_fsync)
        with pytest.raises(ConcurrentFileChange, match="Parent directory changed"):
            workspace.atomic_rewrite_regular_file(
                "safe/value.txt",
                lambda _content: b"safe new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert (outside / "value.txt").read_bytes() == b"outside secret"
    assert (moved_safe / "value.txt").read_bytes() == b"safe old"
    assert list(moved_safe.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_rejects_destination_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    original = tmp_path / "original.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)
    real_fsync = os.fsync
    swapped = False

    def racing_fsync(descriptor: int) -> None:
        nonlocal swapped
        real_fsync(descriptor)
        descriptor_stat = os.fstat(descriptor)
        if stat_module.S_ISREG(descriptor_stat.st_mode) and not swapped:
            swapped = True
            target.rename(original)
            target.write_bytes(b"other")

    try:
        monkeypatch.setattr(workspace_module.os, "fsync", racing_fsync)
        with pytest.raises(ConcurrentFileChange, match="File changed"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"other"
    assert original.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_rejects_in_place_content_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)
    real_fsync = os.fsync
    changed = False

    def racing_fsync(descriptor: int) -> None:
        nonlocal changed
        real_fsync(descriptor)
        descriptor_stat = os.fstat(descriptor)
        if stat_module.S_ISREG(descriptor_stat.st_mode) and not changed:
            changed = True
            target.write_bytes(b"other")

    try:
        monkeypatch.setattr(workspace_module.os, "fsync", racing_fsync)
        with pytest.raises(ConcurrentFileChange, match="File changed"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"other"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_temp_fsync_failure_preserves_original_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)
    real_fsync = os.fsync

    def failing_fsync(descriptor: int) -> None:
        if stat_module.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("simulated temp fsync failure")
        real_fsync(descriptor)

    try:
        monkeypatch.setattr(workspace_module.os, "fsync", failing_fsync)
        with pytest.raises(OSError, match="simulated temp fsync failure"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_rename_failure_preserves_original_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)

    def failing_rename(
        _source: object,
        _destination: object,
        **_kwargs: object,
    ) -> None:
        raise OSError("simulated rename failure")

    try:
        monkeypatch.setattr(workspace_module.os, "rename", failing_rename)
        with pytest.raises(OSError, match="simulated rename failure"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_never_unlinks_a_substituted_temp_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)
    real_rename = os.rename
    substituted_name: str | None = None

    def substituting_rename(
        source: str,
        _destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal substituted_name
        assert src_dir_fd == dst_dir_fd
        substituted_name = source
        real_rename(
            source,
            "stolen-temp",
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        (tmp_path / source).write_bytes(b"foreign")
        raise OSError("simulated substituted temp")

    try:
        monkeypatch.setattr(workspace_module.os, "rename", substituting_rename)
        with pytest.raises(OSError, match="simulated substituted temp"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert substituted_name is not None
    assert (tmp_path / substituted_name).read_bytes() == b"foreign"
    assert (tmp_path / "stolen-temp").read_bytes() == b"new"
    assert target.read_bytes() == b"old"


def test_atomic_rewrite_reports_directory_fsync_failure_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)
    real_fsync = os.fsync

    def failing_directory_fsync(descriptor: int) -> None:
        if stat_module.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("simulated directory fsync failure")
        real_fsync(descriptor)

    try:
        monkeypatch.setattr(workspace_module.os, "fsync", failing_directory_fsync)
        result = workspace.atomic_rewrite_regular_file(
            "value.txt",
            lambda _content: b"new",
            max_bytes=100,
        )
    finally:
        workspace.close()

    assert result.committed is True
    assert result.durable is False
    assert result.warning is not None
    assert "directory fsync failed" in result.warning
    assert target.read_bytes() == b"new"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_rejects_hardlinked_targets(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    linked = tmp_path / "linked.txt"
    target.write_bytes(b"old")
    os.link(target, linked)
    workspace = Workspace(tmp_path)
    try:
        with pytest.raises(ValueError, match="requires one hard link"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"old"
    assert linked.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


@pytest.mark.parametrize("special_mode", [stat_module.S_ISUID, stat_module.S_ISGID])
def test_atomic_rewrite_rejects_setid_targets(
    tmp_path: Path,
    special_mode: int,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    target.chmod(0o755 | special_mode)
    if not target.stat().st_mode & special_mode:
        pytest.skip("test filesystem does not retain setid mode bits")
    workspace = Workspace(tmp_path)
    try:
        with pytest.raises(ValueError, match="rejects setuid/setgid"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_fchown_failure_preserves_original_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)
    real_create_temp = workspace_module._create_atomic_rewrite_temp
    real_fstat = os.fstat
    temp_fd = -1
    altered_temp_stat = False

    def recording_create_temp(parent_fd: int) -> tuple[str, int]:
        nonlocal temp_fd
        name, temp_fd = real_create_temp(parent_fd)
        return name, temp_fd

    def mismatched_temp_owner(descriptor: int) -> os.stat_result:
        nonlocal altered_temp_stat
        value = real_fstat(descriptor)
        if descriptor == temp_fd and not altered_temp_stat:
            altered_temp_stat = True
            fields = list(value)
            fields[4] = value.st_uid + 1
            return os.stat_result(fields)
        return value

    def failing_fchown(_descriptor: int, _uid: int, _gid: int) -> None:
        raise PermissionError("simulated ownership preservation failure")

    try:
        monkeypatch.setattr(
            workspace_module,
            "_create_atomic_rewrite_temp",
            recording_create_temp,
        )
        monkeypatch.setattr(workspace_module.os, "fstat", mismatched_temp_owner)
        monkeypatch.setattr(workspace_module.os, "fchown", failing_fchown)
        with pytest.raises(PermissionError, match="ownership preservation failure"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert altered_temp_stat is True
    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_cancellation_before_staging_never_creates_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)

    def unexpected_temp(_parent_fd: int) -> tuple[str, int]:
        raise AssertionError("cancelled rewrite created a temporary file")

    try:
        monkeypatch.setattr(
            workspace_module,
            "_create_atomic_rewrite_temp",
            unexpected_temp,
        )
        with pytest.raises(AtomicRewriteCancelled, match="cancelled"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
                cancelled=lambda: True,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"old"


def test_atomic_rewrite_cancellation_before_rename_cleans_owned_temp(
    tmp_path: Path,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)
    checks = 0

    def cancel_second_checkpoint() -> bool:
        nonlocal checks
        checks += 1
        return checks == 2

    try:
        with pytest.raises(AtomicRewriteCancelled, match="cancelled"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
                cancelled=cancel_second_checkpoint,
            )
    finally:
        workspace.close()

    assert checks == 2
    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_post_commit_close_failure_returns_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)
    real_create_temp = workspace_module._create_atomic_rewrite_temp
    real_close = os.close
    temp_fd = -1

    def recording_create_temp(parent_fd: int) -> tuple[str, int]:
        nonlocal temp_fd
        name, temp_fd = real_create_temp(parent_fd)
        return name, temp_fd

    def failing_temp_close(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == temp_fd:
            raise OSError("simulated post-commit close failure")

    try:
        monkeypatch.setattr(
            workspace_module,
            "_create_atomic_rewrite_temp",
            recording_create_temp,
        )
        monkeypatch.setattr(workspace_module.os, "close", failing_temp_close)
        result = workspace.atomic_rewrite_regular_file(
            "value.txt",
            lambda _content: b"new",
            max_bytes=100,
        )
    finally:
        workspace.close()

    assert result.committed is True
    assert result.durable is False
    assert result.warning is not None
    assert "closing temporary file failed" in result.warning
    assert "simulated post-commit close failure" in result.warning
    assert target.read_bytes() == b"new"


def test_atomic_rewrite_preserves_extended_attributes(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    if sys.platform == "darwin":
        attribute_name = "com.ghostwheel.atomic-rewrite-test"
        subprocess.run(
            ["xattr", "-w", attribute_name, "preserved", str(target)],
            check=True,
            capture_output=True,
        )

        def read_attribute() -> bytes:
            return subprocess.run(
                ["xattr", "-p", attribute_name, str(target)],
                check=True,
                capture_output=True,
            ).stdout.rstrip(b"\n")

    elif sys.platform.startswith("linux") and hasattr(os, "setxattr"):
        attribute_name = "user.ghostwheel.atomic-rewrite-test"
        os.setxattr(target, attribute_name, b"preserved")

        def read_attribute() -> bytes:
            return os.getxattr(target, attribute_name)

    else:
        pytest.skip("extended attribute test is unsupported on this platform")

    workspace = Workspace(tmp_path)
    try:
        result = workspace.atomic_rewrite_regular_file(
            "value.txt",
            lambda _content: b"new",
            max_bytes=100,
        )
    finally:
        workspace.close()

    assert result.committed is True
    assert target.read_bytes() == b"new"
    assert read_attribute() == b"preserved"


@pytest.mark.parametrize(
    "attribute_name",
    [
        "security.capability",
        "security.ima",
        "security.evm",
        "security.future-integrity-policy",
        "trusted.overlay.opaque",
    ],
)
def test_linux_metadata_copy_rejects_privileged_or_integrity_attributes(
    monkeypatch: pytest.MonkeyPatch,
    attribute_name: str,
) -> None:
    list_calls: list[int] = []

    def listxattr(descriptor: int) -> list[str]:
        list_calls.append(descriptor)
        return [attribute_name]

    def unexpected_attribute_operation(*_args: object) -> None:
        raise AssertionError("rejected attribute was accessed or copied")

    monkeypatch.setattr(workspace_module.os, "listxattr", listxattr, raising=False)
    monkeypatch.setattr(
        workspace_module.os,
        "getxattr",
        unexpected_attribute_operation,
        raising=False,
    )
    monkeypatch.setattr(
        workspace_module.os,
        "setxattr",
        unexpected_attribute_operation,
        raising=False,
    )
    monkeypatch.setattr(
        workspace_module.os,
        "removexattr",
        unexpected_attribute_operation,
        raising=False,
    )

    with pytest.raises(ValueError) as caught:
        workspace_module._copy_linux_file_metadata(11, 12)

    assert attribute_name in str(caught.value)
    assert list_calls == [11]


def test_linux_metadata_copy_allows_selinux_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_calls: list[tuple[int, str, bytes]] = []

    def listxattr(descriptor: int) -> list[str]:
        return ["security.selinux"] if descriptor == 11 else []

    def getxattr(descriptor: int, name: str) -> bytes:
        if descriptor == 11:
            assert name == "security.selinux"
            return b"system_u:object_r:user_home_t:s0"
        raise OSError(errno.ENODATA, "attribute is absent")

    def setxattr(descriptor: int, name: str, value: bytes) -> None:
        set_calls.append((descriptor, name, value))

    monkeypatch.setattr(workspace_module.os, "listxattr", listxattr, raising=False)
    monkeypatch.setattr(workspace_module.os, "getxattr", getxattr, raising=False)
    monkeypatch.setattr(workspace_module.os, "setxattr", setxattr, raising=False)
    monkeypatch.setattr(
        workspace_module.os,
        "removexattr",
        lambda *_args: pytest.fail("no destination attributes should be removed"),
        raising=False,
    )

    workspace_module._copy_linux_file_metadata(11, 12)

    assert set_calls == [(12, "security.selinux", b"system_u:object_r:user_home_t:s0")]


def test_linux_getflags_uses_native_ioctl_and_allows_layout_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout_flags = (
        workspace_module._LINUX_FS_EXTENTS_FL
        | 0x00200000  # FS_EA_INODE_FL
        | 0x10000000  # FS_INLINE_DATA_FL
    )
    calls: list[tuple[int, int, bool]] = []

    def ioctl(
        descriptor: int,
        request: int,
        buffer: bytearray,
        mutate: bool,
    ) -> int:
        calls.append((descriptor, request, mutate))
        buffer[:4] = layout_flags.to_bytes(4, sys.byteorder)
        return 0

    monkeypatch.setattr(workspace_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(workspace_module.fcntl, "ioctl", ioctl)

    flags = workspace_module._linux_descriptor_file_flags(23)

    expected_request = (
        (2 << 30) | (workspace_module.struct.calcsize("@l") << 16) | (ord("f") << 8) | 1
    )
    assert calls == [(23, expected_request, True)]
    assert flags == layout_flags
    assert workspace_module._unsupported_linux_atomic_rewrite_flags(flags) == 0


@pytest.mark.parametrize("ioctl_error", [errno.ENOTTY, errno.EOPNOTSUPP])
def test_linux_getflags_tolerates_unsupported_filesystems(
    monkeypatch: pytest.MonkeyPatch,
    ioctl_error: int,
) -> None:
    def unsupported_ioctl(*_args: object) -> int:
        raise OSError(ioctl_error, "ioctl unsupported")

    monkeypatch.setattr(workspace_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(workspace_module.fcntl, "ioctl", unsupported_ioctl)

    assert workspace_module._linux_descriptor_file_flags(23) == 0


@pytest.mark.parametrize(
    ("machine", "size_bits", "read_direction"),
    [
        ("x86_64", 14, 2),
        ("aarch64", 14, 2),
        ("mips64el", 13, 2),
        ("ppc64le", 13, 2),
        ("alpha", 13, 2),
        ("sparc64", 13, 2),
        ("hppa64", 14, 1),
    ],
)
def test_linux_getflags_request_uses_architecture_uapi(
    machine: str,
    size_bits: int,
    read_direction: int,
) -> None:
    expected = (
        (read_direction << (16 + size_bits))
        | (workspace_module.struct.calcsize("@l") << 16)
        | (ord("f") << 8)
        | 1
    )

    assert workspace_module._linux_fs_ioc_getflags_request(machine) == expected


def test_linux_getflags_rejects_unknown_ioctl_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workspace_module.platform, "machine", lambda: "future64")
    monkeypatch.setattr(
        workspace_module.fcntl,
        "ioctl",
        lambda *_args: pytest.fail("unknown ABI attempted an ioctl"),
    )

    with pytest.raises(RuntimeError, match="does not support Linux ioctl ABI"):
        workspace_module._linux_descriptor_file_flags(23)


def test_linux_getflags_does_not_treat_enosys_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked_ioctl(*_args: object) -> int:
        raise OSError(errno.ENOSYS, "ioctl blocked")

    monkeypatch.setattr(workspace_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(workspace_module.fcntl, "ioctl", blocked_ioctl)

    with pytest.raises(OSError) as caught:
        workspace_module._linux_descriptor_file_flags(23)

    assert caught.value.errno == errno.ENOSYS


def test_linux_getflags_rejects_verity_and_unknown_semantic_flags() -> None:
    flags = (
        workspace_module._LINUX_FS_EXTENTS_FL
        | workspace_module._LINUX_FS_VERITY_FL
        | 0x00000040  # FS_NODUMP_FL
        | 0x00000200  # FS_COMPRBLK_FL is content-coupled; fail closed.
    )

    assert workspace_module._unsupported_linux_atomic_rewrite_flags(flags) == (
        workspace_module._LINUX_FS_VERITY_FL | 0x00000040 | 0x00000200
    )


def test_atomic_rewrite_rejects_semantic_descriptor_flags_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)

    def unexpected_temp(_parent_fd: int) -> tuple[str, int]:
        raise AssertionError("flagged rewrite created a temporary file")

    try:
        monkeypatch.setattr(
            workspace_module,
            "_descriptor_file_flags",
            lambda _descriptor, _stat: workspace_module._LINUX_FS_VERITY_FL,
        )
        monkeypatch.setattr(
            workspace_module,
            "_create_atomic_rewrite_temp",
            unexpected_temp,
        )
        with pytest.raises(ValueError, match="semantic platform flags"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


def test_atomic_rewrite_revalidates_descriptor_flags_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)
    observed_flags = iter([0, workspace_module._LINUX_FS_VERITY_FL])

    try:
        monkeypatch.setattr(
            workspace_module,
            "_descriptor_file_flags",
            lambda _descriptor, _stat: next(observed_flags),
        )
        with pytest.raises(ConcurrentFileChange, match="File flags changed"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS metadata API test")
def test_darwin_metadata_copy_excludes_stat_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []

    def recording_fcopyfile(source: int, destination: int, flags: int) -> None:
        calls.append((source, destination, flags))

    monkeypatch.setattr(workspace_module.posix, "_fcopyfile", recording_fcopyfile)

    workspace_module._copy_darwin_file_metadata(11, 12)

    assert calls == [
        (
            11,
            12,
            workspace_module.posix._COPYFILE_ACL
            | workspace_module.posix._COPYFILE_XATTR,
        )
    ]
    assert not calls[0][2] & workspace_module.posix._COPYFILE_STAT


def test_atomic_rewrite_metadata_copy_failure_preserves_original_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    workspace = Workspace(tmp_path)

    def failing_metadata_copy(_source_fd: int, _destination_fd: int) -> None:
        raise PermissionError("simulated metadata preservation failure")

    try:
        monkeypatch.setattr(
            workspace_module,
            "_copy_atomic_rewrite_metadata",
            failing_metadata_copy,
        )
        with pytest.raises(PermissionError, match="metadata preservation failure"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []


@pytest.mark.skipif(
    not hasattr(os, "chflags") or not hasattr(stat_module, "UF_HIDDEN"),
    reason="platform file flags are unavailable",
)
def test_atomic_rewrite_rejects_platform_flagged_files(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"old")
    os.chflags(target, stat_module.UF_HIDDEN)
    workspace = Workspace(tmp_path)
    try:
        with pytest.raises(ValueError, match="platform flags"):
            workspace.atomic_rewrite_regular_file(
                "value.txt",
                lambda _content: b"new",
                max_bytes=100,
            )
    finally:
        workspace.close()
        os.chflags(target, 0)

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".ghostwheel-edit-*.tmp")) == []
