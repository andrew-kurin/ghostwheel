import copy
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import ghostwheel.tools.search as search_module
import ghostwheel.tools.workspace as workspace_module

from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools.filesystem import ls, read
from ghostwheel.tools.search import GrepIncompleteReason, grep
from ghostwheel.tools.workspace import Workspace

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
