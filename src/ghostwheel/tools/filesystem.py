"""Backward-compatible facade for filesystem tools and result types."""

# ``os`` remains available for callers/tests that historically monkeypatched
# filesystem traversal through this module. The implementations live elsewhere.
import os

from .edit import EditResult, edit
from .listing import (
    DirectoryListing,
    DirEntry,
    FileKind,
    ListingIncompleteReason,
    ls,
)
from .read import ReadIncompleteReason, ReadResult, read

__all__ = [
    "DirectoryListing",
    "DirEntry",
    "EditResult",
    "FileKind",
    "ListingIncompleteReason",
    "ReadIncompleteReason",
    "ReadResult",
    "edit",
    "ls",
    "os",
    "read",
]
