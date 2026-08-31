"""The panel contract.

A panel is one file in `panels/`. It defines a module-level `PANEL` and nothing
else is required. Deleting the file removes the panel from the dashboard; no
registry edit, no import to remove, no other file to touch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Panel:
    """One dashboard section.

    Args:
        id: unique slug, also the DOM id prefix.
        title: heading shown above the panel.
        build: (RunData) -> Dash component. Called once when the app is built.
        order: sort key; lower appears first.
        blurb: one line under the title saying what the reader is looking at.
        register: optional (app, RunData) -> None for panels with callbacks.
        requires: array keys the run must contain, else the panel is skipped.
    """

    id: str
    title: str
    build: Callable[[Any], Any]
    order: int = 100
    blurb: str = ""
    register: Optional[Callable[[Any, Any], None]] = None
    requires: tuple[str, ...] = ()

    def available(self, data) -> bool:
        return all(data.has(k) for k in self.requires)
