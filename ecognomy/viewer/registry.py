"""Panel discovery.

Imports every module in `panels/` and collects its `PANEL`. Adding a panel is
adding a file; removing one is deleting a file.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings

from ecognomy.viewer.panel import Panel


def discover_panels() -> list[Panel]:
    """All panels found in `ecognomy.viewer.panels`, sorted by `order`.

    A panel that fails to import is warned about and skipped rather than taking
    the whole dashboard down -- one broken plot should not cost you the run.
    """
    from ecognomy.viewer import panels as pkg

    found: list[Panel] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{pkg.__name__}.{info.name}")
        except Exception as exc:  # noqa: BLE001 - a bad panel must not be fatal
            warnings.warn(f"panel {info.name!r} failed to import and was skipped: {exc}")
            continue
        panel = getattr(module, "PANEL", None)
        if isinstance(panel, Panel):
            found.append(panel)
        else:
            warnings.warn(f"module {info.name!r} in panels/ defines no PANEL and was skipped")
    return sorted(found, key=lambda p: (p.order, p.id))
