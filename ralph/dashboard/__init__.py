"""Ralph status dashboard — aggregated view of active `ralph run` instances.

Running ralph instances write a small JSON state file plus a tail log under
``~/.ralph/instances/``. The ``ralph dashboard`` command serves a local web UI
that reads those files and presents live progress for every active run.
"""

from ralph.dashboard.sources import (
    InstanceSnapshot,
    InstanceSource,
    LocalFilesSource,
)

__all__ = [
    "InstanceSnapshot",
    "InstanceSource",
    "LocalFilesSource",
]
