"""Notes: what used to be print() calls scattered through the pipeline.

The CLI always told the user things mid-run - "ffmpeg not found", "a random
--ss-key was generated, SAVE THIS", "--invisible is image-only; ignored for
x.mp4". A library must not print, and a web client can't read a server's
stdout, so those messages are now data: each is a Note (level, a stable
`code` to branch on, the human-readable `message`, and any structured `data`).
They are collected on every result (result.notes) and, if the caller passes
`on_note`, delivered as they happen - the CLI's callback simply prints
`note.message`, which keeps its output identical to before.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Note:
    level: str                     # "info" | "warning"
    code: str                      # stable identifier, e.g. "ss_key_generated"
    message: str                   # human-readable (worded for the CLI, which it originated in)
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return {"level": self.level, "code": self.code, "message": self.message, "data": dict(self.data)}


class NoteBus:
    """Collects notes and forwards each to an optional sink as it happens."""

    def __init__(self, sink=None):
        self.items = []
        self._sink = sink

    def add(self, level, code, message, **data):
        note = Note(level, code, message, data)
        self.items.append(note)
        if self._sink is not None:
            self._sink(note)
        return note

    def info(self, code, message, **data):
        return self.add("info", code, message, **data)

    def warning(self, code, message, **data):
        return self.add("warning", code, message, **data)

    def forward(self, notes):
        """Re-emit notes produced elsewhere (a batch worker process)."""
        for n in notes:
            self.items.append(n)
            if self._sink is not None:
                self._sink(n)
