"""Exceptions of the application layer. Each front-end maps them the way that
suits it: the CLI to argparse's usage error (exit 2), the HTTP adapter to a
4xx JSON body, a Python caller just catches WmAppError."""


class WmAppError(Exception):
    """Base class: anything the application layer refuses on purpose."""


class SpecError(WmAppError, ValueError):
    """The request is invalid or self-contradictory (bad value, unknown option,
    --c2pa-cert without --c2pa-key...). A ValueError subclass, so code that
    caught the ValueError the old apply_watermark() raised keeps working."""


class UnavailableError(WmAppError, RuntimeError):
    """A requested feature needs an optional dependency that isn't installed
    (C2PA signing without c2pa-python / cryptography)."""


class LimitError(WmAppError, ValueError):
    """The request is valid but exceeds a configured resource limit (see
    wmapp.limits) - image too large, grid too dense, upload too big."""
