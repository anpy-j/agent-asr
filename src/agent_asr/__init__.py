"""Public Python API. Model dependencies are imported only during inference."""

__version__ = "0.1.1"
API_VERSION = 1


def transcribe(request):
    from .core import transcribe as run

    return run(request)
