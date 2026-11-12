"""HTTP service that wraps one Backend (rung 06).

Run it with ``uvicorn ladder.serve.app:create_app --factory``. FastAPI is an
optional dependency (the ``serve`` extra), so this package is only imported when
the service is actually used.
"""
