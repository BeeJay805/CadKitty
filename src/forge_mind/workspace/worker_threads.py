"""Math-library thread policy for disposable workspace worker processes.

Apply these values to each child environment before Python starts. Native
libraries read their thread settings during import, so a worker-side update can
arrive too late. This policy does not change process, CPU or deadline limits.
"""

WORKER_THREAD_ENV: tuple[tuple[str, str], ...] = (
    ("OPENBLAS_NUM_THREADS", "1"),
    ("OMP_NUM_THREADS", "1"),
    ("OMP_THREAD_LIMIT", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
)


def docker_thread_arguments() -> list[str]:
    """Override image defaults before the worker interpreter starts."""
    return [
        argument for key, value in WORKER_THREAD_ENV for argument in ("--env", f"{key}={value}")
    ]
