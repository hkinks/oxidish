"""Small helpers shared by the evaluation tools: repo root, free ports, CPU pinning, lineup names."""

import os
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def limit_cpus(n: int) -> None:
    """Pin this process to the first n logical CPUs; worker and server processes started afterwards inherit it.
    Keeps long runs under the machine's CPU budget (<= 90% total)."""
    if n <= 0:
        return
    mask = (1 << min(n, os.cpu_count() or n)) - 1
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE      # 64-bit handle: the default c_int truncated it
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.SetProcessAffinityMask.restype = wintypes.BOOL
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
    else:
        os.sched_setaffinity(0, set(range(min(n, os.cpu_count() or n))))


def parse_lineup(specs: list[str]) -> list[tuple[str, str]]:
    """"route:4 champ=path.json:2" -> [(agent name, spec)], named like the server names in-process agents."""
    lineup = []
    for spec in specs:
        head, _, tail = spec.rpartition(":")  # count is after the last colon (paths may have C:)
        key, count = (head, tail) if tail.isdigit() else (spec, "1")
        key, _, path = key.partition("=")
        lineup += [(f"{key}-{i}", path or key) for i in range(int(count))]
    return lineup
