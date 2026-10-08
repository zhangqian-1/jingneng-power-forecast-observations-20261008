"""Supervise two isolated processes inside one container."""
from __future__ import annotations

import logging
import signal
import subprocess
import sys
import threading
import time

LOG = logging.getLogger("services")


def supervise(commands: list[list[str]], stop: threading.Event) -> int:
    children = []
    try:
        for command in commands:
            children.append(subprocess.Popen(command))
        while not stop.wait(0.2):
            for child in children:
                code = child.poll()
                if code is not None:
                    LOG.error("Service pid=%d exited with code=%d; stopping container", child.pid, code)
                    return code if code > 0 else 1
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + 20
        for child in children:
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    commands = [
        [sys.executable, "app/api.py", "--host", "0.0.0.0", "--port", "8000"],
        [sys.executable, "-m", "observations_service.api", "--host", "0.0.0.0", "--port", "8002"],
    ]
    raise SystemExit(supervise(commands, stop))


if __name__ == "__main__":
    main()
