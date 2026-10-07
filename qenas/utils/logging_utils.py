"""Console + file logging, and progress/ETA formatting for long computations."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

LOGGER_NAME = "qenas"


def ensure_utf8_console() -> None:
    """Windows consoles default to cp1252; make stdout/stderr UTF-8 so symbols such as Δ never crash a run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.encoding and stream.encoding.lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    ensure_utf8_console()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                h.close()
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def format_seconds(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "--:--:--"
    seconds = int(round(max(0.0, seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


class ProgressTimer:
    """Tracks elapsed time and estimates remaining time for ``total`` work units.

    ``done_before`` lets a resumed run count already-completed units without
    polluting the per-unit timing estimate.
    """

    def __init__(self, total: int, done_before: int = 0):
        self.total = max(1, int(total))
        self.done_before = int(done_before)
        self.start = time.time()
        self.done_now = 0

    def step(self, n: int = 1) -> None:
        self.done_now += n

    @property
    def done(self) -> int:
        return self.done_before + self.done_now

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def percent(self) -> float:
        return 100.0 * self.done / self.total

    @property
    def eta(self) -> float:
        if self.done_now <= 0:
            return float("inf")
        per_unit = self.elapsed / self.done_now
        return per_unit * (self.total - self.done)

    def summary(self) -> str:
        return (f"Progress: {self.percent:5.1f}% | Elapsed: {format_seconds(self.elapsed)} "
                f"| ETA: {format_seconds(self.eta)}")
