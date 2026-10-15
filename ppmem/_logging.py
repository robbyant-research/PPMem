import logging
import os

# Quiet libkineto. Must be set before torch initializes, i.e. before `import torch`.
os.environ.setdefault("KINETO_LOG_LEVEL", "5")

logger = logging.getLogger("ppmem")


def init_logger(level: str | None = None) -> None:
    """Attach a stderr handler to the `ppmem` logger.

    Level defaults to PPMEM_LOG_LEVEL or INFO. Safe to call more than once.
    """
    if logger.handlers:
        return
    lv = (level or os.environ.get("PPMEM_LOG_LEVEL") or "INFO").upper()
    logger.setLevel(lv)
    handler = logging.StreamHandler()
    handler.setLevel(lv)
    handler.setFormatter(
        logging.Formatter("[ppmem] %(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
