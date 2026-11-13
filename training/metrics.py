import math
import os

import torch


def memory_usage_mb():
    """Return process RSS in MiB, or None if the OS query is unavailable."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters(); counters.cb = ctypes.sizeof(counters)
            if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
        else:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return None
    return None


class LossAccumulator:
    def __init__(self):
        self.loss_sum = 0.0
        self.tokens = 0

    def update(self, loss, token_count):
        if not math.isfinite(float(loss)) or token_count <= 0:
            raise ValueError("loss must be finite and token_count positive")
        self.loss_sum += float(loss) * int(token_count)
        self.tokens += int(token_count)

    @property
    def average_loss(self):
        return self.loss_sum / self.tokens if self.tokens else None

    @property
    def perplexity(self):
        loss = self.average_loss
        max_log = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
        return math.exp(loss) if loss is not None and loss <= max_log else None
