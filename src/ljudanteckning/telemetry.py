from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuStat:
    gpu_id: str
    name: str
    util_pct: int | None
    mem_used_mb: int | None
    mem_total_mb: int | None
    temp_c: int | None
    power_w: int | None
    power_limit_w: int | None


def nvml_available() -> bool:
    try:
        import pynvml  # noqa: F401

        return True
    except Exception:
        return False


def _mb(x: int) -> int:
    return int(round(x / (1024 * 1024)))


def get_gpu_stats(gpu_ids: list[str]) -> list[GpuStat]:
    """
    Read GPU telemetry via NVML. gpu_ids are *logical* IDs as seen by NVML.

    NOTE: In workers we set CUDA_VISIBLE_DEVICES per worker.
    In the main process we *do not* set it, so gpu_ids here should match host NVML IDs.
    (That's exactly what you pass in --gpus today: "0,1,2,...")
    """
    import pynvml  # type: ignore

    try:
        pynvml.nvmlInit()
    except Exception:
        return []

    out: list[GpuStat] = []
    for gid in gpu_ids:
        try:
            idx = int(gid)
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(h).decode("utf-8", errors="replace")

            util = None
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                util = int(getattr(u, "gpu", 0))
            except Exception:
                util = None

            mem_used = mem_total = None
            try:
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                mem_used = _mb(int(m.used))
                mem_total = _mb(int(m.total))
            except Exception:
                mem_used = mem_total = None

            temp = None
            try:
                temp = int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                temp = None

            power = limit = None
            try:
                # milliwatts
                power = int(round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000))
                limit = int(round(pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000))
            except Exception:
                power = limit = None

            out.append(
                GpuStat(
                    gpu_id=gid,
                    name=name,
                    util_pct=util,
                    mem_used_mb=mem_used,
                    mem_total_mb=mem_total,
                    temp_c=temp,
                    power_w=power,
                    power_limit_w=limit,
                )
            )
        except Exception:
            continue

    with suppress(Exception):
        pynvml.nvmlShutdown()

    return out
