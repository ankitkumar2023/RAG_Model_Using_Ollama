from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_name: str | None
    gpu_memory_used_mb: float | None
    gpu_memory_total_mb: float | None
    gpu_utilization_percent: float | None


class SystemMonitor:
    """
    System resource monitor.
    """

    @staticmethod
    def get_cpu_usage() -> float:
        return psutil.cpu_percent(interval=1)

    @staticmethod
    def get_memory_usage() -> dict[str, float]:
        memory = psutil.virtual_memory()

        return {
            "percent": memory.percent,
            "used_gb": round(
                memory.used / (1024**3),
                2,
            ),
            "total_gb": round(
                memory.total / (1024**3),
                2,
            ),
        }

    @staticmethod
    def get_gpu_metrics() -> dict[str, Any]:
        """
        NVIDIA GPU metrics via nvidia-smi.
        """

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    ("--query-gpu=name,memory.used,memory.total,utilization.gpu"),
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            output = result.stdout.strip()

            if not output:
                return {}

            gpu_data = output.split(", ")

            return {
                "gpu_name": gpu_data[0],
                "gpu_memory_used_mb": float(gpu_data[1]),
                "gpu_memory_total_mb": float(gpu_data[2]),
                "gpu_utilization_percent": float(gpu_data[3]),
            }

        except Exception:
            logger.warning("GPU metrics unavailable.")

            return {}

    @classmethod
    def collect_metrics(
        cls,
    ) -> SystemMetrics:
        """
        Collect all system telemetry.
        """

        cpu = cls.get_cpu_usage()

        memory = cls.get_memory_usage()

        gpu = cls.get_gpu_metrics()

        metrics = SystemMetrics(
            cpu_percent=cpu,
            ram_percent=memory["percent"],
            ram_used_gb=memory["used_gb"],
            ram_total_gb=memory["total_gb"],
            gpu_name=gpu.get("gpu_name"),
            gpu_memory_used_mb=gpu.get("gpu_memory_used_mb"),
            gpu_memory_total_mb=gpu.get("gpu_memory_total_mb"),
            gpu_utilization_percent=gpu.get("gpu_utilization_percent"),
        )

        logger.info("System metrics collected.")

        return metrics

    @staticmethod
    def get_system_info() -> dict[str, str]:
        """
        Host machine information.
        """

        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }
