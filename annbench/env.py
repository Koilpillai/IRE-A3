"""Record the machine and library versions a result was produced on."""
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

LIBRARIES = ["numpy", "scipy", "h5py", "faiss-cpu", "hnswlib", "matplotlib", "pandas", "pytest", "tqdm"]


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def physical_cores() -> int:
    """Number of physical CPU cores (hyper-threads excluded); the logical count if that cannot be read."""
    try:
        cores, physical_id = set(), None
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() == "physical id":
                    physical_id = value.strip()
                elif key.strip() == "core id":
                    cores.add((physical_id, value.strip()))
        if cores:
            return len(cores)
    except OSError:
        pass
    return os.cpu_count() or 1


def _ram_gib() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / 2**20, 1)  # MemTotal, kB -> GiB
    except (OSError, ValueError):
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect() -> dict:
    import faiss

    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = version(lib)
        except PackageNotFoundError:
            versions[lib] = None
    return {
        "cpu": _cpu_model(),
        "physical_cores": physical_cores(),
        "logical_cpus": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "libraries": versions,
        "faiss_compile_options": faiss.get_compile_options(),
        "faiss_blas_threshold": faiss.cvar.distance_compute_blas_threshold,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
