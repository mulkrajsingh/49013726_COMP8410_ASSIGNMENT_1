# Sample a worker process, its children and available device/container counters.
# Save resource samples, stage markers and an observed-peak summary.
# Sampled peaks can miss short spikes; the recorded memory scopes differ.

"""Sample a worker and its descendants in a separate process, including native code.

Peaks from samples are observed lower bounds. Device/host/container totals have
different scopes from process RSS and PyTorch allocator measurements.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _number(path):
    value = _read(path)
    try:
        number = int(value)
        return number if number < 2**60 else None
    except (ValueError, TypeError):
        return None


# Read container limits separately from host RAM; they describe different boundaries.
def _cgroup(pid):
    """Find the monitored process cgroup; never substitute host RAM for its limit."""
    contents = _read(f"/proc/{pid}/cgroup") or ""
    for line in contents.splitlines():
        version, controllers, relative = line.split(":", 2)
        base = Path("/sys/fs/cgroup")
        if version == "0" and not controllers:
            candidates = [base / relative.lstrip("/"), base]
            for directory in candidates:
                if (directory / "memory.current").exists():
                    return directory, 2
        elif "memory" in controllers.split(","):
            candidates = [base / "memory" / relative.lstrip("/"), base / "memory"]
            for directory in candidates:
                if (directory / "memory.usage_in_bytes").exists():
                    return directory, 1
    return None, None


# Try the available cgroup versions and keep unavailable sensors blank.
def _container(directory, version):
    if directory is None:
        return {"available": False}
    names = (
        {"memory_bytes": "memory.current", "memory_limit_bytes": "memory.max",
         "lifetime_memory_peak_bytes": "memory.peak", "swap_bytes": "memory.swap.current"}
        if version == 2 else
        {"memory_bytes": "memory.usage_in_bytes", "memory_limit_bytes": "memory.limit_in_bytes",
         "lifetime_memory_peak_bytes": "memory.max_usage_in_bytes"}
    )
    result = {key: _number(directory / name) for key, name in names.items()}
    result.update(available=True, cgroup_version=version, scope=str(directory))
    if version == 2:
        result["cpu_quota"] = _read(directory / "cpu.max")
        result["cpu_usage"] = _read(directory / "cpu.stat")
        result["memory_events"] = _read(directory / "memory.events")
    return result


def _gpu_devices(nvml):
    devices = []
    if nvml is not None:
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            uuid = nvml.nvmlDeviceGetUUID(handle)
            name = nvml.nvmlDeviceGetName(handle)
            devices.append((handle, str(uuid), str(name)))
    return devices


# Sample whole-device GPU use, so these values may include other processes.
def _gpu_sample(nvml, devices, errors):
    rows = []
    for handle, uuid, name in devices:
        row = {"uuid": uuid, "name": name}
        # Unsupported sensors stay null, rather than being reported as zero.
        queries = {
            "memory_used_bytes": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).used,
            "memory_total_bytes": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).total,
            "utilization_percent": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "memory_utilization_percent": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).memory,
            "temperature_c": lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
            "power_watts": lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000,
        }
        for key, query in queries.items():
            try:
                row[key] = query()
            except nvml.NVMLError as error:
                row[key] = None
                errors.add(f"GPU {uuid} {key}: {type(error).__name__}")
        rows.append(row)
    return rows


# Retain the highest observed sample; sampling may miss short spikes.
def _merge_peaks(target, values):
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = max(target.get(key, value), value)


# Sample the worker and its children until the stop file is created.
def _collect(folder, pid, interval, paths):
    import psutil

    folder = Path(folder)
    errors = set()
    nvml = None
    try:
        import pynvml as nvml
        nvml.nvmlInit()
    except Exception as error:
        nvml = None
        errors.add(f"NVML unavailable: {type(error).__name__}")
    devices = _gpu_devices(nvml)
    directory, version = _cgroup(pid)
    cpu_model = platform.processor()
    cpuinfo = _read("/proc/cpuinfo") or ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    try:
        subject = psutil.Process(pid)
        affinity = subject.cpu_affinity() if hasattr(subject, "cpu_affinity") else None
        identity = subject.create_time()
    except psutil.Error:
        affinity, identity = None, None
    metadata = {
        "started_at": _now(), "pid": pid, "cpu_model": cpu_model,
        "logical_cpu_count": psutil.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False),
        "cpu_affinity": affinity, "host_ram_total_bytes": psutil.virtual_memory().total,
        "platform": platform.platform(), "interval_seconds_requested": interval,
        "gpus": [{"uuid": uuid, "name": name} for _, uuid, name in devices],
        "container_at_start": _container(directory, version), "storage_paths": paths,
        "boundaries": [
            "Process metrics include the worker and observed descendants, excluding this sampler.",
            "Summed RSS can double count shared pages; sampled peaks can miss short-lived processes.",
            "CPU percent uses 100 percent per logical core and can exceed 100.",
            "GPU metrics cover each NVML-visible device, including other processes; no per-job energy claim.",
            "Host RAM and filesystem usage can include unrelated work; cgroup limits describe the visible container.",
            "Cgroup lifetime peak is not reset and is not an individual attempt peak.",
        ],
    }
    _json(folder / "metadata.json", metadata)
    summary = {
        "started_at": metadata["started_at"], "samples": 0, "complete": False,
        "requested_interval_seconds": interval, "max_sample_gap_seconds": 0,
        "process_peaks": {}, "host_peaks": {}, "container_peaks": {},
        "gpu_peaks": {}, "storage_peaks": {}, "errors": [],
    }
    previous_cpu, observed_cpu = {}, 0.0
    previous_time = None
    started = time.monotonic()
    try:
        with (folder / "samples.jsonl").open("x", buffering=1) as stream:
            while True:
                now = time.monotonic()
                processes = []
                alive = False
                try:
                    subject = psutil.Process(pid)
                    alive = subject.is_running() and subject.status() != psutil.STATUS_ZOMBIE
                    if identity is not None and subject.create_time() != identity:
                        alive = False
                    if alive:
                        processes = [subject]
                        try:
                            processes += subject.children(recursive=True)
                        except (psutil.Error, OSError) as error:
                            errors.add(f"Child enumeration: {type(error).__name__}")
                except (psutil.Error, OSError) as error:
                    errors.add(f"Worker lookup: {type(error).__name__}")
                totals = {"rss_bytes": 0, "process_count": 0, "threads": 0,
                          "read_bytes": 0, "write_bytes": 0}
                delta_cpu = 0.0
                for process in processes:
                    if process.pid == os.getpid():
                        continue
                    try:
                        with process.oneshot():
                            key = (process.pid, process.create_time())
                            cpu = process.cpu_times()
                            cpu = cpu.user + cpu.system
                            # New children contribute their lifetime CPU; initial parent is a baseline.
                            before = previous_cpu.get(key, cpu if previous_time is None else 0)
                            delta_cpu += max(0.0, cpu - before)
                            previous_cpu[key] = cpu
                            totals["rss_bytes"] += process.memory_info().rss
                            totals["threads"] += process.num_threads()
                            totals["process_count"] += 1
                            if hasattr(process, "io_counters"):
                                io = process.io_counters()
                                totals["read_bytes"] += io.read_bytes
                                totals["write_bytes"] += io.write_bytes
                    except (psutil.Error, OSError) as error:
                        errors.add(f"Process sample: {type(error).__name__}")
                observed_cpu += delta_cpu
                gap = now - previous_time if previous_time is not None else None
                totals["cpu_percent"] = delta_cpu / gap * 100 if gap else None
                totals["observed_cpu_seconds"] = observed_cpu
                if not totals["process_count"]:
                    totals = {key: None for key in totals}
                ram = psutil.virtual_memory()
                host = {"ram_used_bytes": ram.total - ram.available,
                        "ram_available_bytes": ram.available, "swap_used_bytes": psutil.swap_memory().used}
                storage = {}
                for path in paths:
                    try:
                        usage = psutil.disk_usage(path)
                        storage[path] = {"total_bytes": usage.total, "used_bytes": usage.used,
                                         "free_bytes": usage.free, "used_percent": usage.percent}
                    except OSError as error:
                        errors.add(f"Storage {path}: {type(error).__name__}")
                row = {
                    "at": _now(), "elapsed_seconds": now - started,
                    "stage": _read(folder / "stage.txt") or "running",
                    "process": totals, "host": host, "container": _container(directory, version),
                    "gpus": _gpu_sample(nvml, devices, errors), "storage": storage,
                    "errors": sorted(errors),
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                summary["samples"] += 1
                if summary["samples"] % 10 == 0:
                    os.fsync(stream.fileno())
                _merge_peaks(summary["process_peaks"], totals)
                _merge_peaks(summary["host_peaks"], {k: v for k, v in host.items() if "available" not in k})
                _merge_peaks(summary["container_peaks"], row["container"])
                for gpu in row["gpus"]:
                    _merge_peaks(summary["gpu_peaks"].setdefault(gpu["uuid"], {}), gpu)
                for path, usage in storage.items():
                    peaks = summary["storage_peaks"].setdefault(
                        path, {"minimum_free_bytes": usage["free_bytes"], "initial_used_bytes": usage["used_bytes"]}
                    )
                    peaks["minimum_free_bytes"] = min(peaks["minimum_free_bytes"], usage["free_bytes"])
                    _merge_peaks(peaks, {k: v for k, v in usage.items() if k != "free_bytes"})
                    peaks["final_used_bytes"] = usage["used_bytes"]
                    peaks["peak_growth_bytes"] = peaks["used_bytes"] - peaks["initial_used_bytes"]
                if gap is not None:
                    summary["max_sample_gap_seconds"] = max(summary["max_sample_gap_seconds"], gap)
                previous_time = now
                if (folder / "stop").exists() or not alive:
                    summary["complete"] = (folder / "stop").exists()
                    summary["stop_reason"] = "requested" if summary["complete"] else "worker_exited"
                    break
                time.sleep(max(0, interval - (time.monotonic() - now)))
    except BaseException as error:
        errors.add(f"Sampler interrupted: {type(error).__name__}: {error}")
        raise
    finally:
        summary.update(finished_at=_now(), elapsed_seconds=time.monotonic() - started,
                       errors=sorted(errors))
        summary["container_at_end"] = _container(directory, version)
        _json(folder / "summary.json", summary)
        if nvml is not None:
            nvml.nvmlShutdown()


class ResourceMonitor:
    """Independent sampler so native CUDA work cannot block Python telemetry."""

    def __init__(self, folder, paths=(), interval=0.5, require_gpu=False):
        self.folder = Path(folder).resolve()
        self.paths = sorted({str(Path(p).expanduser().resolve()) for p in paths if Path(p).expanduser().exists()})
        self.interval = interval
        self.require_gpu = require_gpu
        self.process = None

    def start(self):
        if not 0.1 <= self.interval <= 60:
            raise ValueError("Resource sample interval must be between 0.1 and 60 seconds")
        self.folder.mkdir(parents=True, exist_ok=False)
        arguments = [sys.executable, str(Path(__file__).resolve()), "--collect", str(self.folder),
                     "--pid", str(os.getpid()), "--interval", str(self.interval)]
        for path in self.paths:
            arguments += ["--path", path]
        self.process = subprocess.Popen(arguments, start_new_session=True)
        deadline = time.monotonic() + 15
        samples = self.folder / "samples.jsonl"
        while not samples.exists() or not samples.stat().st_size:
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.stop()
                raise RuntimeError("Resource sampler could not start; install psutil and nvidia-ml-py")
            time.sleep(0.05)
        metadata = json.loads((self.folder / "metadata.json").read_text())
        if self.require_gpu:
            with samples.open() as stream:
                first = json.loads(stream.readline())
            gpu_memory = any(gpu.get("memory_used_bytes") is not None for gpu in first["gpus"])
            if not metadata["gpus"] or not gpu_memory or first["process"]["rss_bytes"] is None:
                self.stop()
                raise RuntimeError("CPU/GPU telemetry unavailable; check permissions and NVML/nvidia-ml-py before paid inference")
        return self

    def stage(self, name):
        # Atomic rename prevents an empty stage being read during an update.
        temporary = self.folder / "stage.partial"
        temporary.write_text(name)
        os.replace(temporary, self.folder / "stage.txt")

    def stop(self):
        if self.process is None:
            return {"complete": False, "errors": ["sampler not started"]}
        (self.folder / "stop").touch()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        path = self.folder / "summary.json"
        if path.exists():
            return json.loads(path.read_text())
        return {"complete": False, "errors": ["sampler ended without summary; retain raw samples"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--interval", default=0.5, type=float)
    parser.add_argument("--path", action="append", default=[])
    args = parser.parse_args()
    _collect(args.collect, args.pid, args.interval, args.path)
