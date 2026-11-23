"""Local-only NVIDIA driver, device, compiler and tile runtime inventory."""
import ctypes as C
import ctypes.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return {"code": p.returncode, "output": p.stdout.strip()}


def driver_devices():
    info = {"devices": []}
    try:
        lib = C.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")

        def call(name, *args):
            code = getattr(lib, name)(*args)
            if code:
                raise RuntimeError(f"{name} returned CUDA driver error {code}")

        call("cuInit", 0)
        count, version = C.c_int(), C.c_int()
        call("cuDriverGetVersion", C.byref(version))
        call("cuDeviceGetCount", C.byref(count))
        info["driver_api_version"] = version.value
        for ordinal in range(count.value):
            device, major, minor = C.c_int(), C.c_int(), C.c_int()
            name, memory = C.create_string_buffer(256), C.c_size_t()
            call("cuDeviceGet", C.byref(device), ordinal)
            call("cuDeviceGetName", name, len(name), device)
            call("cuDeviceComputeCapability", C.byref(major), C.byref(minor), device)
            call("cuDeviceTotalMem_v2", C.byref(memory), device)
            bus = C.create_string_buffer(32)
            call("cuDeviceGetPCIBusId", bus, len(bus), device)
            info["devices"].append({"ordinal": ordinal, "name": name.value.decode(), "pci_bus_id": bus.value.decode(),
                                    "cc": f"{major.value}{minor.value}", "memory_bytes": memory.value})
    except (AttributeError, OSError, RuntimeError) as exc:
        info["error"] = str(exc)
    return info


def inventory():
    roots = [Path(os.environ[k]) for k in ("CUDA_HOME", "CUDA_PATH") if os.environ.get(k)]
    roots += [Path("/usr/local/cuda"), Path("/opt/cuda")]
    nvcc = next((str(p / "bin/nvcc") for p in roots if os.access(p / "bin/nvcc", os.X_OK)), None) or shutil.which("nvcc")
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "driver": driver_devices(), "nvcc": nvcc, "host_cxx": shutil.which(os.environ.get("CXX", "c++"))}
    if nvcc:
        roots.insert(0, Path(nvcc).resolve().parents[1])
        info["nvcc_version"] = command([nvcc, "--version"])
        info["nvcc_arches"] = command([nvcc, "--list-gpu-arch"])
    info["runtime_headers"] = [str(h) for root in dict.fromkeys(roots)
                               for h in [root/"include/cuda_runtime.h", *root.glob("targets/*/include/cuda_runtime.h")] if h.is_file()]
    for name in ("ptxas", "tileiras", "compute-sanitizer"):
        path = next((str(p/"bin"/name) for p in roots if os.access(p/"bin"/name, os.X_OK)), None) or shutil.which(name)
        info[name] = path
    if shutil.which("nvidia-smi"):
        query = ["--query-gpu=index,name,driver_version,memory.total", "--format=csv"]
        info["nvidia_smi"] = command(["nvidia-smi", *query])
        if info["nvidia_smi"]["code"]:
            for device in info["driver"]["devices"]:
                device["nvidia_smi"] = command(["nvidia-smi", "-i", device["pci_bus_id"], *query])
    elif sys.platform == "darwin" and shutil.which("system_profiler"):
        info["display"] = command(["system_profiler", "SPDisplaysDataType"])
    info["tile_import"] = command([sys.executable, "-c",
        "import cuda.tile, cupy, numpy; from importlib.metadata import version; "
        "print('cuda-tile',version('cuda-tile'),'cupy',cupy.__version__,'numpy',numpy.__version__)"])
    info["triton_import"] = command([sys.executable, "-c",
        "import triton, torch, numpy; print('triton',triton.__version__,'torch',torch.__version__,"
        "'torch_cuda',torch.version.cuda)"])
    return info
