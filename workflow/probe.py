"""Local-only NVIDIA driver, device, compiler and tile runtime inventory."""
import ctypes as C
import ctypes.util
import json
import re
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
            # CUdevice_attribute values from the Toolkit cuda.h public driver ABI.
            attributes = {}
            for key, attr in {"sm_count":16, "warp_size":10, "max_threads_per_block":1,
                              "max_threads_per_sm":39, "shared_bytes_per_block":8,
                              "shared_bytes_per_sm":81, "optin_shared_bytes_per_block":97,
                              "registers_per_sm":82, "l2_bytes":38, "memory_bus_bits":37}.items():
                value = C.c_int()
                if lib.cuDeviceGetAttribute(C.byref(value), attr, device) == 0:
                    attributes[key] = value.value
            info["devices"].append({"ordinal": ordinal, "name": name.value.decode(), "pci_bus_id": bus.value.decode(),
                                    "cc": f"{major.value}{minor.value}", **attributes, "memory_bytes": memory.value})
    except (AttributeError, OSError, RuntimeError) as exc:
        info["error"] = str(exc)
    return info


def cupti_paths(toolkit):
    roots = [toolkit/"extras/CUPTI", toolkit, *sorted((toolkit/"targets").glob("*"))]
    headers = [p for root in roots for p in (root/"include", root/"include/cupti")]
    libraries = [p for root in roots for p in (root/"lib64", root/"lib", *sorted((root/"lib").glob("*-linux-gnu")))]
    return {"include":next((str(p) for p in headers if (p/"cupti.h").is_file()),None),
            "library":next((str(p) for p in libraries if (p/"libcupti.so").is_file()),None)}


def toolchain(preferred=None):
    roots = [Path(os.environ[k]) for k in ("CUDA_HOME", "CUDA_PATH") if os.environ.get(k)]
    roots += [Path("/usr/local/cuda"), Path("/opt/cuda")]
    if preferred:
        roots.insert(0, Path(preferred))
    nvcc = next((str(p / "bin/nvcc") for p in roots if os.access(p / "bin/nvcc", os.X_OK)), None) or shutil.which("nvcc")
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvcc": nvcc, "host_cxx": shutil.which(os.environ.get("CXX", "c++"))}
    if nvcc:
        roots.insert(0, Path(nvcc).resolve().parents[1])
        info["cupti"] = cupti_paths(roots[0])
        info["nvcc_version"] = command([nvcc, "--version"])
        info["nvcc_arches"] = command([nvcc, "--list-gpu-arch"])
    info["runtime_headers"] = [str(h) for root in dict.fromkeys(roots)
                               for h in [root/"include/cuda_runtime.h", *root.glob("targets/*/include/cuda_runtime.h")] if h.is_file()]
    for name in ("ptxas", "tileiras", "compute-sanitizer", "nsys", "cuobjdump"):
        path = next((str(p/"bin"/name) for p in roots if os.access(p/"bin"/name, os.X_OK)), None) or shutil.which(name)
        info[name] = path
    return info


def inventory():
    info = toolchain()
    info["driver"] = driver_devices()
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


def probe_topic(topic):
    if topic=="gpu":
        result = driver_devices()
    elif topic=="toolchain":
        result = toolchain(os.getenv("CUDA_HOME"))
        result["nvcc_ccbin"] = os.getenv("NVCC_CCBIN")
        roots = [*Path("/usr/local").glob("cuda*"),Path("/opt/cuda")]
        if result["nvcc"]:
            roots.append(Path(result["nvcc"]).resolve().parents[1])
        result["installed_toolkits"] = sorted({str(p.resolve()) for p in roots if os.access(p/"bin/nvcc",os.X_OK)})
        compilers = [str(p) for d in (Path("/usr/bin"),Path("/usr/local/bin")) for p in d.glob("*++*")
                     if re.fullmatch(r"(?:g|clang)\+\+(?:-[0-9.]+)?",p.name) and os.access(p,os.X_OK)]
        result["available_ccbin"] = sorted(set(compilers + ([result["nvcc_ccbin"]] if result["nvcc_ccbin"] else [])))
        for key in ("host_cxx","nvcc_ccbin"):
            if result[key] and shutil.which(result[key]):
                result[key+"_version"] = command([result[key],"--version"])["output"].splitlines()[0]
    elif topic=="system":
        result = {"platform":platform.platform(),"machine":platform.machine(),"cpu_count":os.cpu_count(),
                  "visible_gpu_mask":os.getenv("CUDA_VISIBLE_DEVICES"),
                  "toolkits":[str(p) for p in sorted(Path("/usr/local").glob("cuda*")) if (p/"bin/nvcc").exists()]}
    elif topic=="runtimes":
        result = command([sys.executable,"-c","from importlib.util import find_spec\n"
                          "for m in ('triton','torch','cupy','cuda.tile'):\n"
                          " try: print(m, bool(find_spec(m)))\n except ModuleNotFoundError: print(m, False)"])
    else:
        raise ValueError("unknown probe topic")
    return result



if __name__ == "__main__":
    print(json.dumps(probe_topic(sys.argv[1])))
