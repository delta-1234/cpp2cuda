#!/usr/bin/env bash
# Invoked inside private user, mount, PID and network namespaces on Linux.
set -euo pipefail
root=$1
work=$2
python=$3
prefix=$4
shift 4
mount --make-rprivate /
mount -t tmpfs -o mode=755 tmpfs "$root"
bind_ro() {
    local src=$1 dest=$2
    mkdir -p "$root$dest"
    mount --bind "$src" "$root$dest"
    mount -o remount,bind,ro "$root$dest"
}
for dir in /bin /sbin /usr /lib /lib64 /etc; do
    if [[ -d $dir ]]; then bind_ro "$dir" "$dir"; fi
done
# Only the selected Python runtime is exposed from the user's home.
for dir in "$prefix" "$(readlink -f "$prefix")" "${CUDA_HOME:-/usr/local/cuda}"; do
    if [[ -d $dir && ! -d $root$dir ]]; then bind_ro "$dir" "$dir"; fi
done
mkdir -p "$root"/{dev/shm,proc,sys,tmp,work}
mount -t sysfs -o ro,nosuid,nodev,noexec sysfs "$root/sys"
for node in /dev/null /dev/zero /dev/random /dev/urandom /dev/nvidia*; do
    if [[ -c $node ]]; then
        touch "$root$node"
        mount --bind "$node" "$root$node"
    elif [[ -d $node ]]; then
        bind_ro "$node" "$node"
    fi
done
mount -t proc proc "$root/proc"
mount -t tmpfs -o mode=1777 tmpfs "$root/dev/shm"
bind_ro "$work" /work
mount --bind "$work/runs" "$root/work/runs"
mkdir -p "$root/tmp/agent-home"
cd /
# Drop the namespace capabilities across exec and keep subprocesses from regaining them.
exec chroot "$root" "$python" -IS -c '
import ctypes, os, sys
lib = ctypes.CDLL(None, use_errno=True)
if lib.prctl(38, 1, 0, 0, 0) != 0: raise OSError(ctypes.get_errno(), "no_new_privs")
for cap in range(64):
    if lib.prctl(24, cap, 0, 0, 0) != 0 and ctypes.get_errno() != 22:
        raise OSError(ctypes.get_errno(), "drop bounding capability")
os.chdir("/work")
os.environ["HOME"] = "/tmp/agent-home"
os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
' "$@"
