# 五点窗口能量示例

这个小程序用于练习从 C++ 源码到 CUDA 候选的完整流程。输入参数是序列长度，每个位置输出以它为中心、最多五个元素的平方和。序列由下标确定，可以直接用 CPU 输出检查窗口边界和 GPU 尾块。[C++ 源码](window_energy.cpp) · [任务清单](task.json)

## 复现迁移与优化

在仓库根目录配置好模型 API 后执行下列命令。`many` 用例的序列长度为 8,193。`--specialist-trials 1` 与下方实测使用的候选数一致，每个专项 agent 至少测试一个候选。

```sh
bash workflow/cuda.sh --agent --backend pi --spec example/window_energy/task.json \
  --optimize --profile-case many --specialist-trials 1
```

## 实测环境与结果

测试在 NVIDIA GeForce RTX 3090 上运行，使用 CUDA Toolkit 11.3（nvcc 11.3.58）和 Nsight Systems 2022.1.3.3。Pi 1.6.0 通过 Chat Completions 接口调用 `qwen3.8-max`。

迁移阶段生成了 SIMT 和共享内存 Tile 两种实现。恢复运行时，控制器从已选检查点重新测量，两位专项 agent 各提交一个候选。最终源码通过了 1、259 和 8,193 个输出的 CPU 逐字节对照、CUPTI 内核启动检查和 Compute Sanitizer 显存检查。[选中的 CUDA 源码](generated/window_energy_cuda.cu) · [测量与选版记录](validation/pi-rtx3090.json)

### 最终实现相对单线程 CPU 的计算加速比

在同一台机器上，以 8,193 个输出为一组，各取 7 次实测的中位数。CPU 使用 Xeon Gold 6248R 的逻辑核 6，GCC 12.3.0 以 `-O3` 编译。计时从计算平方和开始，到写入预分配的内存数组结束。GPU 使用 Nsight Systems 记录的设备时间戳，每次加总该实现的全部内核。文件读写、文本格式化、初始化、内存分配和主机与设备之间的传输均在计时区间之外。

| 实现 | 计算耗时（µs） | 相对单线程 CPU 的加速比 |
| --- | ---: | ---: |
| 单线程 C++ | 43.309 | 1.00× |
| CUDA SIMT | 2.752 | 15.74× |
| CUDA 共享内存 Tile | 5.472 | 7.91× |

SIMT 在一个内核中生成输入并求和。Tile 先生成平方序列，再启动窗口求和内核，表中包含两次启动对应的设备计算时间。三个实现的 8,193 行输出均与原 C++ 程序逐字节一致。[CPU 计算基准](benchmark_cpu.cpp) · [七次原始计时、内核启动数与源码哈希](validation/compute-rtx3090.json)

在已分配 GPU 的节点上，从仓库根目录运行下列命令。将 `CUDA_HOME`、`NVCC_CCBIN`、`NSYS` 设置为本机工具路径，按目标计算能力调整构建参数中的 `86`。

```sh
repo=$PWD
src=$repo/example/window_energy
work=$repo/runs/window-energy-compute
mkdir -p "$work"
c++ -O3 -std=c++17 "$src/benchmark_cpu.cpp" -o "$work/bench"
c++ -O3 -std=c++17 "$src/window_energy.cpp" -o "$work/reference"
bash workflow/build_port.sh "$CUDA_HOME" 86 17 17 "$src/generated" "$work/runs"
(
  set -e
  cd "$work"
  ./reference 8193 > expected
  taskset -c 6 ./bench 8193 > actual-cpu 2> cpu-samples-us.txt
  cmp expected actual-cpu
  for rep in 1 2 3 4 5 6 7; do
    for mode in simt tile; do
      taskset -c 6 python3 "$repo/workflow/profile_kernels.py" "$NSYS" "$mode" 8193 > "$mode-$rep.log"
      cmp expected runs/actual
    done
  done
)
```

`cpu-samples-us.txt` 每行是一次 CPU 计算时间。每个 GPU 日志末行的 `total_kernel_us` 是该次计算的内核总时间。分别取中位数，用 CPU 中位数除以各 GPU 中位数即可复算表中的加速比。

### 专项优化相对各自起点的计算加速比

下表比较迁移过程中每位 agent 接手和提交候选时的内核。两列均为 8,193 个输出、三次独立采样的中位数。这里的加速比以该角色接手的 GPU 版本为基线。候选的最终选版还需满足工作流的其他验收条件。[选版规则](../../README.md#候选选版循环)

| 专项 | 接手时内核耗时（µs） | 候选内核耗时（µs） | 相对起点的计算加速比 | 选版结果 |
| --- | ---: | ---: | ---: | --- |
| SIMT | 5.280 | 2.784 | 1.897× | 采用候选 |
| Tile | 5.728 | 3.072 | 1.865× | 保留起点 |

纠错回归另有可复用的故障候选，以及定向重测、全量验证和检查点回退的记录。[故障候选](repair_fixture/window_energy.cu) · [纠错报告](validation/agent-repair-rtx3090.json)

## 离线准备本次使用的 Pi 版本

本次测试使用官方 Pi 1.6.0 独立版。Linux x86-64 用户可在联网准备机下载同一归档，然后把归档和仓库一起传到计算节点。[Pi 1.6.0 官方发布页](https://github.com/earendil-works/pi/releases/tag/v1.6.0)

```sh
# 准备机
curl -fL -o pi-linux-x64.tar.gz \
  https://github.com/earendil-works/pi/releases/download/v1.6.0/pi-linux-x64.tar.gz

# 离线计算节点，在仓库根目录执行，归档路径按实际位置填写
archive=/path/to/pi-linux-x64.tar.gz
echo "ef77dbe4379b3bdeb425a1e1b862fa3c915c255000dfe4965c98eee96a5765d1  $archive" | sha256sum -c - && \
  mkdir -p runs/deps/pi-1.6.0 && tar -xzf "$archive" -C runs/deps/pi-1.6.0
```
