# FactorPort 量化因子 CUDA 迁移与优化工作流

通用移植入口为 `bash workflow/cuda.sh --agent --source <程序.cpp>` 或 `--agent --spec <任务.json>`。原因子通过 `--generate-cuda example/order-fill-rate-spread/OrderFillRateSpread.cpp` 进入同一工具循环。`--agent --inspect --task '问题'` 支持无源码的节点分析。模型根据本地观测选择探测、查阅手册、实验、生成用例、写代码、编译和验证。控制器以独立 CPU 结果和真实 GPU 启动记录验收。

在通用入口追加 `--backend pi` 可选择 Pi RPC 后端。Pi 用配置的模型 API 调用当前角色的 `fp_*` 工具。控制器执行工具并验收，运行中的 agent 不下载依赖。Pi 的离线准备和计算节点命令见 README。

编程手册位于 `offline_docs/nvidia/`，包含原版 PDF、可按页检索的文本和 SHA-256 清单。资料准备使用 `workflow/prepare_docs.py`，运行中的 agent 只查询本地副本，先核对资料版本与工具链。工具协议、通用输入输出约定和示例参见 README。收到错误时可通过 `diagnostics`、`log` 和 `test_case` 定位失败，修改后全量验证。检查点与回退语义详见 `offline_docs/agent-repair.md`。

因子入口默认在完整转写后运行两个专项优化 agent。通用任务用 `--optimize` 开启，已有候选通过 `--candidate` 提供。先用目标设备建立正确性、端到端和 Nsight 内核基准，再依次运行 SIMT 与 Tile 会话。每次 `trial` 须说明基于硬件的优化假设，控制器以同一用例实测选版。检查 `report.json` 的 `optimization` 字段。可用 `--backend transformers` 选择预装的 Hugging Face ReAct 工具运行时，命令见 README。

以下步骤描述订单因子及既有实现的维护和专项基准。`--agent '维护任务'` 使用这一流程。CPU 判定规则在 `example/order-fill-rate-spread/OrderFillRateSpread.cpp` 和 `example/order-fill-rate-spread/factor_replay.hpp`，窗口归约的验收入口为 `workflow/run.py`，流式路径的验收入口为 `workflow/check_stream.py` 和 `workflow/scale.py`。

### 1. 先盘点环境

在计算节点运行 `python3 workflow/run.py --probe-only`。读取本次打印的 `report.json`，确认 CUDA Driver API 是否能枚举目标 GPU，记录型号、计算能力、显存、驱动 API 版本、`nvcc --version`、可编译架构、运行时头文件及 Triton、cuTile 的导入状态。输出目录会为每次运行单独创建。`--device` 使用当前进程中的可见序号。

存在多套 Toolkit 时按实测能力选择 `CUDA_HOME`。需要时通过 `NVCC_CCBIN` 指向兼容的主机编译器。若整机的 `nvidia-smi` 查询因单卡异常失败，可单独查询健康卡，按 UUID 设置 `CUDA_VISIBLE_DEVICES` 后重新探测。无 GPU 时继续完成 CPU 真值和窗口打包验证。有 `nvcc` 时仍尝试本地编译。只有实际设备运行和逐行比对通过，才把该路径记为已验证。

### 2. 读取离线材料并建立基准

资料来源限于仓库中的代码、`offline_docs/`、本机 Toolkit 头文件及样例，以及任务开始前已提供的文档。整个任务禁用网页搜索、远程文档、包索引和在线安装。记录用到的版本及本地路径。资料不足时检索本地头文件、编译器帮助和编译诊断。依赖缺失要按路径记录，并继续测试已满足条件的实现。

读取 README 中的输入格式和因子定义。先按原始 CPU 路径回放。保持同毫秒事件顺序、每只证券的独立状态、100 笔委托窗口、500 笔成交预热、已滑出窗口的成交引用以及单侧缺失值。将所给样例的 CPU 输出与 `example/order-fill-rate-spread/data/factor_output.csv` 比较。

### 3. 确定并行边界，再分别转写

端到端任务优先使用 `example/order-fill-rate-spread/stream/replay.cpp`。每只证券维护 100 笔活动委托。CPU 对照 `cpu` 增量维护四个量并直接计算比率，`legacy` 按原算法逐窗口扫描。GPU 每段首行接收绝对值，其余行接收增减量。SIMT、共享内存 Tile 及预编译的 Triton Tile 在设备上分别作段内前缀和。每批最多 8,192 个待输出时点，结果按原始到达顺序写入最终 CSV。常驻 `--serve` 模式跨文件复用 CUDA 资源，各输入文件独立重建交易状态。

在计算节点用 `bash workflow/cuda.sh --build-stream` 编译。用 `python workflow/check_stream.py --exe runs/stream/replay --with-triton` 验证空文件头、单侧委托、同一进程多次回放、多证券及批次边界。用 `bash workflow/cuda.sh --scale --core 6 --repeat 7` 在同一 CPU 亲和性、同一磁盘和同一原始输入下测试冷启动及常驻进程。报告两种 CPU 对照、输出行数、事件数和实测交叉区间。不要把内核加速比用作端到端加速比。

以下为原始窗口路径，可用来诊断打包输入、拆分 I/O 开销和比对旧实现。

逐笔更新有先后依赖。CPU 负责匹配和校验委托、更新累计成交量，为每个待输出时点冻结当时的窗口。同时写下 `time_ms,symbol,trade_count`。GPU 对这些独立窗口做并行归约、除法和作差。主机打包过程中不计算成交率。修改因子规则时，先核对这一划分是否仍然成立。

- **SIMT** 入口为 `example/order-fill-rate-spread/cuda/simt.cuh`。一组 32 个线程处理一个窗口，每个线程读取四个列位置，使用 warp shuffle 累加四个整数统计量。
- **分块 Tile（CUDA C++）** 入口为 `example/order-fill-rate-spread/cuda/shared_tile.cuh`。每个线程块负责 8 个窗口 × 128 列，使用共享内存归约。处理尾块时，无效行仍须参与块内同步。
- **Triton Tile DSL** 入口为 `example/order-fill-rate-spread/cuda/triton_tile.py`。使用二维块加载、沿订单维度的 `tl.sum` 和尾块掩码。工作流检测到本地 Triton、PyTorch 和 NumPy 后，在同一 GPU 和同一窗口数据上独立执行。
- **NVIDIA cuTile DSL** 入口为 `example/order-fill-rate-spread/cuda/cutile_backend.py`。通过二维 tile load、沿列归约和 tile store 实现同一公式。安装了本地 cuTile、CuPy、NumPy 后实际尝试 JIT 和运行。需要专门验收此编程模型时，运行时添加 `--require-cutile`。其状态单独记录。

### 4. 验证结果后测性能

运行 `python3 workflow/run.py --input example/order-fill-rate-spread/data/sample_ticks.csv --repeat 7`。依次检查 CPU、独立主机窗口解码、各 GPU 路径的 CSV。多证券输入按逐笔到达顺序对齐每一行，三个键及 `NA` 必须匹配，三个数值的绝对容差约为输出末位的一个单位（1e-8）。每条 DSL 路径均单独尝试，某个 DSL 缺少依赖不影响其他路径。边界输入覆盖预热、单侧窗口、同毫秒事件、跨窗口成交、64 位求和、尾块和空输入。

若出现差异，先在该次运行的 `main/` 或 `edge/` 中定位首个不匹配的键。先检查 `cpu.csv` 和 `host_windows.csv`，再检查内核的行索引、尾块、整数宽度、同步及输出顺序。修改后重跑两条 CUDA 路径。可使用本地 `compute-sanitizer --tool memcheck --error-exitcode 9` 检查原生后端，命令参数依次为后端程序、`windows.bin`、临时输出文件和设备序号。

### 5. 报告可复现的结果

从 `report.json` 提取硬件与工具版本、样本量、各实现状态、首个失败原因和重复次数。分别报告 GPU event 内核中位数、相同已打包数据的单线程 C++ 计算中位数，以及从原始 CSV 到结果 CSV 的端到端中位数。端到端包括进程启动、窗口打包、文件读写、主机设备传输和结果格式化。报告实际的加速或减速。设备不可用时将该项保留为未测。

退出码 0 表示本次要求的路径通过（`--probe-only` 仅完成盘点）。1 表示某个实现或校验失败。2 表示缺少必需运行条件。SIMT 与 CUDA C++ 分块 Tile 是默认必验路径。即使其中一条失败，也要执行其余能测试的路径。

公开文档、文件名和验证报告以硬件型号及工具版本描述环境。发布前移除主机名、SSH 别名、GPU UUID 和用户目录。原始节点日志保存在 Git 忽略的 `runs/` 中。

示例目录 `example/` 中的性能表、图和随附报告只发布排除 I/O 的计算耗时及加速比，注明计时范围、样本量、重复次数和基线。工作流的端到端计时留在运行目录中供选版与诊断。
