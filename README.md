# OrderFillRateSpread（买卖委托成交率差）

独立 C++17 程序，读取单个交易日的逐笔委托与成交 CSV，按证券输出卖方成交比例、买方成交比例和两者之差。程序使用 C++ 标准库，内置模拟数据生成器。

## 配置模型 API

计算节点 上的仓库根目录 `.env` 已配置为模型服务入口。修改此文件即可更换模型、地址或密钥。新建仓库可复制 [.env.example](.env.example) 并填入实际值，也可执行 `bash workflow/cuda.sh --configure-api` 交互配置。交互输入的密钥隐藏，文件权限为 0600。实际 `.env` 已加入 Git 忽略规则。

```bash
cd "$(git rev-parse --show-toplevel)"
bash workflow/cuda.sh --check-api
```

| 设置 | 用途 |
| --- | --- |
| `OPENAI_BASE_URL` | 模型 API 的基础地址，保留服务商要求的版本路径。 |
| `OPENAI_MODEL` | 服务提供的模型 ID。 |
| `OPENAI_API_KEY` | Bearer 密钥。免密内网服务可留空。 |
| `FACTOR_API_PROTOCOL` | 默认 `chat`，调用 Chat Completions；设为 `responses` 可调用 Responses API。 |

已导出的同名环境变量覆盖文件值。`FACTOR_API_CONFIG` 可显式指定 dotenv 或 JSON 配置文件。默认先读仓库 `.env`，然后依次尝试 `~/.config/order-fill-rate-cuda/api.json` 和 `~/.env`，选中第一份存在的文件。dotenv 支持单行值、引号和 `export` 前缀，按文本读取。API 连通检查只发一条简短消息。

### 维护已有实现

使用 `bash workflow/cuda.sh --agent '维护任务'` 时，控制器先盘点 GPU 和 Toolkit，将任务、设备信息及项目规则传给所配置的模型。模型可以按需读取仓库源码和 `offline_docs/` 的文本材料、编辑候选副本中已有的 `cuda/` 或 `stream/` 源文件、读取编译错误并继续修复。编译、验证和基准运行在 计算节点 的独立 Linux 网络、挂载和 PID 命名空间内。候选源码对这些子进程只读，工作产物写入该次运行的 `runs/`。模型 API 调用由外层控制器发起，模型接口应支持文本生成。

当前候选版本通过 CPU 对照和基准回放后，控制器才将补丁应用到工作目录。若原文件在运行期间被其他任务修改，补丁保存在运行目录供核对。需要仅生成候选补丁时使用以下命令。

```bash
bash workflow/cuda.sh --agent --proposal-only '检查 SIMT 实现，提出并验证一个针对当前输入的优化。'
```

每次运行会打印 `runs/api-agent-*/report.json` 路径，同目录包含 `changes.patch`、分阶段日志及候选副本。默认最多 24 次模型决策，可用 `--max-steps` 调整。默认验证流式 CPU、SIMT、CUDA C++ Tile 和 AOT Triton；编辑原始打包实现时还会运行相应路径的验证。

## 用 agent 分析计算节点并移植 C++

在 计算节点 的仓库根目录运行下列命令。其他 Linux CUDA 节点可直接用 Python 3.10+ 运行 `workflow/port_agent.py`，通过 `CUDA_VISIBLE_DEVICES`、`CUDA_HOME` 和 `NVCC_CCBIN` 指定调度分配和编译环境。构建隔离使用 Linux 用户、挂载、进程及网络命名空间，需要本机支持非特权 user namespace、`unshare` 和 `ip`。GPU 验收使用 Toolkit 的 CUPTI 和 Compute Sanitizer。

```sh
# 仅分析已有服务器和离线手册
bash workflow/cuda.sh --agent --inspect --task '检查当前 GPU 和编译器，找出适用的共享内存机制，列出参考文档和页码。'

# 转写因子，保留专用的逐笔回放、边界检验和基准
bash workflow/cuda.sh --generate-cuda OrderFillRateSpread.cpp

# 其他 C++ 程序。用例既可随任务提供，也可由模型阅读源码后提出
bash workflow/cuda.sh --agent --spec examples/window_energy/task.json
bash workflow/cuda.sh --agent --source examples/window_energy/window_energy.cpp
```

模型从源文件清单和任务描述开始，每轮选择一个或一组工具。控制器执行后把观测值、文档摘录或错误发回模型。模型可调整并行方案、编译器选择和文件组织。默认允许 26 轮，`--max-steps` 可调整。每轮允许最多 6 个工具动作，遇错即把诊断返回模型。接口使用流式文本和结构化工具调用，复用仓库的模型配置。

| 工具 | 实现与用途 |
| --- | --- |
| `probe`、`select_toolchain` | 按需探测真实设备、计算能力、体系结构、编译器、可见 Toolkit 或 Python 运行时。可以从探测到的 Toolkit 和宿主编译器中选择组合。 |
| `catalog`、`search`、`page` | 查询本地手册版本，按英文术语或 API 符号搜索，按文档 ID 和 PDF 物理页码读取。返回正文在原 PDF 中可核对。 |
| `read`、`case`、`baseline` | 阅读参考源码及项目内双引号头文件。模型可补充参数或小型文本输入，控制器用原始 C++ 程序独立生成预期输出。 |
| `experiment` | 在单独的离线目录编译并运行一次小型 CUDA 探针，用返回值或编译诊断判断功能是否可用。 |
| `write`、`edit`、`remove`、`build` | 在候选目录创建、局部修复或移除源文件，按模型选取的 C++ 标准、探测到的架构编译和链接。 |
| `verify`、`benchmark`、`finish` | 两条内核分别与 CPU 对照，使用 CUPTI 核对实际启动符号，使用 Compute Sanitizer 检查显存访问。通过后各后端轮换执行三次冷启动计时。修改候选代码后须重新构建和验收。 |
| `blocked` | 记录 GPU 或编译环境确实缺失的观测证据。有主机编译器时仍可建立 CPU 基准。 |

**通用任务约定**　CPU 是可编译的 C++17 命令行程序。GPU 可执行文件接收 `simt` 或 `tile` 加原来的参数。默认逐字节比较标准输出。对于写文件的程序，用例 `args` 中的 `{output}` 代表由控制器分配的结果文件。可通过 `--input 相对路径` 提供一个文件参数，可重复指定。更复杂的用例采用 JSON 清单。通用清单默认为 `stdio`，兼容原订单因子合同的源码可通过 `"profile": "factor"` 复用专项验收。`source`、额外的 `units`、`headers` 和 `fixtures` 均相对于清单目录，`cases` 包含名字与参数数组。模型新增的测试可以使用 `{input}` 注入自拟的文本数据。随清单提供的用例始终保留。当前通用核验面向确定性输出，专用因子另有 CSV 语义、无效输入、窗口和预热边界检验。[清单示例](examples/window_energy/task.json)

**运行记录**　每次产生独立的 `runs/cuda-agent-*/`。其中 `candidate/port/` 是本次实际生成的源码，`candidate/runs/replay` 是可执行文件，`reply-*.txt` 和 `tools.jsonl` 记录模型回复和工具观测，`report.json` 包含硬件、源码哈希、动作、验证及计时。状态 `analyzed` 对应环境分析，`verified` 对应完成 GPU 验收和基准。性能是否优于 CPU 通过实测值判断。工具子进程在无外网的命名空间中运行，模型 API 由外层控制器调用。

`--generate-cuda` 默认使用上述 agent。`--generate-cuda OrderFillRateSpread.cpp --per-file` 可复现以前逐文件生成的流程。单独的 `--agent '维护任务'` 继续用于修改仓库中已存在的实现。

### 按实际 GPU 分别优化 SIMT 和 Tile

`--generate-cuda` 默认在转写验收后依次启动 SIMT 和 Tile 两个专项 agent。用 `--no-optimize` 只做转写、验收和端到端测试。通用 `--agent --source` 或 `--agent --spec` 配合 `--optimize` 可使用相同的专项流程。已有可编译的候选可以直接作为起点。

```sh
# 在 计算节点 上复用已安装的开源 Hugging Face ReAct 后端和仓库的模型 API
WORKFLOW_PYTHON="$(command -v python3)" \
  bash workflow/cuda.sh --generate-cuda OrderFillRateSpread.cpp \
    --candidate generated/order-fill-rate --backend transformers --profile-case orders-131072
```

两个 agent 各有独立对话和最多 12 轮工具调用决策，可用 `--specialist-steps` 调整。深度搜索可加 `--specialist-trials 3 --specialist-steps 24`，要求每个角色至少实际验证、测量三种不同的候选版本。SIMT agent 重点检查线程到数据的映射、合并访问、分支、warp 归约及寄存器压力。Tile agent 重点检查分块形状、共享内存复用、bank conflict、同步和尾块。两者都能看到目标卡的 SM 数、warp 大小、线程与共享内存上限、L2 容量，以及实测的内核启动次数、块形状、每线程寄存器数和共享内存用量。

`trial` 接收本次改动的硬件依据，然后重建、全量验证、测端到端时间，再运行本地 Nsight Systems。默认选择 CPU 基准中耗时最长的用例，也可以用 `--profile-case` 指定能代表生产负载且会启动 GPU 的用例。两个后端在同一卡上交替采样，各采三次，内核指标为该进程所有 CUDA 内核运行时间之和的中位数。冷启动端到端基准在独立的无 profiler 进程中测量，包含文件 I/O 和 CUDA 初始化。

默认保留条件为目标内核中位数至少下降 3%，另一种内核及每个基准用例上两个 GPU 模式的端到端耗时增幅均在 5% 以内。较慢或错误的草稿可供诊断，下一位 agent 从已选版本开始。若某个 agent 未能测完一个新版本，报告将标为未完成，并保留通过验证的候选。查看 `report.json` 的 `optimization.baseline`、`roles.simt.trials`、`roles.tile.trials` 和 `optimization.selected`，可核对每个提案的假设、源码检查点、测量值和取舍。

`--backend builtin` 是默认的轻量工具循环。`--backend transformers` 使用 计算节点 上已有的 Transformers 4.41.2 `ReactJsonAgent` 管理动作解析、工具调度与对话记忆。两种后端均读取常规 API 配置，通过同一控制器离线执行工具；可用的 Python 环境需预先准备好对应运行库。

### 诊断和修复已有 CUDA 候选

```sh
bash workflow/cuda.sh --agent --spec examples/window_energy/task.json \
  --candidate examples/window_energy/repair_fixture --task '找出与 CPU 不一致的原因，修复后重测并完整验收。'
```

示例的 `repair_fixture` 是用于验证纠错的带缺陷候选。也可以传入自己的源码目录或上次运行的 `candidate/port`。`--candidate` 只把源文件复制到新运行目录。失败后，模型可查具体日志、后端和用例，按唯一匹配片段修改代码，再用 `test_case` 定向重测。控制器识别重复和交替失败，为通过全量验收的源码建立检查点。轮数耗尽后的回退版本需重新通过当前用例，报告使用 `verified_fallback` 标记，并保留失败的草稿。[纠错流程与参考实现](offline_docs/agent-repair.md)

计算节点 已使用实际的 `qwen3.8-max` 跑通节点分析、带清单的通用程序和由模型生成用例的裸 C++ 程序。原因子从零生成实验在第 16 轮产生了通过验收的源码，随后继续尝试优化，早期控制器在用尽轮数后退出。该轮源码已在带检查点的控制器上重新通过 7 组完整验收和计时。独立的故障注入实验中，模型定位了 Tile 第二个线程块的索引错误，修改一行，先定向重测再通过全量验收。[工具调用与因子验证记录](validation/port-agent-rtx3090.json) · [自主修复与回退记录](validation/agent-repair-rtx3090.json)

### 直接运行已验证的模型版本

先前的逐文件实验使用 `qwen3.8-max` 分三次生成了 [主机程序](generated/order-fill-rate/host.cpp)、[GPU 接口](generated/order-fill-rate/gpu.hpp) 与 [CUDA 内核](generated/order-fill-rate/kernels.cu)。三个文件均可在 计算节点 的原始回复中按 SHA-256 核对。SIMT 每个线程计算一行，Tile 使用 128 个线程在共享内存中归约订单，每块最多处理 8 行。主机复用原头文件的逐笔回放，按输出时点在内存中打包窗口，每 65,536 行为一批。同一进程中的批次复用显存。

```sh
cd "$(git rev-parse --show-toplevel)"
bash workflow/cuda.sh --build-generated
bash workflow/cuda.sh --generated simt sample_ticks.csv runs/model-simt.csv
bash workflow/cuda.sh --generated tile sample_ticks.csv runs/model-tile.csv
cmp factor_output.csv runs/model-simt.csv
cmp factor_output.csv runs/model-tile.csv
```

输入和输出使用不同文件，输出路径会被覆盖。单次回放包含 CUDA 初始化。复现实验时，可以在 `--generate-cuda OrderFillRateSpread.cpp` 后加 `--extra-orders 524288 2097152`，把两个更大规模加入新生成候选的基准测试。

在 RTX 3090 上，两种生成的内核都通过了逐字节 CPU 对照，Compute Sanitizer 的边界样本检查均为 0 错误。下表是每种实现各启动三个新进程后取的 CSV 到 CSV 耗时中位数，包含解析、输出及 GPU 初始化。CPU 对照为输入的原始 C++ 程序。

| 委托笔数 | 输出行数 | CPU（毫秒） | SIMT（毫秒） | Tile（毫秒） |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | 22,794 | 72 | 218 | 213 |
| 131,072 | 94,880 | 254 | 466 | 442 |
| 524,288 | 381,472 | 988 | 1,355 | 1,358 |
| 2,097,152 | 1,530,728 | 4,330 | 5,444 | 5,456 |

四个实测规模上，原始 CPU 程序的端到端耗时均较短。生成版本仍需为每个输出时点扫描和复制窗口，再向设备传输整批窗口。[源码哈希、内核启动记录和各次计时](validation/generated-cuda-rtx3090.json)

## 优先使用流式回放

在 计算节点 上使用下列入口，可直接从原始逐笔 CSV 写到最终因子 CSV。SIMT、CUDA C++ 分块 Tile 和提前编译的 Triton Tile 共用一个 C++ 前端，执行阶段使用同一套 CSV 解析、委托检查和输出格式化。

```sh
cd "$(git rev-parse --show-toplevel)"
bash workflow/cuda.sh --build-stream
bash workflow/cuda.sh --stream simt sample_ticks.csv runs/simt.csv
bash workflow/cuda.sh --stream tile sample_ticks.csv runs/tile.csv
bash workflow/cuda.sh --stream triton sample_ticks.csv runs/triton.csv
bash workflow/cuda.sh --stream cpu sample_ticks.csv runs/incremental-cpu.csv
```

`--serve` 允许多次回放共用 GPU 上下文、CUDA stream、页锁定主机内存和显存。输入协议为每行一个「输入路径、制表符、输出路径」。进程收到 EOF 后退出，每个 CSV 的委托状态独立。返回 `READY` 表示工作进程已启动，`OK` 后依次为输出行数、批次数、分段数及本进程一次性初始化耗时（毫秒）。

```sh
printf '%s\t%s\n' sample_ticks.csv runs/first.csv sample_ticks.csv runs/second.csv |
  bash workflow/cuda.sh --stream simt --serve
```

每个 GPU 批次最多 8,192 个时点，同一证券每段最多 128 行。段首以当前委托状态为基准，段内记录四个 int64 变化量，GPU 计算前缀和及成交率。只有分段索引按证券分组，因子按逐笔到达顺序输出。CPU 对照 `cpu` 直接增量维护四个统计量，`legacy` 保留原来的 100 笔窗口扫描。旧打包后端仍可用于逐阶段诊断。

在 计算节点 上，主实验覆盖 13 至 1,530,669 行因子输出。约 8.3 万输出行时，新建 GPU 进程开始稳定超过原扫描 CPU。在测试上限，流式 GPU 与增量 CPU 的常驻端到端耗时接近。[规模、开销拆分、两组 CPU 基线和交叉点](validation/stream-results.md)

```sh
bash workflow/cuda.sh --scale --core 6 --repeat 7
```

命令会生成固定种子的限价撮合行情，各规模共享同一行情前缀。每次 CSV 回放都校验最终结果的 SHA-256。修改规模可用 `--sizes 2600,32768,131072`（单位为提交委托数），只测常驻模式可用 `--phase warm`。运行目录保留输入、最近一次各后端的输出和 `scale.json`。复现开销拆分可运行 `workflow/profile_io.py --help`。

## 离线 CUDA agent 工作流

把本仓库作为 agent 的工作目录，给它指令「按 AGENTS.md 在本机离线验证 OrderFillRateSpread 的 SIMT 与 Tile 实现」。阶段、排错分支和验收条件见 [AGENTS.md](AGENTS.md)。[离线资料目录](offline_docs/README.md) 包含 计算节点 上的 Toolkit 资料位置和共享数据格式。

```sh
python3 workflow/run.py --probe-only
python3 workflow/run.py --input sample_ticks.csv --repeat 7
```

运行器需要 Python 3.8+ 和 C++17 主机编译器。两条原生后端通过本地 nvcc 编译，使用 CUDA C++14。每次运行打印独立的 `runs/run-*/report.json` 路径，并在该目录保留 CPU 和 GPU 结果。未连 GPU 的机器可完成 CPU 及打包验证。运行器不下载依赖。严格物理隔离时，可在断网计算节点或已预装工具链、禁用网络的本地容器内运行。

### 在 计算节点 复现

从仓库根目录运行，按本地安装情况设置 `WORKFLOW_PYTHON`、`CUDA_HOME` 和 `NVCC_CCBIN`。`--device 0` 选择调度器分配的第一块可见设备。

```sh
cd "$(git rev-parse --show-toplevel)"
bash workflow/cuda.sh --device 0 --repeat 7
```

### 原始窗口打包路径的实测结果

RTX 3090（sm_86，24 GiB），驱动 550.76，CUDA Toolkit 11.3，Triton 3.2.0。以下为 912 个窗口、7 次运行的中位数。每个窗口包含 100 笔委托。三种 GPU 实现的样例和边界输出都通过逐行比对。[环境检测](validation/rtx3090-probe.json) · [完整验证报告](validation/rtx3090.json)

| 实现 | 已打包窗口的计算 | 原始 CSV 至结果 CSV |
| --- | ---: | ---: |
| 单线程 C++ | 0.879 ms | 36.00 ms |
| CUDA C++ SIMT，warp shuffle | 0.0092 ms | 321.19 ms |
| CUDA C++ 分块 Tile，共享内存 | 0.0092 ms | 523.80 ms |
| Triton Tile DSL，二维块归约 | 0.1556 ms | 4611.18 ms |

同一份已打包窗口上，两种 CUDA C++ 内核的计算时间约为单线程 C++ 的 1/95，Triton Tile 约为 1/5.6。GPU 数据来自包围预热后一次发射的 CUDA events。完整流程计入进程启动、窗口打包、落盘、CUDA 初始化、传输和输出格式化，Triton 另含 Python 框架导入。这份小样本从原始 CSV 开始处理时，CPU 耗时最短。

两种 CUDA C++ 后端另通过 5,472 行跨批次测试及边界输入的显存检查。[跨批次报告](validation/rtx3090-multibatch.json) · [SIMT memcheck](validation/simt-memcheck.txt) · [Tile memcheck](validation/shared_tile-memcheck.txt)

仓库另含 NVIDIA cuTile DSL 实现 `cuda/cutile_backend.py`。服务器的验证环境缺少 `cuda.tile`，该路径尚未实机验证。在配有相应本地依赖的节点上，添加 `--require-cutile` 可将其列为必验项。

## 运行样例

在本目录下运行。本地随附的可执行文件适用于 Apple Silicon Mac。在 Linux 仓库中，请先按下方通用命令编译。

```sh
./order_fill_rate_spread sample_ticks.csv > factor_output.csv
head -n 4 factor_output.csv
```

省略输入文件时默认读取当前目录的 `sample_ticks.csv`。这台 Mac 上重新编译可使用以下命令，显式选择与本机链接器兼容的 SDK。

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk OrderFillRateSpread.cpp -o order_fill_rate_spread
```

在已配置 C++17 的 Linux 环境或使用默认 SDK 的常规开发环境中，可使用通用编译命令。

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic OrderFillRateSpread.cpp -o order_fill_rate_spread
```

## 模拟行情

`sample_ticks.csv` 使用种子 `8001` 生成。两个证券代码是示例标识，价格、数量和时间均为模拟值。生成器从 09:30 起提交限价单，价格以分计，委托量为 100 至 1,200 股的整百股。能成交的委托先匹配对手方最优报价，同价按到达顺序撮合，以在簿委托的价格成交。一次委托可以产生多笔成交，未成交部分继续排队。样例仅包含连续竞价的委托和成交。

| 示例证券 | 委托笔数 | 成交笔数 | 因子输出行数 |
| --- | ---: | ---: | ---: |
| 000001.SZ | 1,268 | 928 | 429 |
| 000002.SZ | 1,332 | 982 | 483 |
| 合计 | 2,600 | 1,910 | 912 |

生成另一组行情并计算因子（在同一构建环境下，相同种子可复现相同数据）：

```sh
./order_fill_rate_spread --generate 42 > ticks_42.csv
./order_fill_rate_spread ticks_42.csv > factors_42.csv
```

## 输入格式

首行列名及顺序固定。每个文件从一个交易日的首条事件开始，按时间非递减排列。同毫秒内按行顺序处理，委托必须先于引用它的成交出现。

```csv
time_ms,symbol,type,price_cent,volume,bid_order,ask_order
34200010,000001.SZ,S,1000,500,0,1
34200020,000001.SZ,B,1001,200,2,0
34200020,000001.SZ,T,1000,200,2,1
```

| 字段 | 含义 |
| --- | --- |
| `time_ms` | 从当日零点起的毫秒数，34200000 表示 09:30:00.000。 |
| `symbol` | 证券代码。每只证券独立维护委托号、窗口和成交计数。 |
| `type` | `B` 买入限价委托，`S` 卖出限价委托，`T` 成交。 |
| `price_cent` | 委托限价或成交价，单位为分，取正整数。 |
| `volume` | 该笔委托量或该次成交量，单位为股，取正整数。 |
| `bid_order`、`ask_order` | 买、卖委托号。新委托只填写本侧的正整数编号，另一侧填 0。成交填写双方编号。同一证券当日的委托号唯一。 |

示例中，编号 2 的买单以 10.00 元成交 200 股，编号 1 的卖单还剩 300 股。程序会检查成交引用、限价和剩余量，遇到无效记录时报告 CSV 行号。

## 因子计算

OrderFillRateSpread 是一个高频因子，用最近 100 笔有效委托中卖方与买方的成交比例之差，衡量两侧挂单成交的快慢。原报告的假设是，当卖单很快成交时，卖方可能尝试提高报价。当卖单成交较慢时，急于卖出的参与者可能压低报价。买方的逻辑对称。

每只证券取最近 100 笔委托（买卖合计）。每笔成交按双方委托号累加成交量。已滑出窗口的委托仍可被成交引用，但只有当前窗口中的委托参与求和。

```text
卖方成交比例 = 窗口内卖单的累计成交量之和 / 窗口内卖单的委托量之和
买方成交比例 = 窗口内买单的累计成交量之和 / 窗口内买单的委托量之和
OrderFillRateSpread = 卖方成交比例 - 买方成交比例
```

截图第 104 行的成交笔数条件被截断。本版沿用根据缓冲容量作出的推定：当该证券已有 100 笔委托且累计成交达到 500 笔时开始输出，此后每笔成交更新一次。500 笔仅用作启动条件。买卖方向按输入的 `B`、`S` 识别，修正了截图中对单个字符调用 `strcmp` 的问题。若窗口缺少某一侧委托，该侧比例和因子输出为 `NA`。

随附的 `factor_output.csv` 首条记录如下。该时刻卖方成交比例约为 67.71%，买方约为 77.66%，因子为 −0.09951241。

```csv
time_ms,symbol,trade_count,ask_fill_rate,bid_fill_rate,alpha
34304125,000002.SZ,500,0.67708333,0.77659574,-0.09951241
```

## 原报告测试结果（阳雄方法）

测试区间为 2018-01-02 至 2018-06-11，报告记载的测试天数为 106 天。以下数值转录自截图。原报告注明，当时使用的数据仅深市提供逐笔委托，因此该次测试中沪市证券没有因子值。

| 周期 | IC | ICIR |
| --- | ---: | ---: |
| 1 分钟 | 0.0219 | 1.1315 |
| 5 分钟 | 0.0334 | 0.8044 |
| 15 分钟 | 0.0406 | 0.8904 |
| 30 分钟 | 0.0448 | 0.9224 |
