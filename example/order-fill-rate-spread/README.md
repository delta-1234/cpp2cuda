# OrderFillRateSpread 示例

这个高频订单因子从逐笔委托和成交计算买卖双方的成交率差。以下交易字段、预热规则和规模基准均属于本示例。其他量化因子的输入、计算语义和输出由各自的 C++ 参考程序定义。文中的命令均在仓库根目录运行。

```sh
bash workflow/cuda.sh --generate-cuda example/order-fill-rate-spread/OrderFillRateSpread.cpp
```

原有简写 `bash workflow/cuda.sh --generate-cuda OrderFillRateSpread.cpp` 也可直接运行这个示例。该入口会生成 SIMT 与 Tile 候选，验证正确性并分别优化内核。可用 `--no-optimize` 只运行转写和验收。`--extra-orders 524288 2097152` 会额外生成这两种规模的模拟订单。显式传入 `--per-file` 可复现早期逐文件生成器。参考程序、因子定义和固定输出分别保存在 [OrderFillRateSpread.cpp](OrderFillRateSpread.cpp)、[factor_replay.hpp](factor_replay.hpp) 和 [factor_output.csv](data/factor_output.csv)。

## 两个专项 agent 的实测

在 RTX 3090（82 个 SM，计算能力 8.6）上，以仓库里的模型移植为起点，通过配置的模型 API 调用 `qwen3.8-max`，后端使用 Hugging Face Transformers 4.41.2 的 `ReactJsonAgent`。CUDA 编译器为 nvcc 11.3.58。下表使用 131,072 笔委托作为内核测量用例，每个角色各试一个独立候选。Nsight 对每次回放实际启动的全部 GPU 内核求和，再取三次采样的中位数。专项优化倍数为同一 GPU 路径的起点耗时除以候选耗时。最终选版还需满足工作流的其他验收条件。

| 专项 | 起点内核（µs） | 候选内核（µs） | 相对起点的计算加速比 | 最终选版 |
| --- | ---: | ---: | ---: | --- |
| SIMT | 404.355 | 241.250 | 1.676× | 保留起点 |
| Tile | 280.706 | 281.816 | 0.996× | 保留起点 |

两条路径均通过七组 CPU 逐字节对照和 Compute Sanitizer 显存检查。两个候选补丁都可分别应用在仓库随附的起点上，复原模型当时编译的源码。[SIMT 实验补丁](simt-trial.patch) · [Tile 实验补丁](tile-trial.patch)

专项报告列出每个候选的硬件假设、内核采样值、机器码指纹和选版结果。[专项优化记录](validation/factorport-specialists-rtx3090.json)

## 运行已验证的示例移植

先前的逐文件实验使用 `qwen3.8-max` 分三次生成了 [主机程序](generated/host.cpp)、[GPU 接口](generated/gpu.hpp) 与 [CUDA 内核](generated/kernels.cu)。三个文件均可在计算节点的原始回复中按 SHA-256 核对。SIMT 每个线程计算一行，Tile 使用 128 个线程在共享内存中归约订单，每块最多处理 8 行。主机复用原头文件的逐笔回放，按输出时点在内存中打包窗口，每 65,536 行为一批。同一进程中的批次复用显存。

```sh
bash workflow/cuda.sh --build-generated
bash workflow/cuda.sh --generated simt example/order-fill-rate-spread/data/sample_ticks.csv runs/model-simt.csv
bash workflow/cuda.sh --generated tile example/order-fill-rate-spread/data/sample_ticks.csv runs/model-tile.csv
cmp example/order-fill-rate-spread/data/factor_output.csv runs/model-simt.csv
cmp example/order-fill-rate-spread/data/factor_output.csv runs/model-tile.csv
```

输入和输出使用不同文件，输出路径会被覆盖。单次回放包含 CUDA 初始化。复现实验时，可以在 `--generate-cuda example/order-fill-rate-spread/OrderFillRateSpread.cpp` 后加 `--extra-orders 524288 2097152`，把两个更大规模加入新生成候选的基准测试。

在 RTX 3090 上，两种生成的实现均与原 C++ 程序的输出逐字节一致，Compute Sanitizer 的边界样本检查均为 0 错误。[源码哈希、内核启动和验证记录](validation/generated-cuda-rtx3090.json)

## 优先使用流式回放

在计算节点上使用下列入口，可直接从原始逐笔 CSV 写到最终因子 CSV。SIMT、CUDA C++ 分块 Tile 和提前编译的 Triton Tile 共用一个 C++ 前端，执行阶段使用同一套 CSV 解析、委托检查和输出格式化。

```sh
bash workflow/cuda.sh --build-stream
bash workflow/cuda.sh --stream simt example/order-fill-rate-spread/data/sample_ticks.csv runs/simt.csv
bash workflow/cuda.sh --stream tile example/order-fill-rate-spread/data/sample_ticks.csv runs/tile.csv
bash workflow/cuda.sh --stream triton example/order-fill-rate-spread/data/sample_ticks.csv runs/triton.csv
bash workflow/cuda.sh --stream cpu example/order-fill-rate-spread/data/sample_ticks.csv runs/incremental-cpu.csv
```

`--serve` 允许多次回放共用 GPU 上下文、CUDA stream、页锁定主机内存和显存。输入协议为每行一个「输入路径、制表符、输出路径」。进程收到 EOF 后退出，每个 CSV 的委托状态独立。返回 `READY` 表示工作进程已启动，`OK` 后依次为输出行数、批次数、分段数及本进程一次性初始化耗时（毫秒）。

```sh
printf '%s\t%s\n' example/order-fill-rate-spread/data/sample_ticks.csv runs/first.csv example/order-fill-rate-spread/data/sample_ticks.csv runs/second.csv |
  bash workflow/cuda.sh --stream simt --serve
```

每个 GPU 批次最多 8,192 个时点，同一证券每段最多 128 行。段首以当前委托状态为基准，段内记录四个 int64 变化量，GPU 计算前缀和及成交率。只有分段索引按证券分组，因子按逐笔到达顺序输出。CPU 对照 `cpu` 直接增量维护四个统计量，`legacy` 保留原来的 100 笔窗口扫描。旧打包后端仍可用于逐阶段诊断。

流式回放可用空输入、同一进程多次回放、多证券交错以及跨批次用例复核输出。下节提供已打包窗口上 CPU 与 GPU 的纯计算对照。[流式回放验证记录](validation/stream-results.md)

## 验证示例的窗口打包实现

把本仓库作为 agent 的工作目录，给它指令「按 AGENTS.md 在本机离线验证 OrderFillRateSpread 的 SIMT 与 Tile 实现」。阶段、排错分支和验收条件见 [AGENTS.md](../../AGENTS.md)。[离线资料目录](../../offline_docs/README.md) 包含计算节点上的 Toolkit 资料位置和共享数据格式。

```sh
python3 workflow/run.py --probe-only
python3 workflow/run.py --input example/order-fill-rate-spread/data/sample_ticks.csv --repeat 7
```

运行器需要 Python 3.8+ 和 C++17 主机编译器。两条原生后端通过本地 nvcc 编译，使用 CUDA C++14。每次运行打印独立的 `runs/run-*/report.json` 路径，并在该目录保留 CPU 和 GPU 结果。未连 GPU 的机器可完成 CPU 及打包验证。运行器不下载依赖。严格物理隔离时，可在断网计算节点或已预装工具链、禁用网络的本地容器内运行。

### 在计算节点复现

从仓库根目录运行，按本地安装情况设置 `WORKFLOW_PYTHON`、`CUDA_HOME` 和 `NVCC_CCBIN`。`--device 0` 选择调度器分配的第一块可见设备。

```sh
bash workflow/cuda.sh --device 0 --repeat 7
```

### 原始窗口打包路径的实测结果

RTX 3090（sm_86，24 GiB），驱动 550.76，CUDA Toolkit 11.3，Triton 3.2.0。在相同的预打包窗口上，单线程 C++ 和 GPU 都执行按买卖方向归约、计算两侧成交率和求差。CPU 计时覆盖内存中的计算循环，GPU 通过 CUDA event 计量预热后的归约内核。两者均排除逐笔回放、打包、文件读写和结果格式化，GPU 另排除初始化、分配和主机与设备间传输。[CPU 计时实现](packed_cpu.cpp) · [环境记录](../../validation/rtx3090-probe.json)

每个窗口含 100 笔委托。表中为重复测量的中位数，加速比取同规模单线程 C++ 中位数除以 GPU 中位数。

| 窗口数 | 测量次数 | 单线程 C++（µs） | SIMT（µs） | SIMT 加速比 | 共享内存 Tile（µs） | Tile 加速比 | Triton Tile（µs） | Triton 加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 912 | 7 | 879.127 | 9.216 | 95.39× | 9.216 | 95.39× | 155.648 | 5.65× |
| 5,472 | 3 | 2,649.120 | 33.152 | 79.91× | 24.128 | 109.79× | — | — |

912 个窗口的三种 GPU 实现都通过样例和边界输入的逐行比对。5,472 个窗口的测量覆盖两个原生 CUDA 后端。[912 个窗口的逐次计时](validation/rtx3090.json) · [5,472 个窗口的逐次计时](validation/rtx3090-multibatch.json)

另一次独立采样中，94,772 个窗口的 SIMT 内核中位数为 463.552 µs（5 次）。该组仅记录了 SIMT 内核。[逐次设备计时](validation/packed-simt-kernel-rtx3090.json)

两个原生 CUDA 后端也通过了边界输入的显存检查。[SIMT memcheck](validation/simt-memcheck.txt) · [Tile memcheck](validation/shared_tile-memcheck.txt)

仓库另含 NVIDIA cuTile DSL 实现 `cuda/cutile_backend.py`。服务器的验证环境缺少 `cuda.tile`，该路径尚未实机验证。在配有相应本地依赖的节点上，添加 `--require-cutile` 可将其列为必验项。

## 运行样例

在仓库根目录编译并回放固定行情，再与随附的预期结果逐字节比较。

```sh
mkdir -p runs
factor=example/order-fill-rate-spread
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "$factor/OrderFillRateSpread.cpp" -o runs/order_fill_rate_spread
runs/order_fill_rate_spread "$factor/data/sample_ticks.csv" > runs/order-output.csv
cmp "$factor/data/factor_output.csv" runs/order-output.csv
head -n 4 runs/order-output.csv
```

省略输入参数时，程序读取当前工作目录的 `sample_ticks.csv`。在 macOS 上可给编译命令加上 `-isysroot "$(xcrun --sdk macosx --show-sdk-path)"` 以显式选择 SDK。

## 模拟行情

`sample_ticks.csv` 使用种子 `8001` 生成。两个证券代码是示例标识，价格、数量和时间均为模拟值。生成器从 09:30 起提交限价单，价格以分计，委托量为 100 至 1,200 股的整百股。能成交的委托先匹配对手方最优报价，同价按到达顺序撮合，以在簿委托的价格成交。一次委托可以产生多笔成交，未成交部分继续排队。样例仅包含连续竞价的委托和成交。

| 示例证券 | 委托笔数 | 成交笔数 | 因子输出行数 |
| --- | ---: | ---: | ---: |
| 000001.SZ | 1,268 | 928 | 429 |
| 000002.SZ | 1,332 | 982 | 483 |
| 合计 | 2,600 | 1,910 | 912 |

以下命令生成另一组行情并计算因子。同一构建环境下，相同种子可复现相同数据。

```sh
runs/order_fill_rate_spread --generate 42 > runs/ticks_42.csv
runs/order_fill_rate_spread runs/ticks_42.csv > runs/factors_42.csv
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

## 维护示例的既有实现

使用 `bash workflow/cuda.sh --agent '维护任务'` 时，控制器先盘点 GPU 和 Toolkit，将任务、设备信息及项目规则传给所配置的模型。模型可以按需读取仓库源码和 `offline_docs/` 的文本材料、编辑候选副本中已有的 `cuda/` 或 `stream/` 源文件、读取编译错误并继续修复。编译、验证和基准运行在计算节点的独立 Linux 网络、挂载和 PID 命名空间内。候选源码对这些子进程只读，工作产物写入该次运行的 `runs/`。模型 API 调用由外层控制器发起，模型接口应支持文本生成。

当前候选版本通过 CPU 对照和基准回放后，控制器才将补丁应用到工作目录。若原文件在运行期间被其他任务修改，补丁保存在运行目录供核对。需要仅生成候选补丁时使用以下命令。

```bash
bash workflow/cuda.sh --agent --proposal-only '检查 SIMT 实现，提出并验证一个针对当前输入的优化。'
```

每次运行会打印 `runs/api-agent-*/report.json` 路径，同目录包含 `changes.patch`、分阶段日志及候选副本。默认最多 24 次模型决策，可用 `--max-steps` 调整。默认验证流式 CPU、SIMT、CUDA C++ Tile 和 AOT Triton；编辑原始打包实现时还会运行相应路径的验证。
