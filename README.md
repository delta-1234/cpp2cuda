# OrderFillRateSpread（买卖委托成交率差）

独立 C++17 程序，读取单个交易日的逐笔委托与成交 CSV，按证券输出卖方成交比例、买方成交比例和两者之差。程序使用 C++ 标准库，内置模拟数据生成器。

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

### 实测结果

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
