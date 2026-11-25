# 截面动量排名（MomentumRank）

这个因子衡量一只证券的收盘涨跌幅在当日可比证券中的排名。计算只使用当日收盘价和前收盘价。输出保留输入行序，便于按日期和证券与其他因子合并。示例中的价格和证券记录均为模拟数据。

## 输入与计算

输入为带表头的 CSV，每行是一只证券在一个交易日的快照。同一证券每天一行，日期按升序排列，日内证券顺序任意。价格为整数分，取值为 0 至 999999999。日期用八位数字表示。

```csv
date,symbol,prev_close_cent,close_cent
20241014,600001.SH,1000,1010
20241014,000002.SZ,2000,1980
20241014,600003.SH,3000,3030
```

1. 前收盘价或收盘价为 0 时，该行输出 `NA,NA`，并从当日排名样本中排除。
2. 每只有效证券先计算 `return_bp = (收盘价 − 前收盘价) × 10000 / 前收盘价`。结果向零取整。同一交易日的证券按这个整数基点收益率排名。
3. 设当日有 `N` 只有效证券，其中 `L` 只收益率更低，`E` 只收益率相同（含自身）。输出 `rank_ppm = (2L + E − N) × 1000000 / (N − 1)`，向零取整。并列证券共享其所占名次的平均分，单只有效证券得 0 分。

上例中三只证券的收益率依次为 100、−100、100 bp，两只并列第一的证券各得 500000 分，另一只得 −1000000 分。CPU 参考实现按日排序并合并并列区间。[C++ 入口](MomentumRank.cpp) · [CPU 排名实现](rank_cpu.hpp) · [完整手算样例](data/sessions.expected.csv)

## 运行

从仓库根目录执行以下命令。第一组只需要支持 C++17 的编译器。

```sh
mkdir -p runs
c++ -O3 -std=c++17 example/momentum_rank/MomentumRank.cpp -o runs/momentum-rank
runs/momentum-rank example/momentum_rank/data/sessions.csv > runs/momentum-rank.csv
diff -u example/momentum_rank/data/sessions.expected.csv runs/momentum-rank.csv
```

在配好模型 API 的计算节点上，下面的入口让 Pi 根据 C++ 和本地头文件编写 CUDA，随后编译、验证并测速。任务清单只将输入 CSV 提供给 agent。控制器单独编译 C++ 参考程序生成期望输出。[任务清单](task.json)

```sh
bash workflow/cuda.sh --agent --backend pi --spec example/momentum_rank/task.json
```

固定用例覆盖仅表头的输入、五个交易日的手算样例，以及日内证券数分别为 31、32、33、257 和 513 的五个截面。另附的独立验证脚本用有理数计算收益率和平均名次，可换种子生成新行情。将下面的 `run` 设为上一步打印的运行目录。在相同 GPU 分配下，复用控制器已编译的 CPU 和 GPU 程序。

```sh
run=runs/cuda-agent-实际目录名
python3 example/momentum_rank/verify_holdout.py --cpu "$run/oracle/runs/cpu" \
  --gpu "$run/candidate/runs/replay" --seed 91028
```

## CUDA 实现

Pi 生成的两个后端共用日期分组和整数收益率内核。SIMT 为每行分配一个线程，遍历当日可比证券并统计排名。Tile 为每个交易日分配一个 256 线程块，有效证券数至多 4096 时，在共享内存里用双调排序和二分查找求并列名次。超过这个容量时，线程块分片读取同一交易日的收益率并累计名次。[CUDA 源码](generated/replay.cu)

在计算节点上也可以直接编译仓库里的 CUDA 源码，复用本机 Toolkit。在已分配 GPU 的环境中，从仓库根目录执行。

```sh
(cd example/momentum_rank && NVCC_CCBIN=/usr/bin/g++-7 bash ../../workflow/build_port.sh \
    /usr/local/cuda-11.3 86 17 17 generated ../../runs/momentum-rank-cuda)
runs/momentum-rank-cuda/replay tile example/momentum_rank/data/sessions.csv
```

## 测试环境与实测结果

### 测试环境与 Agent 模型

| 项目 | 本次实测配置 |
| --- | --- |
| Agent 后端与模型 | Pi 1.6.0，通过配置的 API 调用 `qwen3.8-max`。首次会话从 C++ 生成 CUDA，第二次会话根据诊断修复候选。 |
| CPU 基线 | Intel Xeon Gold 6248R @ 3.00 GHz。纯计算基线使用单线程 C++，固定在逻辑核 6。 |
| GPU | NVIDIA GeForce RTX 3090，计算能力 8.6（sm_86），82 个 SM。 |
| 编译工具链 | CUDA Toolkit 11.3（nvcc 11.3.58），nvcc 主机编译器 G++ 7.5.0。CPU 参考实现使用 GCC 12.3.0，CPU 与 CUDA 均以 C++17 和 `-O3` 编译。 |
| 计时工具 | CPU 使用 `std::chrono::steady_clock`，GPU 使用 Nsight Systems 2022.1.3.3 记录设备时间戳。 |

冷启动测试与纯计算测试均运行 Pi 修复后的 CUDA 实现。迁移报告与纯计算报告中的 CUDA 源码 SHA-256 一致。[Agent 运行记录](validation/pi-rtx3090.json) · [硬件、工具版本与源码哈希](validation/compute-rtx3090.json)

### 正确性与冷启动端到端耗时

首次草稿在多交易日样例上产生排名偏差，额外的竞争检查也发现了 Tile 共享计数器的读写冲突。保留草稿并将诊断交给第二次 Pi 会话后，模型修正了交易日索引和同步。最终两个后端均被 CUPTI 记录到实际内核启动，通过了逐字节比对和 Compute Sanitizer 的显存检查。

独立验证使用种子 91028，覆盖 7 个交易日、6417 行。最大截面有 5003 行，其中 4346 行有效，触发了 Tile 的分片路径。CPU、SIMT、Tile 的输出均与独立有理数实现一致。在固定的 866 行边界样例上，Tile 的 `racecheck` 和 `synccheck` 都报告零错误。

| 固定用例 | 行数 | CPU | SIMT | Tile |
| --- | ---: | ---: | ---: | ---: |
| 仅表头 | 0 | 2.541 ms | 3.140 ms | 3.235 ms |
| 多日手算样例 | 17 | 2.626 ms | 122.541 ms | 139.394 ms |
| 多日边界样例 | 866 | 4.410 ms | 149.911 ms | 146.687 ms |

表中为每种模式三次独立进程运行的中位数，包含程序启动、CSV 读写以及非空输入的 CUDA 初始化和传输。在这两个非空规模上，CPU 端到端耗时更短。留出测试只核对正确性。[实测记录、源码哈希和原始计时](validation/pi-rtx3090.json) · [此次 Pi 迁移使用的原始单文件因子](https://github.com/delta-1234/cpp2cuda/blob/8bfb5547d79851c49262dbe285f59f9a216374d9/example/momentum_rank/MomentumRank.cpp)

## 单线程 CPU 与 GPU 的纯计算耗时

CPU 使用命令行程序共用的 `compute_ranks`，从内存里的行情开始，计入输出初始化、日期分组、收益率计算、排序和并列名次计算。CPU 进程固定在一个逻辑核，预热至少 50 ms。小输入会自动成批计时，再按批内计算次数折算单次耗时。[CPU 计算实现](rank_cpu.hpp)

GPU 直接调用已验证 CUDA 源码中的 `launch_all`。Nsight Systems 用设备时间戳逐次加总日期分组、CUB 前缀扫描、收益率和排名的全部内核。CSV 读写、数据打包、CUDA 初始化、显存分配、主机与设备传输、主机发射内核及内核间空隙均排除在 GPU 计时外。SIMT、Tile 各预热 5 次，交替执行，每轮均逐元素比对 CPU 的收益率和排名。加速比为 CPU 单次计算中位数除以 GPU 全部计算内核总耗时的中位数。

在计算节点的仓库根目录执行下列命令复现。默认测试固定样例、留出集及不同交易日数和证券数的模拟行情。用 `--input /path/to/quotes.csv` 可测自己的非空输入。

```sh
# 使用当前环境分配的可见 GPU。
export CUDA_HOME=/usr/local/cuda-11.3 NVCC_CCBIN=/usr/bin/g++-7
export NSYS=/usr/local/cuda-11.7/nsight-systems-2022.1.3/bin/nsys
python3 example/momentum_rank/benchmark_compute.py --core 6 --warmup 5 --repeat 11
```

14 组输入的 SIMT 和 Tile 输出均与 CPU 逐元素一致。每组取 11 次测量的中位数。表中耗时统一为微秒，加速比大于 1 表示 GPU 计算更快。

| 输入（交易日 × 每日证券数） | 行数 | CPU（µs） | SIMT（µs） | Tile（µs） | SIMT 加速比 | Tile 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 手算样例（5 日） | 17 | 0.46 | 17.60 | 19.17 | 0.03× | 0.02× |
| 边界样例（5 日） | 866 | 41.85 | 31.94 | 31.52 | 1.31× | 1.33× |
| 留出集（7 日） | 6,417 | 457.30 | 182.66 | 3,574.35 | 2.50× | 0.13× |
| 1 × 256 | 256 | 7.04 | 23.65 | 23.01 | 0.30× | 0.31× |
| 2 × 256 | 512 | 20.24 | 23.20 | 22.75 | 0.87× | 0.89× |
| 4 × 256 | 1,024 | 52.18 | 24.00 | 23.84 | 2.17× | 2.19× |
| 8 × 256 | 2,048 | 122.05 | 24.64 | 24.35 | 4.95× | 5.01× |
| 16 × 256 | 4,096 | 366.99 | 25.12 | 24.54 | 14.61× | 14.95× |
| 32 × 256 | 8,192 | 527.15 | 25.12 | 23.42 | 20.99× | 22.50× |
| 64 × 256 | 16,384 | 1,054.03 | 26.40 | 26.02 | 39.93× | 40.51× |
| 256 × 256 | 65,536 | 4,263.45 | 50.24 | 37.76 | 84.86× | 112.91× |
| 1024 × 256 | 262,144 | 17,665.62 | 120.16 | 74.91 | 147.02× | 235.82× |
| 64 × 1024 | 65,536 | 4,822.74 | 131.97 | 48.86 | 36.55× | 98.70× |
| 16 × 4096 | 65,536 | 5,319.86 | 484.19 | 167.36 | 10.99× | 31.79× |

每天 256 只证券时，512 行的两条 GPU 路径均慢于 CPU，1,024 行均快于 CPU，交叉点位于这两个采样规模之间。每日截面宽度也影响结果。总量同为 65,536 行，从 256 日 × 256 只改为 16 日 × 4,096 只时，Tile 加速比从 112.91× 降至 31.79×。留出集中有一天的有效证券数为 4,346，超过共享内存排序的 4,096 项容量，Tile 转入分片逐项比较，该组 Tile 的耗时高于 CPU。[11 次原始计时与环境记录](validation/compute-rtx3090.json)
