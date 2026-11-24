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

上例中三只证券的收益率依次为 100、−100、100 bp，两只并列第一的证券各得 500000 分，另一只得 −1000000 分。CPU 参考实现按日排序并合并并列区间。[C++ 源码](MomentumRank.cpp) · [完整手算样例](data/sessions.expected.csv)

## 运行

从仓库根目录执行以下命令。第一组只需要支持 C++17 的编译器。

```sh
mkdir -p runs
c++ -O3 -std=c++17 example/momentum_rank/MomentumRank.cpp -o runs/momentum-rank
runs/momentum-rank example/momentum_rank/data/sessions.csv > runs/momentum-rank.csv
diff -u example/momentum_rank/data/sessions.expected.csv runs/momentum-rank.csv
```

在配好模型 API 的 计算节点 上，下面的入口让 Pi 根据 C++ 和本地头文件编写 CUDA，随后编译、验证并测速。任务清单只将输入 CSV 提供给 agent。控制器单独编译 C++ 参考程序生成期望输出。[任务清单](task.json)

```sh
bash workflow/cuda.sh --agent --backend pi --spec example/momentum_rank/task.json
```

固定用例覆盖仅表头的输入、五个交易日的手算样例，以及日内证券数分别为 31、32、33、257 和 513 的五个截面。另附的独立验证脚本用有理数计算收益率和平均名次，可换种子生成新行情。将下面的 `run` 设为上一步打印的运行目录。命令从报告读取同一 GPU 的可见性设置，并复用控制器已编译的 CPU 和 GPU 程序。

```sh
run=runs/cuda-agent-实际目录名
export CUDA_VISIBLE_DEVICES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["observations"]["toolchain"]["cuda_visible_devices"])' "$run/report.json")
python3 example/momentum_rank/verify_holdout.py --cpu "$run/oracle/runs/cpu" \
  --gpu "$run/candidate/runs/replay" --seed 91028
```

## CUDA 实现

Pi 生成的两个后端共用日期分组和整数收益率内核。SIMT 为每行分配一个线程，遍历当日可比证券并统计排名。Tile 为每个交易日分配一个 256 线程块，有效证券数至多 4096 时，在共享内存里用双调排序和二分查找求并列名次。超过这个容量时，线程块分片读取同一交易日的收益率并累计名次。[CUDA 源码](generated/replay.cu)

在 计算节点 上也可以直接编译仓库里的 CUDA 源码，复用本机 Toolkit。先按前文设置可见 GPU，然后从仓库根目录执行。

```sh
(cd example/momentum_rank && NVCC_CCBIN=/usr/bin/g++-7 bash ../../workflow/build_port.sh \
    /usr/local/cuda-11.3 86 17 17 generated ../../runs/momentum-rank-cuda)
runs/momentum-rank-cuda/replay tile example/momentum_rank/data/sessions.csv
```

## 测试环境与实测结果

Pi 1.6.0 使用配置的 `qwen3.8-max`，在 RTX 3090（sm_86）、CUDA Toolkit 11.3 上从 C++ 生成 CUDA。首次草稿在多交易日样例上产生排名偏差，额外的竞争检查也发现了 Tile 共享计数器的读写冲突。保留草稿并将诊断交给第二次 Pi 会话后，模型修正了交易日索引和同步。最终两个后端均被 CUPTI 记录到实际内核启动，通过了逐字节比对和 Compute Sanitizer 的显存检查。

独立验证使用种子 91028，覆盖 7 个交易日、6417 行。最大截面有 5003 行，其中 4346 行有效，触发了 Tile 的分片路径。CPU、SIMT、Tile 的输出均与独立有理数实现一致。在固定的 866 行边界样例上，Tile 的 `racecheck` 和 `synccheck` 都报告零错误。

| 固定用例 | 行数 | CPU | SIMT | Tile |
| --- | ---: | ---: | ---: | ---: |
| 仅表头 | 0 | 2.541 ms | 3.140 ms | 3.235 ms |
| 多日手算样例 | 17 | 2.626 ms | 122.541 ms | 139.394 ms |
| 多日边界样例 | 866 | 4.410 ms | 149.911 ms | 146.687 ms |

表中为每种模式三次独立进程运行的中位数，包含程序启动、CSV 读写以及非空输入的 CUDA 初始化和传输。在这两个非空规模上，CPU 端到端耗时更短。留出测试只核对正确性。[实测记录、源码哈希和原始计时](validation/pi-rtx3090.json)
