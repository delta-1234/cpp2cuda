# 离线资料入口

将预先取得的开发指南、API 说明或示例放在本目录中。为每份材料注明标题、版本和适用的 GPU 架构。运行期间直接读取本地文件。

## 本地工具链参考

本次计算使用 Toolkit 11.3。下列资料随服务器上的 Toolkit 安装，可通过文件系统直接查看。

| 内容 | 查询方式或示例路径 |
| --- | --- |
| CUDA 版本清单 | `/usr/local/cuda-11.3/version.json` |
| 并行归约示例 | `/usr/local/cuda-11.3/samples/6_Advanced/reduction/reduction_kernel.cu` |
| Warp shuffle 示例 | `/usr/local/cuda-11.3/samples/6_Advanced/shfl_scan/shfl_scan.cu` |
| 协作组与归约接口 | `/usr/local/cuda-11.3/include/cooperative_groups/reduce.h` |
| 编译目标列表 | 运行 `/usr/local/cuda-11.3/bin/nvcc --list-gpu-arch` |
| 本地 Triton 语言接口 | `python3 -c "import triton.language.core as core; print(core.__file__)"` |
| 显存检查工具 | 运行 `/usr/local/cuda-11.3/bin/compute-sanitizer --help` |

## 两条路径共用的数据格式

`--pack` 按 CPU 回放的输出顺序写入两个文件。`keys.csv` 每行是一个时点的三个键。`windows.bin` 没有文件头，每行使用 1,024 字节的小端数据。

| 每行的元素 | 含义 |
| --- | --- |
| 第 0–127 个 int32 | 委托量，卖为正、买为负。前 100 列含委托，其余 28 列补零。 |
| 第 128–255 个 int32 | 对应委托在该时刻已累计的成交量，单位为股。 |

两条原生内核、Triton Tile 和 cuTile 使用同一约定。先按方向累加委托量和已成交量，累加器为 int64，再以 float64 计算比率。每个时点输出三个连续的小端 float64，顺序是卖方成交率、买方成交率和两者之差。内部缺失标记为 −2，写 CSV 时转换为 `NA`。两种比率均在 [0, 1] 内，差值在 [−1, 1] 内。每个 GPU 批次最多含 4,096 个窗口。

在其他节点复验 cuTile DSL 时，将与当地 `cuda-tile` 版本匹配的 API 手册、示例或离线包随任务提供。编译器、驱动、目标架构和 Python 库的兼容性通过实际内核编译与启动确认。

## 流式路径的数据布局

`stream/scan.hpp` 定义了三个连续数组。每个输出时点有 32 字节的四个 int64 值和 4 字节的行号。每个段还有一个 4 字节起始下标，每批附一个结束下标。第一个时点的四个值是窗口总量和已成交量，后续时点为相对上一输出的变化。每段仅包含同一证券的最多 128 个时点。内核在段内作前缀和，并把三个 float64 结果写回原来的行号。部分段直接使用有效长度。GPU 进程直接消费页锁定内存，这些数组不会写成中间文件。
