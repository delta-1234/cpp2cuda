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

| 专项 | 接手时内核耗时 | 候选内核耗时 | 选版结果 |
| --- | ---: | ---: | --- |
| SIMT | 5.280 µs | 2.784 µs | 采用。将依下标生成输入和窗口求和合为一次内核启动。 |
| Tile | 5.728 µs | 3.072 µs | 保留原 Tile。候选在 259 和 8,193 个输出上的 Tile 端到端中位数分别增加 8.2% 和 8.6%，超过 5% 门槛。 |

内核时间为 8,193 个输出上三次采样的中位数。恢复选中源码并重新编译后，该用例的冷启动端到端中位数为 CPU 3.589 ms、SIMT 220.432 ms、Tile 216.187 ms。这个规模下 CPU 更快。

纠错回归另有可复用的故障候选，以及定向重测、全量验证和检查点回退的记录。[故障候选](repair_fixture/window_energy.cu) · [纠错报告](validation/agent-repair-rtx3090.json) · [通用迁移报告](../../validation/port-agent-rtx3090.json)

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
