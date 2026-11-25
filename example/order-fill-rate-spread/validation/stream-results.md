# 流式回放验证

以原 C++ 程序生成的 CSV 为对照，逐字节检查增量 CPU、SIMT、CUDA C++ 分块 Tile 和 Triton Tile 的输出。用例覆盖空输入、单侧委托、64 位求和、引用已滑出窗口的委托、8,191／8,192／8,193 行批次边界、146 只证券交错和同一进程连续回放不同文件。三种 GPU 内核均在 8,193 行的输入上通过了 Compute Sanitizer 显存检查。[各用例的输出行数和 SHA-256](stream-checks.json) · [显存检查日志](stream-memcheck.txt)

两组模拟行情按种子和提交委托数保存输入及输出哈希，可用于复核相同规模的回放。[种子 8001](stream-scale-rtx3090.json) · [种子 42](stream-scale-seed42.json) · [种子 8001 的小规模用例](stream-warm-seed8001.json) · [种子 42 的小规模用例](stream-warm-seed42.json)

流式路径的现有回放记录用于验证正确性。订单因子的 CPU/GPU 计算对照采用已打包窗口上的归约，包含两侧成交率和因子差值。[计算耗时与加速比](../README.md#原始窗口打包路径的实测结果)
