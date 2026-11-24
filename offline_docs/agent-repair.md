# CUDA agent 的纠错机制

Aider 会把编辑、lint 或测试错误作为下一轮模型输入，并对连续修复轮数设置上限。OpenHands SDK 的 stuck detector 会对照动作和观测，识别重复错误和交替出现的重复模式。此仓库参照这些设计，把 CUDA 编译器、独立 CPU 对照、CUPTI 和显存检查作为修复依据。模型运行期间使用本地实现。[Aider 修复循环](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/coders/base_coder.py#L933-L945) · [Aider 测试反馈](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/coders/base_coder.py#L1598-L1624) · [OpenHands 重复模式检测](https://github.com/OpenHands/software-agent-sdk/blob/53ecc14a8a039faed26fc045b813321bad8e594f/openhands-sdk/openhands/sdk/conversation/stuck_detector.py)

1. 工具失败时，控制器返回错误阶段、实际日志、相关用例和后端，记录当时的源码哈希。通用程序的输出差异包含第一个不同字节的位置和两侧短片段。
2. 模型可以通过 `diagnostics` 查看历史错误，通过 `log`、`read`、`experiment` 检查假设，随后调用 `edit` 或 `write` 修复代码。未解决的同一错误重复出现或两类失败交替出现时，下一轮会收到改查证据、改用其他方案或恢复版本的提示。
3. `test_case` 可重跑一个指定后端和用例。已知失败用例会在下一次全量验收前优先重测。完整验收仍覆盖两个后端、CPU 对照、实际内核启动以及显存检查。
4. 每次完整验收成功后按源文件内容生成检查点。模型可随时 `restore`；旧草稿另外保留。恢复会清除当前构建及验证状态，必须重新编译、验收和计时。
5. 耗尽工具轮数时，如果已有通过验证的版本而当前版本尚未完成，控制器会恢复最近的检查点，在当前测试集上重新编译、全量验证、计时。成功报告为 `verified_fallback`。没有通过过完整验收或恢复后验收失败时，报告失败。性能提升以实测时间为准。

运行目录的 `report.json` 保存失败记录、候选哈希、检查点和计时；`tools.jsonl` 保存工具观测。数据依赖特定任务、编译器和服务器，重用另一次运行的源码时可从 `--candidate` 开始重新验证。

在 计算节点 上可用 `bash workflow/cuda.sh --test-repair` 重跑确定性的控制器回归。它用预设动作复现同一错误两次，修好后再引入回归并耗尽轮数，真实执行 CUDA 构建、定向重测、完整验证和回退。另一次真实模型自主修复的操作序列及一行改动见[实测报告](../example/window_energy/validation/agent-repair-rtx3090.json)。
