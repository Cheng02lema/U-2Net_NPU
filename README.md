

# U-2Net\_NPU

**U-2Net\_NPU** 是基于流行的 **U-2Net** 模型的硬件加速版本，针对华为的 **昇腾 NPU**（神经网络处理单元）进行了优化。本项目提供了两个迁移到 NPU 的示例脚本，分别用于训练和推理。

## 环境要求

在运行本仓库中的脚本之前，您需要安装以下环境：

* **华为昇腾 NPU 硬件** 或兼容的 NPU 设备。
* **昇腾 SDK**（包括 `CANN` 和 `MindSpore`）以配置 NPU 环境。(跳过)
* **Python**（兼容的版本，例如 Python 3.7+）。
* **PyTorch**（或根据您的环境选择的相关框架）。
* **U-2Net** 源代码，您可以从官方 U-2Net 仓库获取，地址：[U-2Net GitHub 仓库](https://github.com/xuebinqin/U-2-Net)。

## 项目内容

本仓库包含以下文件：

* **NPU\_Training.py**：用于在昇腾 NPU 上训练 U-2Net 模型的示例脚本。
* **NPU\_TEST\_human\_seg\_test.py**：用于在人类分割任务中运行 U-2Net 推理的示例脚本，已针对 NPU 进行了优化。
* **U-2Net**：原始的 U-2Net 源代码，包含模型架构和训练管道，您可以从 [U-2Net GitHub 仓库](https://github.com/xuebinqin/U-2-Net) 获取。

## 安装步骤

1. **安装昇腾 SDK**（跳过）
   请按照华为官方文档的指导，安装 **昇腾 CANN SDK** 和 **MindSpore** 框架，以确保与您的系统和 NPU 硬件兼容。
   安装指南：[昇腾 CANN SDK 安装](https://support.huawei.com/enterprise/en/doc/EDOC1100056929)

2. **克隆 U-2Net 仓库**
    GitHub 仓库下载 U-2Net_npu 源代码：

   ```
 https://github.com/Cheng02lema/U-2Net_NPU.git
   ```

   确保将 U-2Net 源代码放置在本项目的正确目录中，以便使用。

3. **准备您的环境**(在服务器中不需要安装这些包)
   确保 NPU 驱动程序和相关库已正确安装，并且在您的 Python 环境中已配置好相关依赖：

   ```
   pip install -r requirements.txt
   ```

4. **在 NPU 上训练模型**
   运行 `NPU_Training.py` 脚本，在昇腾 NPU 上训练 U-2Net 模型：

   ```
   python NPU_Training.py
   ```

   该脚本会配置训练过程，并利用 NPU 的计算能力加速模型训练。

5. **进行人类分割推理**
   运行 `NPU_TEST_human_seg_test.py` 脚本，使用预训练的 U-2Net 模型进行人类分割任务的推理。此脚本已针对 NPU 进行了优化：

   ```
   python NPU_TEST_human_seg_test.py
   ```

## 示例使用

### 1. **在 NPU 上训练模型**

要在昇腾 NPU 上训练 U-2Net 模型，只需执行训练脚本：

```bash
python NPU_Training.py
```

这将开始使用 U-2Net 架构的训练过程，NPU 会加速整个训练过程。

### 2. **进行人类分割推理**

训练完成后，您可以使用以下命令测试模型在示例图像上的分割效果：

```bash
python NPU_TEST_human_seg_test.py
```

该脚本会加载训练好的模型，并使用 NPU 进行快速推理，执行人类分割任务。

## 注意事项

* 本项目中的 U-2Net 模型是对原始 U-2Net 的优化版本，专门为 NPU 性能进行了调整。
* 在运行脚本之前，请确保您的 NPU 配置正确，且已安装好昇腾 SDK。
* 如果在 NPU 配置或安装过程中遇到问题，请参考昇腾文档进行故障排查。

## 许可证

本项目采用 **MIT 许可证**，具体内容请参见 [LICENSE](LICENSE) 文件。

