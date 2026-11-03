# CPU 数据挖掘调优实验

以 arXiv 论文数据为例，在 CPU 上比较分类、年份回归和 Louvain 社区发现。对应调优课件的三个改进角度：参数调优（随机森林叶节点与深度）、模型替换（决策树、MLP、HGBR、Extra Trees）、长尾处理与评价（类别加权、过采样、macro-F1 与逐类召回）。引用邻居特征作为 MLP 的扩展消融。

按以下顺序运行：教师基准计时 → 训练/验证/测试划分与树模型调优 → 复用划分运行 MLP 对照 → 多种子汇总与绘图。随机森林参数、MLP 轮数和 MLP 主方案在验证集上选择；其余模型作为预先声明的对照。最终测试只用于评价。

## 准备

使用 `conda env create -f environment.yml` 创建 OpenBLAS 环境，再执行 `conda activate cpu-tuning-experiment`。务必激活环境后运行，直接调用环境内的 Python 可能缺少 Windows DLL 搜索路径。课件使用 Python 3.9；本项目环境使用 Python 3.11，不把跨版本耗时视作严格复现。也可按 `requirements.txt` 安装训练依赖，绘图另需 `requirements-plot.txt`。`conda-win-64.lock` 保留旧实验的 Windows/MKL 环境，仅供重现旧结果；它与默认 OpenBLAS 环境的耗时需分开报告。

需要数据集的 `raw` 目录，包含 `node-feat.csv.gz`、`node-label.csv.gz`、`node-year.csv.gz`（也支持 `node_year.csv.gz`）和 `edge.csv.gz`。

下列 PowerShell 命令使用相对路径，分别指向数据集和最终结果目录，可按实际目录修改。每个 `--out` 必须是尚不存在的新目录。所有命令在项目目录运行：

```powershell
$data = './data/arxiv'
$work = './results'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
python -B run.py benchmark --data $data --out "$work/benchmark"
python -B run.py baseline --data $data --out "$work/baseline-42" --seed 42
python -B run.py mlp --data $data --run "$work/baseline-42" --cache "$work/features" --out "$work/mlp-42"
python -B run.py graph-restarts --data $data --run "$work/baseline-42" --out "$work/graph-42"
```

`benchmark` 复现教师的分类及回归随机森林线程对照、MLP、Logistic Regression、HGBR、全量分类随机森林及教师口径的 Louvain。参数默认 50K 抽样、150 棵树；`--skip-full`、`--skip-graph`、`--samples`、`--graph-nodes` 和 `--trees` 可用于试跑。`--jobs` 仅指定随机森林的任务并行数；MLP、HGBR 与逻辑回归的数值库线程受上述环境变量控制。逻辑回归不再传入已弃用且在当前多分类设置中无并行收益的 `n_jobs=4`。`benchmark` 的 70/30 分层划分与教师脚本一致，调优流程还要从训练部分划出验证集。

完整实验及多种子汇总见 [基线说明](BASELINE_EXPERIMENT.md) 和 [MLP 说明](MLP_EXPERIMENT.md)。每个阶段以 `python -B run.py 命令 --help` 查看参数。另支持 `python -B -m cpu_tuning`；旧脚本作为兼容入口保留。

## 代码结构与验证

`cpu_tuning/benchmark.py` 负责教师基准计时，`baseline.py` 负责树模型调参和图任务，`mlp.py`、`features.py`、`protocol.py` 负责 MLP 与图特征，汇总器和绘图器独立运行。`io.py` 管理数据读取、哈希与实时记录。数据路径由命令行传入，后续流程按文件名及 SHA-256 验证数据；复制实验目录后可重新指定数据目录。

基线运行记录包源码哈希，跨种子汇总拒绝混合源码版本及不同的硬件、系统和数值库线程环境。基线全部任务成功后才标记完成，并记录划分、图节点与原始结果的文件哈希；后续 MLP、图重启和基线汇总拒绝未完成或相关文件被改动的运行。旧基线缺少完成标记、文件哈希或源码哈希时需要重新运行；旧的基线和 MLP 结果若缺少计时环境信息，也需重新生成。不同版本的耗时不应直接合并。

| 阶段 | 输入 | 主要输出 | 对应实验要求 |
|---|---|---|---|
| `benchmark` | 原始数据 | 教师模型的计时和质量记录 | 建立 CPU 基准 |
| `baseline` | 原始数据 | 固定划分、树模型与社区结果 | 叶节点/深度调参、模型替换、类别加权 |
| `mlp` | 已完成基线、原始数据 | 五组验证曲线、预测、逐类指标 | MLP 对照、过采样、图特征扩展 |
| `graph-restarts` | 已完成基线、原始数据 | 同一图上的重启结果和总耗时 | 可选社区优化 |
| `summarize-baseline` / `summarize-mlp` | 多种子运行目录 | 均值、标准差、配对差值 | 质量与耗时比较 |
| `plot-mlp` | MLP 汇总目录 | PNG/SVG 图表 | 报告素材 |

报告可围绕开头的三个改进角度组织，图特征和图重启是可选扩展，无须把每个扩展各算成一个主要角度。

设置环境变量 `CPU_TUNING_TEST_ROOT` 为测试产物目录后，运行 `python -B -m unittest discover -s tests -v`，可执行小规模合成数据流程，包括汇总与绘图。合成结果仅验证程序，不是实验成绩。正式训练耗时受硬件、数值库和线程设置影响，应以本机实测为准。

课程最终提交为 PDF 报告及代码原文件。报告应对照基准与所选 2～3 个改进角度，同时列出质量指标、训练/推理时间及额外调参成本；命名按“学号+姓名+数据挖掘第一次实验报告/代码”。本项目负责生成可供报告引用的实测记录。
