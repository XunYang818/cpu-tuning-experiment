# 分类改进：MLP、引用邻居特征与温和过采样

本轮依据《给同学的改进建议》预先声明五组对照，使用自己的机器重新运行。建议中的参考数值不作为本实验结果。此前随机森林、回归和社区发现的结果保留，本轮集中解决分类效果不足的问题。

## 五组对照及解释范围

| 名称 | 特征 | 过采样 | 对照目的 |
|---|---|---|---|
| raw | 原始 128 维 | 无 | 补充教师列出的 MLP 模型基线 |
| raw_ros | 原始 128 维 | 训练集少数类补至 200 | 单独观察温和采样 |
| hop1 | [X, PX]，256 维 | 无 | 一阶邻居特征 |
| hop2 | [X, PX, P²X]，384 维 | 无 | 与 hop1 比较第二阶的增益 |
| hop2_ros | [X, PX, P²X]，384 维 | 训练集少数类补至 200 | 观察特征与采样组合 |

其中引用边转成无向二值邻接矩阵，合并重复和反向重复边，自环统一为 1，`P = D⁻¹(A + I)` 按行归一化。聚合函数只接受原始特征和边，不接受标签或年份。全图 169,343 节点参与无标签特征聚合，监督任务仍使用此前抽取的 50,000 节点。

**这是传导式设置**：预测时允许使用验证、测试节点的无标签特征以及全图引用边，提升包含额外图信息的贡献。它和只用表格特征的基线信息条件不同，不能描述为严格归纳式提升，也不能代表预测未来论文。OGB 官方使用时间切分，而本课程实验沿用随机分层切分，不能与 OGB 榜单分数直接比较。[OGB 数据说明](https://ogb.stanford.edu/docs/nodeprop/)

## 训练与选择协议

- 原封不动读取三个已保存种子 42、123、2026 的索引；每次 28K 训练、7K 验证、15K 测试。不同方法在同一个种子内共享划分。
- MLP 使用单隐藏层 128、ReLU、Adam、学习率 3e-4、L2 alpha=1e-3、batch_size=200，最大 200 个 epoch，patience=20。与教师的架构和主要学习参数一致，训练轮数和早停机制明确调整如下。
- **使用独立验证集早停**：逐轮 `partial_fit`，按验证 macro-F1 最大、再按 accuracy 最大选择 epoch，相同分数保留最早轮数。所有组别完成验证后，按相同准则预先选定一个方法；随后才进行最终测试评估。学习曲线和选型记录均保存。达到最大 epoch 只表示预算耗尽，不表示收敛。
- 最终将训练和验证合为 35K，重新拟合标准化和模型，固定训练至各组已选中的 epoch。测试集不用于选轮数或选择方法。所有五组最终测试属于预先声明的消融对照，主方案取验证选择结果，不按测试最优挑选。
- 标准化在当前训练数据上拟合；有过采样时先只从当前训练 ID 中重复抽样，再拟合标准化。验证和测试保持原始分布。这样同时改变了训练分布和标准化统计量，应将它们一起视为采样处理流程。
- 不直接使用“重复样本后 `early_stopping=True`”的组合，因为 MLP 内部会再划验证集，重复样本可能同时进入训练及内部早停集。外部固定验证集避免了这个问题。[MLPClassifier 文档](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- macro-F1 固定计算所有 40 类；`rare10_macro_f1` 的 10 类按当前未采样训练集的类别数量升序确定，同数量按标签排序，验证时依据 28K，最终时依据 35K。其列表写入记录。不同种子的最少 10 类可能不同，仅在同一种子内作配对比较。
- 此前树实验已经查看过这些测试集。后续设计受到已有结果启发，因此属于公开记录的后续实验，不能宣称是全新的盲测。后续不再依据测试结果调整本轮参数。

MLP 的隐藏层配置相同，但增加输入维度也增加了第一层参数量，因此不能把全部收益归因于纯粹的无参数预处理。参考中给出的轮数、耗时也不能当作本机预期。

## 复现

先从项目 README 建好环境。需要 SciPy（当前 scikit-learn 环境中已存在，代码不自动安装）。所有命令建议设置 `PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。缓存和输出均位于挂载项目根目录的 `codex_proc`。每次输出目录必须尚不存在。

```powershell
$conda = 'D:\miniforge3\Scripts\conda.exe'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
foreach ($seed in @(42,123,2026)) {
  & $conda run --no-capture-output -n cpu-tuning-experiment python -B mlp_experiment.py --run "D:\作业\数据挖掘\codex_proc\cpu-tuning\v2-full-50000-$seed" --cache 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v3-graph-features' --out "D:\作业\数据挖掘\codex_proc\cpu-tuning\v3-mlp-$seed"
  if ($LASTEXITCODE -ne 0) { throw "实验失败：$seed" }
}
& $conda run --no-capture-output -n cpu-tuning-experiment python -B summarize_mlp.py --runs 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v3-mlp-42' 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v3-mlp-123' 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v3-mlp-2026' --out 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v3-mlp-summary'
```

汇总会从保存的预测重新计算指标，核对真实节点标签、划分、选型及逐轮曲线；同时检查每次配置和数据哈希相同。输出逐种子指标、均值及样本标准差、配对差值、逐类别结果、验证选择和学习曲线。三次重复不支持夸大统计显著性。

绘图脚本 `plot_mlp.py --summary <汇总目录> --out <新的图目录>` 使用 Matplotlib，输出 PNG/SVG。它不参与训练、调参或指标计算。本机实验环境已获准补装 Matplotlib；绘图前可设置 `MPLCONFIGDIR` 指向 `codex_proc` 下的缓存目录。

## 耗时与溯源

`protocol.json` 在运行前写入全部对照、选择规则和源代码 SHA-256；图特征缓存也校验数据、数组 SHA-256。`results.jsonl` 实时保存曲线、警告、选型和结果；最终预测保存在 `test_predictions.npz`，逐类 F1/召回率在 `per_class.csv`。缓存仅依赖特征与边，与任何划分或标签无关，可以跨种子复用。

图特征计算、缓存保存、缓存访问、采样与标准化、验证搜索、最终训练和最终预测分别计时。最终训练耗时不包含验证搜索和图特征构造；应连同这些额外成本报告。已有回归和社区结果在独立文件保留，不能把分类特征的提升直接套用到回归或社区任务。

纯内存协议测试：`python -B -m unittest test_mlp_protocol -v`，覆盖无向边去重、自环、孤立节点、两阶聚合、采样不越出训练 ID 和标准化不拟合留出集等关键约束。

## 本机已验证结果的位置

- 三次原始运行：`D:\作业\数据挖掘\codex_proc\cpu-tuning\v3-mlp-{42,123,2026}`。
- 最终核验与汇总：`D:\作业\数据挖掘\codex_proc\cpu-tuning\v3-mlp-summary-final`。
- 同一 Miniforge 环境生成的 PNG/SVG 图表：`D:\作业\数据挖掘\codex_proc\cpu-tuning\v3-mlp-figures-conda`。
- `fact_notes.md` 保存本轮事实及限制；`validation_selected_mean_sd.csv` 保存实际按验证集选择方案的最终表现。
- 实际训练版本快照存于 `codex_proc\cpu-tuning\v3-executed-mlp-source.py`，与协议内源文件哈希一致。随后仅加固异常索引的拒绝检查；所有已有划分都已重新通过该检查。特征、训练、采样和指标算法未变化。

三次正式运行中，原始特征 MLP 的 accuracy 为 **0.5448 ± 0.0065**，二阶邻居特征为 **0.6867 ± 0.0023**；对应 macro-F1 从 **0.3403 ± 0.0073** 到 **0.4861 ± 0.0075**。二阶特征加过采样的 macro-F1 为 **0.4908 ± 0.0103**，但准确率 **0.6831 ± 0.0065**，增益并不在每个种子上都成立。这些是固定组别的消融结果；按验证集选出的方案在三个种子分别是 hop2、hop2_ros、hop2，其最终平均准确率 **0.6824 ± 0.0053**，macro-F1 **0.4847 ± 0.0085**。
