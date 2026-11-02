# 基线、回归与社区发现实验

详细记录原有树模型、回归和图任务的协议。统一入口见 README。

## 实验设计

1. **参数调优**：分类及回归随机森林以教师的 150 棵树、叶节点至少 5 条为基线，在训练集上尝试有限个叶节点和深度参数，用验证集选择，按 accuracy（分类）或 MAE（回归）排序；同分按候选顺序。基线与选中方案各自用训练加验证集重新拟合，再在相同测试集评估。
2. **长尾处理**：分类随机森林独立比较 `class_weight=balanced`，与未加权基线同配置；同时报告 accuracy、macro/weighted F1、每类召回率。类别加权是模型改动，多指标是评价方式，不保证少数类或整体效果一定改善。
3. **模型及图参数比较**：分类比较单棵决策树，回归增加极端随机树作为替代模型，并保留教师的直方图梯度提升回归作为基线对照；替代模型采用事先指定参数、未基于测试集选择。图任务以默认分辨率 1 的 Louvain 为基线，比较 0.5、1.5，固定同一无向诱导子图。模块度同时按各自分辨率和统一分辨率 1 计算；不同分辨率的目标函数不同，不可直接以各自模块度判断优劣。图参数为探索性对照，不用测试集选型。

监督样本默认 50,000，先按类别分层划分 70% 开发集及 30% 测试集，再将开发集分为训练/验证（80%/20%），分类回归共用索引；因此与教师原脚本 70/30 测试划分相同，但选型需要额外验证集。全类样本过少不能分层时程序明确报错；请增大样本。图默认随机抽取 30,000 个节点，包含孤立节点，引用边转无向简单图并去除自环；与教师脚本遗漏孤立节点的实现存在口径差异。种子固定时同一数据版本可重复，数据文件 SHA-256 和实际划分索引均会保存。

## 环境与数据

需 Python 3.9+、numpy、scikit-learn、networkx（NetworkX >= 2.8，提供 Louvain）。安装或下载前请先自行确认许可；本项目不会自动安装或下载。`requirements.txt` 给出兼容范围，正式运行后记录实际版本及数值库线程信息。本机已创建 Miniforge 环境 `cpu-tuning-experiment`，通过 `conda run -n cpu-tuning-experiment python` 运行可正确激活依赖的动态库；实际包版本以每次输出的 metadata 为准；`environment.yml` 固定主要依赖版本，`conda-win-64.lock` 记录本机 Windows 环境的精确 Conda 包，适合在同平台复现。

将已获得的数据集解压，参数 `--data` 指向包含 `raw` 子目录的 arxiv 目录，或直接指向 `raw`。需 `node-feat.csv.gz`、`node-label.csv.gz`、`node-year.csv.gz`（也支持 `node_year.csv.gz`）和 `edge.csv.gz`。不修改原始数据。可从助教给出的地址手动获取；下载前需得到许可。

## 运行（PowerShell）

工作目录切换到本项目。过程和输出写入挂载项目根目录 `codex_proc`；每次运行需指定一个**不存在**的输出目录，避免覆盖和混淆。示例路径请替换为本机实际数据位置。

```powershell
$conda = 'D:\miniforge3\Scripts\conda.exe'
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& $conda run --no-capture-output -n cpu-tuning-experiment python -B run.py baseline --data 'D:\作业\数据挖掘\codex_proc\cpu-tuning\data\extracted\arxiv' --out 'D:\作业\数据挖掘\codex_proc\cpu-tuning\smoke-42' --samples 20000 --graph-nodes 3000 --trees 10 --jobs 2
& $conda run --no-capture-output -n cpu-tuning-experiment python -B run.py baseline --data 'D:\作业\数据挖掘\codex_proc\cpu-tuning\data\extracted\arxiv' --out 'D:\作业\数据挖掘\codex_proc\cpu-tuning\full-42' --jobs 4
```

真实数据种子 42 下，10,000 条样本的稀有类无法完成第二次分层；已用 20,000 条完成试跑。小样本或其他种子可能因稀有类无法分层；应增大样本量。`--trees 10` 仅适合调试，正式基线默认 150。`--seed` 可更改以重复全流程；不同种子改变抽样与划分，不是同一测试集上的对照。`--tasks classification regression graph` 可选择任务；每次输出独立目录。先小样本估时，再执行正式规模；网格有限，正式运行仍可能需要数小时。

## 输出与口径

`metadata.json` 记录参数、版本、硬件、线程变量、数据摘要和哈希；`split_indices.npz` 保存原始节点 ID；`results.jsonl` 每完成一项立即追加，可追踪候选、验证结果、最终测试、警告、异常与耗时；`summary.csv` 是便于筛选的扁平摘要；`class_recall.csv` 为最终分类各类召回率；`graph_nodes.npy` 保存图节点；`charts.svg` 由实测最终结果生成。调参计时包括候选训练和验证预测；最终训练/预测时间单独记录。训练计时包括模型内的全部处理；加载数据、数组切片和计算指标不计入。旧版运行的计时包括索引切片，不能与新版耗时直接混合比较。输入完整性检查包含行数、有限值、标签整数、边界和标签分布。失败会写入结果并导致进程非零退出，不将失败伪装成成功。日志与结果仅作为事实素材，不是报告。

`n_jobs` 只控制支持它的随机森林并行度，不等于全部底层线程；应在启动 Python 前设置上述线程变量。系统硬件、库版本、后台负载影响计时。记录结果后请对照原始 JSONL 核实 CSV 和图。代码只搜索有限候选，不保证最优；测试集不参与选择，测试数据仍是随机抽样而非按时间切分，不能直接代表未来论文泛化能力。由于图的社区没有标签，模块度不是外部真实性指标。

## 多种子汇总

每个种子用相同的规模、参数、依赖和线程设置独立运行，例如 `--seed 42`、`--seed 123`、`--seed 2026`，输出目录必须各不相同。同一种子内各模型使用相同测试集；不同种子重新抽样和划分，不代表固定测试集上的重复计时。运行完后：

```powershell
& $conda run --no-capture-output -n cpu-tuning-experiment python -B run.py summarize-baseline --runs 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v2-full-50000-42' 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v2-full-50000-123' 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v2-full-50000-2026' --out 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v2-aggregate'
```

脚本要求每个任务和方法的结果完整且成功，并核对数据哈希、规模、线程、种子唯一性和库版本；不应将不同版本代码的结果合并。输出逐种子指标、均值及样本标准差、相对随机森林基线的配对差值、各类别召回率和比较图。每个种子中的调参各自使用其验证集，`rf_tuned` 的选中参数可能不同；重复结果体现的是整套流程的波动。样本标准差基于 `n-1`，仅三次运行时不宜夸大统计结论。

## 图任务：固定目标的多次重启

为避免仅通过改变分辨率比较不同优化目标，可在已经完成的单种子图实验上执行：

```powershell
& $conda run --no-capture-output -n cpu-tuning-experiment python -B run.py graph-restarts --run 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v2-full-50000-42' --out 'D:\作业\数据挖掘\codex_proc\cpu-tuning\v2-graph-restarts-42' --restarts 3
```

程序按原实验保存的节点及相同边文件重建图，核对边文件哈希、图规模、NetworkX 版本和第一轮模块度，然后固定分辨率 1，运行 3 个随机初始化，选出同一图上模块度最高的划分。`trials.jsonl` 逐次记录结果，`selection.json` 记录与第一轮的差值、全部重启的累计算法耗时和模块度计算耗时。选择过程针对同一张图的内部目标，因包含基线结果，最优值不会低于基线，但这不证明外部社区真实性提高；额外耗时也须一同报告。不同采样种子的图应分别运行、分别比较。三次运行后，可在 `aggregate.py` 的原命令中增加 `--graph-restarts`，后接三个重启结果目录，汇总脚本会核对每个主运行均有对应结果并额外生成 `graph_restarts.csv`；汇总输出目录须是新目录。
