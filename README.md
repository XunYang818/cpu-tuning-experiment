# CPU 数据挖掘调优实验

以教师的 `benchmark.py` 为参考，覆盖论文分类、年份回归和引用图社区发现。代码只使用 CPU。正式结果必须使用真实数据运行后获得，项目不预置实验结果。

## 实验设计

1. **参数调优**：分类及回归随机森林以教师的 150 棵树、叶节点至少 5 条为基线，在训练集上尝试有限个叶节点和深度参数，用验证集选择，按 accuracy（分类）或 MAE（回归）排序；同分按候选顺序。基线与选中方案各自用训练加验证集重新拟合，再在相同测试集评估。
2. **长尾处理**：分类随机森林独立比较 `class_weight=balanced`，与未加权基线同配置；同时报告 accuracy、macro/weighted F1、每类召回率。类别加权是模型改动，多指标是评价方式，不保证少数类或整体效果一定改善。
3. **模型及图参数比较**：回归增加教师的直方图梯度提升模型作为模型对照；图任务以默认分辨率 1 的 Louvain 为基线，比较 0.5、1.5，固定同一无向诱导子图。模块度同时按各自分辨率和统一分辨率 1 计算；不同分辨率的目标函数不同，不可直接以各自模块度判断优劣。图参数为探索性对照，不用测试集选型。

监督样本默认 50,000，先按类别分层划分 70% 开发集及 30% 测试集，再将开发集分为训练/验证（80%/20%），分类回归共用索引；因此与教师原脚本 70/30 测试划分相同，但选型需要额外验证集。全类样本过少不能分层时程序明确报错；请增大样本。图默认随机抽取 30,000 个节点，包含孤立节点，引用边转无向简单图并去除自环；与教师脚本遗漏孤立节点的实现存在口径差异。种子固定时同一数据版本可重复，数据文件 SHA-256 和实际划分索引均会保存。

## 环境与数据

需 Python 3.9+、numpy、scikit-learn、networkx（NetworkX >= 2.8，提供 Louvain）。安装或下载前请先自行确认许可；本项目不会自动安装或下载。`requirements.txt` 给出兼容范围，正式运行后记录实际版本及数值库线程信息。本机已创建 Miniforge 环境 `cpu-tuning-experiment`，通过 `conda run -n cpu-tuning-experiment python` 运行可正确激活依赖的动态库；实际包版本以每次输出的 metadata 为准。

将已获得的数据集解压，参数 `--data` 指向包含 `raw` 子目录的 arxiv 目录，或直接指向 `raw`。需 `node-feat.csv.gz`、`node-label.csv.gz`、`node-year.csv.gz`（也支持 `node_year.csv.gz`）和 `edge.csv.gz`。不修改原始数据。可从助教给出的地址手动获取；下载前需得到许可。

## 运行（PowerShell）

工作目录切换到本项目。过程和输出写入挂载项目根目录 `codex_proc`；每次运行需指定一个**不存在**的输出目录，避免覆盖和混淆。示例路径请替换为本机实际数据位置。

```powershell
$conda = 'D:\miniforge3\Scripts\conda.exe'
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& $conda run --no-capture-output -n cpu-tuning-experiment python experiment.py --data 'D:\作业\数据挖掘\codex_proc\cpu-tuning\data\extracted\arxiv' --out 'D:\作业\数据挖掘\codex_proc\cpu-tuning\smoke-42' --samples 20000 --graph-nodes 3000 --trees 10 --jobs 2
& $conda run --no-capture-output -n cpu-tuning-experiment python experiment.py --data 'D:\作业\数据挖掘\codex_proc\cpu-tuning\data\extracted\arxiv' --out 'D:\作业\数据挖掘\codex_proc\cpu-tuning\full-42' --jobs 4
```

真实数据种子 42 下，10,000 条样本的稀有类无法完成第二次分层；已用 20,000 条完成试跑。小样本或其他种子可能因稀有类无法分层；应增大样本量。`--trees 10` 仅适合调试，正式基线默认 150。`--seed` 可更改以重复全流程；不同种子改变抽样与划分，不是同一测试集上的对照。`--tasks classification regression graph` 可选择任务；每次输出独立目录。先小样本估时，再执行正式规模；网格有限，正式运行仍可能需要数小时。

## 输出与口径

`metadata.json` 记录参数、版本、硬件、线程变量、数据摘要和哈希；`split_indices.npz` 保存原始节点 ID；`results.jsonl` 每完成一项立即追加，可追踪候选、验证结果、最终测试、警告、异常与耗时；`summary.csv` 是便于筛选的扁平摘要；`class_recall.csv` 为最终分类各类召回率；`graph_nodes.npy` 保存图节点；`charts.svg` 由实测最终结果生成。调参计时包括候选训练和验证预测；最终训练/预测时间单独记录。训练计时包括模型内的全部处理，加载数据和计算指标不计入。输入完整性检查包含行数、有限值、标签整数、边界和标签分布。失败会写入结果并导致进程非零退出，不将失败伪装成成功。日志与结果仅作为事实素材，不是报告。

`n_jobs` 只控制支持它的随机森林并行度，不等于全部底层线程；应在启动 Python 前设置上述线程变量。系统硬件、库版本、后台负载影响计时。记录结果后请对照原始 JSONL 核实 CSV 和图。代码只搜索有限候选，不保证最优；测试集不参与选择，测试数据仍是随机抽样而非按时间切分，不能直接代表未来论文泛化能力。由于图的社区没有标签，模块度不是外部真实性指标。
