# CPU 数据挖掘调优实验

以教师 `benchmark.py` 为参考，覆盖论文分类、年份回归和图社区发现，全部在 CPU 上执行。项目按三条主线组织：模型与参数对照、引用关系辅助特征、类别不均衡处理。详细协议分别见 [基线及图任务](BASELINE_EXPERIMENT.md) 和 [MLP 分类改进](MLP_EXPERIMENT.md)。

## 从哪里读代码

```text
run.py                       统一命令入口
cpu_tuning/
  cli.py                     命令分发，按需加载各阶段
  io.py                      数据读取、哈希、JSON 结果记录
  protocol.py                MLP 对照配置、划分检查、训练集采样、分类指标
  features.py                无标签引用图特征及缓存
  baseline.py                分类/回归基线、参数搜索、Louvain 对照
  mlp.py                     外部验证早停、最终重训、预测保存
  graph_restarts.py          固定图和分辨率的 Louvain 重启
  baseline_summary.py        基线多种子汇总
  mlp_summary.py             从保存预测核验指标并汇总
  plots.py                   从已核验的 CSV 生成 PNG/SVG
tests/                       协议、命令入口和小规模完整流程测试
```

数据与记录读写、协议、特征模块不依赖训练入口；汇总器不导入训练脚本。根目录的六个旧脚本仅用于兼容原有命令，实际实现集中在包内。

## 环境与数据

本机 Miniforge 环境：`cpu-tuning-experiment`，Python 3.11。`environment.yml` 固定主要依赖，`conda-win-64.lock` 保存 Windows 精确包清单；`requirements.txt` 为训练依赖，`requirements-plot.txt` 追加绘图依赖。安装和下载必须先取得许可，项目本身不会执行自动安装或下载。

数据参数指向包含 `raw` 的 arxiv 目录，或直接指向 `raw`。需要 `node-feat.csv.gz`、`node-label.csv.gz`、`node-year.csv.gz`（也支持下划线版本）及 `edge.csv.gz`。本机数据位于 `D:\作业\数据挖掘\codex_proc\cpu-tuning\data\extracted\arxiv`。

所有过程产物放在挂载项目根目录的 `codex_proc`；每次运行或汇总输出使用新目录，避免覆盖。随机种子、数据哈希、划分、依赖和线程设置记录在对应运行的元数据中。

## 统一运行入口

```powershell
$conda = 'D:\miniforge3\Scripts\conda.exe'
$proc = 'D:\作业\数据挖掘\codex_proc\cpu-tuning'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:MPLCONFIGDIR="$proc\mpl-cache"
& $conda run --no-capture-output -n cpu-tuning-experiment python -B run.py --help
```

| 命令 | 用途 | 主要输入 |
|---|---|---|
| `baseline` | 运行分类、回归和图基线 | `--data`、`--out` |
| `mlp` | 运行五组 MLP 消融 | `--run` 已完成基线目录、`--cache`、`--out` |
| `graph-restarts` | 固定图上的多次 Louvain 初始化 | `--run`、`--out` |
| `summarize-baseline` | 汇总基线及图结果 | `--runs`、`--out` |
| `summarize-mlp` | 从保存预测重算指标并汇总 | `--runs`、`--out` |
| `plot-mlp` | 生成消融、耗时和学习曲线图 | `--summary`、`--out` |

例如 `python -B run.py mlp --help` 查看某一阶段的完整参数。也支持 `python -B -m cpu_tuning --help`。

下面从已有三个基线运行复现分类改进，并保存到一个新目录：

```powershell
$session = "$proc\reproduce-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$mlpRuns = @()
foreach ($seed in @(42,123,2026)) {
  $mlpOut = "$session\mlp-$seed"
  & $conda run --no-capture-output -n cpu-tuning-experiment python -B run.py mlp --run "$proc\v2-full-50000-$seed" --cache "$proc\v3-graph-features" --out $mlpOut
  if ($LASTEXITCODE -ne 0) { throw "MLP 实验失败：$seed" }
  $mlpRuns += $mlpOut
}
& $conda run --no-capture-output -n cpu-tuning-experiment python -B run.py summarize-mlp --runs $mlpRuns --out "$session\summary"
if ($LASTEXITCODE -ne 0) { throw '汇总失败' }
& $conda run --no-capture-output -n cpu-tuning-experiment python -B run.py plot-mlp --summary "$session\summary" --out "$session\figures"
```

首次运行需先按 [基线说明](BASELINE_EXPERIMENT.md) 生成对应的保存划分。正式规模为 50K 监督样本、30K 图节点；真实数据 seed=42 下的 10K 抽样因极少数类无法再分层，试跑可使用 20K 和较少树数。不同种子内的模型共用测试集，不同种子重新抽样和划分。

## 验证

```powershell
$env:CPU_TUNING_TEST_ROOT="$proc\tests"
& $conda run --no-capture-output -n cpu-tuning-experiment python -B -m unittest discover -s tests -v
```

测试包含无向图去重与自环、孤立节点、二阶聚合、训练集采样边界、标准化范围、划分互斥、统一入口及旧入口兼容，还会用明确标记的合成数据运行“基线 → MLP 五组对照”的完整流程。合成数据只用于软件验证；产物保留在指定测试目录，不作为实验成绩。

## 已有真实结果

| 内容 | 本机目录（均在 `$proc` 下） |
|---|---|
| 三次基线原始记录 | `v2-full-50000-42`、`v2-full-50000-123`、`v2-full-50000-2026` |
| 基线和图任务汇总 | `v2-aggregate-final` |
| 三次 MLP 原始记录与预测 | `v3-mlp-42`、`v3-mlp-123`、`v3-mlp-2026` |
| MLP 核验与汇总 | `v3-mlp-summary-final` |
| MLP 图表 | `v3-mlp-figures-conda` |

`results.jsonl` 保存原始记录，MLP 的 `test_predictions.npz` 可独立重算测试指标。`mean_sd.csv` 为均值及样本标准差（n=3），`paired_*` 为同种子内差值；`selected_by_validation.csv` 记录验证选择，不能用测试最优组替代该流程。新运行额外记录包内源代码哈希；已有记录仍保留原执行版本的哈希和源码快照。

引用特征使用全图无标签特征和引用边，属于传导式设置；与纯特征输入的信息条件不同。沿用的随机分层测试集已在前序实验中查看，不能称为新的盲测或直接对比 OGB 时间划分榜单。更多局限见 MLP 说明。

## 旧命令兼容

`experiment.py`、`aggregate.py`、`graph_restarts.py`、`mlp_experiment.py`、`summarize_mlp.py`、`plot_mlp.py` 保留原参数接口，分别转发到上述六个命令的实现。新使用者从 `run.py` 开始即可。
