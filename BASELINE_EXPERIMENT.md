# 树模型与社区参数调优实验

## 设计

教师基准命令 `benchmark` 使用固定种子无放回抽样，分类分层 70/30 划分。比较分类及回归随机森林的 `n_jobs=1、2、4`，运行 MLP、Logistic Regression 和 HGBR，并单独以 80/20 划分运行全量随机森林分类。记录训练与预测耗时、质量指标、警告以及森林平均节点数和深度。HGBR 不支持 `n_jobs`，只运行一次，底层线程由环境变量控制；不能把它标作不同 `n_jobs` 的对照。教师口径 Louvain 复用抽样随机数状态，从边构造诱导子图（不添加孤立节点，保留自环）。下述调优流程采用不同图构造口径，不能与教师图基准直接比较。

`tune` 将 70% 开发集进一步按 80/20 划为训练和验证，正式规模为 28K/7K/15K。分类按验证 macro-F1 最大、再按 accuracy 最大选择随机森林叶节点和深度候选；回归按验证 MAE 最小选择。最终在训练加验证集重训，并在相同测试集评估。分类还对照类别加权和决策树，报告 accuracy、macro/weighted F1 和逐类召回；F1 固定使用数据集全部类别，避免不同划分遗漏稀有类别后改变分母。回归对照 HGBR 和 Extra Trees，报告 R² 与 MAE。不同随机种子重新抽样划分。图任务比较同一无向诱导子图上的不同 Louvain 分辨率，保留孤立节点并去除自环；不同分辨率使用不同目标，因此同时报告统一分辨率 1 的模块度。

## 运行

准备及变量见 README。先用较小规模估计成本，再运行正式规模；稀有类别过少时分层可能失败，应增大样本。正式配置默认 50,000 样本、150 棵树和 30,000 图节点。

```powershell
python -B run.py benchmark --data $data --out "$work/benchmark-small" --samples 20000 --trees 10 --skip-full --skip-graph
python -B run.py benchmark --data $data --out "$work/benchmark-full"
foreach ($seed in @(42,123,2026)) {
  python -B run.py tune --data $data --out "$work/tune-$seed" --seed $seed
  if ($LASTEXITCODE -ne 0) { throw "调优失败：$seed" }
}
python -B run.py summarize-baseline --runs "$work/tune-42" "$work/tune-123" "$work/tune-2026" --out "$work/tune-summary"
python -B run.py graph-restarts --data $data --run "$work/tune-42" --out "$work/graph-42" --restarts 3
```

固定图重启在分辨率 1 上比较不同随机初始化，记录模块度和额外耗时。若要汇总图重启，三个种子都需各运行一次，再向 `summarize-baseline` 的 `--graph-restarts` 传入对应目录。图的模块度是内部指标，不证明外部社区真实性。

图重启在运行前核对基线的 Python、NumPy、NetworkX 和包源码哈希，任一缺失或不一致时要求重跑基线。`selection.json` 保存这些版本信息，汇总时再次核对；旧图重启结果缺少这些字段时需重新运行，不能混入新版耗时比较。

可用 `tune --tasks classification`、`--tasks regression`、`--tasks graph` 或它们的组合只运行指定任务。`summarize-baseline` 支持这些运行，但同次汇总的任务集合必须一致；表格和图只包含所选任务，没有分类时不生成逐类召回率表。`--graph-restarts` 只适用于包含图任务的运行。

## 输出和可比性

`metadata.json` 记录配置、环境、数据文件名和 SHA-256 及包源码哈希；`split_indices.npz`、`graph_nodes.npy` 保存划分；`results.jsonl` 每项完成后立即写入，包含模型参数、警告、失败及耗时。`summary.csv`、`class_recall.csv` 和 `charts.svg` 用于检查结果。训练和预测计时不包括加载、预处理和指标计算；调参总耗时另外记录。

只有所选任务全部成功，元数据才写入 `status: complete` 及相关划分、图节点、原始结果和分类逐类召回率表的 SHA-256。汇总器按所选任务验证所需文件，并检查逐类召回率是否有限且在 0～1 范围内、测试计数是否一致、模型及类别是否完整。中断目录可供检查，但不能直接作为后续阶段的输入。旧分类结果缺少 `class_recall.csv` 哈希时也需重新运行，不应给历史文件补签以绕过检查。

多种子汇总核对数据、参数、依赖、硬件、数值库线程池和源码哈希；缺少源码哈希的旧基线不能混入新版汇总。MLP 和图重启需用 `--data` 重新指定原始数据，并通过哈希检查。按时间划分的泛化能力、不同硬件的运行时长以及 OGB 排名不能由此随机划分直接推断。
