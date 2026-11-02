"""Reproducible CPU experiment; outputs are real measurements only."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import networkx as nx
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_info


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_array(path, dtype):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        return np.loadtxt(f, delimiter=',', dtype=dtype, ndmin=2)


def record(out, result):
    with (out / 'results.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
    print(json.dumps(result, ensure_ascii=False), flush=True)


def measures(task, truth, prediction):
    if task == 'classification':
        return {'accuracy': float(accuracy_score(truth, prediction)),
                'macro_f1': float(f1_score(truth, prediction, average='macro', zero_division=0)),
                'weighted_f1': float(f1_score(truth, prediction, average='weighted', zero_division=0))}
    return {'r2': float(r2_score(truth, prediction)), 'mae': float(mean_absolute_error(truth, prediction))}


def make_model(task, name, params, args):
    if task == 'classification':
        if name == 'decision_tree':
            return DecisionTreeClassifier(random_state=args.seed, **params)
        return RandomForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, random_state=args.seed, **params)
    if name == 'hist_gradient_boosting':
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, early_stopping=True,
                                             validation_fraction=0.1, n_iter_no_change=20, random_state=args.seed)
    forest = ExtraTreesRegressor if name == 'extra_trees' else RandomForestRegressor
    return forest(n_estimators=args.trees, n_jobs=args.jobs, random_state=args.seed,
                                 max_features='sqrt', **params)


def fit_eval(out, task, name, stage, params, X, y, tr, ev, args, recalls=None):
    result = {'task': task, 'name': name, 'stage': stage, 'params': params,
              'seed': args.seed, 'n_train': len(tr), 'n_eval': len(ev), 'status': 'ok'}
    try:
        model = make_model(task, name, params, args)
        result['model_params'] = model.get_params(deep=False)
        X_train, y_train, X_eval, y_eval = X[tr], y[tr], X[ev], y[ev]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            start = time.perf_counter(); model.fit(X_train, y_train); result['fit_seconds'] = time.perf_counter() - start
            start = time.perf_counter(); pred = model.predict(X_eval); result['predict_seconds'] = time.perf_counter() - start
        result['warnings'] = [str(w.message) for w in caught]
        result['metrics'] = measures(task, y_eval, pred)
        if recalls is not None:
            labels = np.unique(y)
            values = recall_score(y[ev], pred, labels=labels, average=None, zero_division=0)
            counts = Counter(y[ev].tolist())
            for label, value in zip(labels, values):
                recalls.append([name, int(label), int(counts[int(label)]), float(value)])
    except Exception:
        result['status'] = 'error'; result['error'] = traceback.format_exc()
    record(out, result)
    return result


def supervised(out, task, X, y, train, val, test, args, recalls):
    candidates = [dict(min_samples_leaf=5), dict(min_samples_leaf=2),
                  dict(min_samples_leaf=10), dict(min_samples_leaf=5, max_depth=20)]
    started = time.perf_counter()
    trials = [fit_eval(out, task, 'rf_candidate_%d' % i, 'validation', p, X, y, train, val, args)
              for i, p in enumerate(candidates)]
    duration = time.perf_counter() - started
    good = [r for r in trials if r['status'] == 'ok']
    if not good:
        raise RuntimeError(task + ': all candidates failed')
    best = max(good, key=lambda r: r['metrics']['accuracy'] if task == 'classification' else -r['metrics']['mae'])
    record(out, {'task': task, 'stage': 'selection', 'status': 'ok', 'selected': best['name'],
                 'params': best['params'], 'criterion': 'accuracy' if task == 'classification' else 'mae',
                 'search_seconds': duration, 'candidate_count': len(candidates)})
    both = np.concatenate([train, val])
    variants = [('rf_baseline', candidates[0]), ('rf_tuned', best['params'])]
    if task == 'classification':
        variants.append(('rf_balanced', dict(min_samples_leaf=5, class_weight='balanced')))
        variants.append(('decision_tree', dict(min_samples_leaf=5)))
    else:
        variants.append(('hist_gradient_boosting', {}))
        variants.append(('extra_trees', dict(min_samples_leaf=5)))
    failed = any(r['status'] != 'ok' for r in trials)
    for name, params in variants:
        r = fit_eval(out, task, name, 'test', params, X, y, both, test, args,
                     recalls if task == 'classification' else None)
        failed |= r['status'] != 'ok'
    if failed:
        raise RuntimeError(task + ': one or more runs failed; see results.jsonl')


def graph_experiment(out, edge_path, count, args, raw_count):
    edges = read_array(edge_path, np.int64)
    if edges.shape[1] != 2 or edges.size == 0 or edges.min() < 0 or edges.max() >= raw_count:
        raise ValueError('edge table must have two valid node-ID columns')
    rng = np.random.RandomState(args.seed + 1)
    nodes = rng.choice(raw_count, count, replace=False)
    np.save(out / 'graph_nodes.npy', nodes)
    selected = np.zeros(raw_count, dtype=bool); selected[nodes] = True
    induced = edges[selected[edges[:, 0]] & selected[edges[:, 1]]]
    g = nx.Graph(); g.add_nodes_from(map(int, nodes)); g.add_edges_from(induced.tolist())
    loops = nx.number_of_selfloops(g); g.remove_edges_from(list(nx.selfloop_edges(g)))
    info = {'nodes': g.number_of_nodes(), 'edges': g.number_of_edges(),
            'isolates': nx.number_of_isolates(g), 'removed_self_loops': loops,
            'induced_directed_rows': len(induced)}
    if not g.number_of_edges():
        raise ValueError('induced graph has no edges; increase graph-nodes')
    failed = False
    for resolution in (1.0, 0.5, 1.5):
        r = {'task': 'graph', 'stage': 'comparison', 'name': 'louvain', 'resolution': resolution,
             'seed': args.seed, 'graph': info, 'status': 'ok'}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                start = time.perf_counter()
                communities = nx.community.louvain_communities(g, seed=args.seed, resolution=resolution)
                r['fit_seconds'] = time.perf_counter() - start
                r['communities'] = len(communities)
                r['modularity_at_resolution'] = float(nx.community.modularity(g, communities, resolution=resolution))
                r['modularity_at_1'] = float(nx.community.modularity(g, communities, resolution=1))
                r['warnings'] = [str(w.message) for w in caught]
        except Exception:
            r['status'] = 'error'; r['error'] = traceback.format_exc(); failed = True
        record(out, r)
    if failed:
        raise RuntimeError('one or more graph runs failed; see results.jsonl')


def summarize(out):
    path = out / 'results.jsonl'
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    fields = ['task', 'stage', 'name', 'status', 'fit_seconds', 'predict_seconds', 'search_seconds',
              'accuracy', 'macro_f1', 'weighted_f1', 'r2', 'mae', 'resolution', 'communities',
              'modularity_at_resolution', 'modularity_at_1']
    with (out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow(dict(r, **r.get('metrics', {})))
    # A compact, self-contained SVG with measured values; no external plotting dependency.
    display = [r for r in rows if r.get('status') == 'ok' and r.get('stage') in ('test', 'comparison')]
    height = 60 + 36 * len(display)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="30" font-size="18">Measured results (see summary.csv for all metrics)</text>']
    max_time = max((r['fit_seconds'] for r in display), default=1) or 1
    for i, r in enumerate(display):
        m = r.get('metrics', {})
        label = f"{r['task']} / {r['name']}" + (f" resolution={r['resolution']}" if 'resolution' in r else '')
        value = (f"accuracy={m['accuracy']:.4f}, macro F1={m['macro_f1']:.4f}" if 'accuracy' in m else
                 f"R2={m['r2']:.4f}, MAE={m['mae']:.4f}" if 'r2' in m else
                 f"communities={r['communities']}, modularity@1={r['modularity_at_1']:.4f}")
        width = 180 * r['fit_seconds'] / max_time
        parts.append(f'<text x="20" y="{60+i*36}" font-size="14">{escape(label)}: {escape(value)}; fit={r["fit_seconds"]:.2f}s</text>')
        parts.append(f'<rect x="900" y="{48+i*36}" width="{width:.2f}" height="12" fill="#3973ac"/>')
    parts.append('</svg>'); (out / 'charts.svg').write_text('\n'.join(parts), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=50000); p.add_argument('--graph-nodes', type=int, default=30000)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--trees', type=int, default=150)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--tasks', nargs='+', choices=['classification', 'regression', 'graph'],
                   default=['classification', 'regression', 'graph'])
    a = p.parse_args()
    if min(a.samples, a.graph_nodes, a.trees, a.jobs) < 1:
        p.error('samples, graph-nodes, trees and jobs must be positive')
    raw = a.data / 'raw' if (a.data / 'raw').is_dir() else a.data
    year = next((raw / n for n in ('node-year.csv.gz', 'node_year.csv.gz') if (raw / n).is_file()), None)
    files = {'features': raw / 'node-feat.csv.gz', 'labels': raw / 'node-label.csv.gz',
             'year': year, 'edges': raw / 'edge.csv.gz'}
    for key, path in files.items():
        if path is None or not path.is_file():
            raise FileNotFoundError(f'missing {key}: {path} under {raw}')
    a.out.mkdir(parents=True, exist_ok=False)
    try:
        metadata = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
                    'logical_cpus': os.cpu_count(), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
                    'networkx': nx.__version__, 'threadpools': threadpool_info(),
                    'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                    'files': {k: {'path': str(v.resolve()), 'sha256': sha256(v)} for k, v in files.items()}}
        X = read_array(files['features'], np.float32)
        labels = read_array(files['labels'], np.float64); years = read_array(files['year'], np.float64)
        if labels.shape != (len(X), 1) or years.shape != (len(X), 1):
            raise ValueError('features, labels and years must have matching row counts and one target column')
        if not all(np.isfinite(v).all() for v in (X, labels, years)) or not np.equal(labels, np.floor(labels)).all():
            raise ValueError('missing/nonfinite data or non-integer labels')
        y = labels[:, 0].astype(np.int64); yr = years[:, 0]
        if a.samples > len(X) or a.graph_nodes > len(X):
            raise ValueError('sample size exceeds node count')
        counts = Counter(y.tolist())
        metadata['data'] = {'features_shape': list(X.shape), 'label_counts': {str(k): v for k, v in sorted(counts.items())},
                            'year_min': float(yr.min()), 'year_max': float(yr.max())}
        dump(a.out / 'metadata.json', metadata)
        idx = np.random.RandomState(a.seed).choice(len(X), a.samples, replace=False)
        dev, test = train_test_split(idx, test_size=.3, stratify=y[idx], random_state=a.seed)
        train, val = train_test_split(dev, test_size=.2, stratify=y[dev], random_state=a.seed)
        np.savez_compressed(a.out / 'split_indices.npz', sampled=idx, train=train, validation=val, test=test)
        recalls = []; failures = []
        for task in a.tasks:
            try:
                if task == 'graph': graph_experiment(a.out, files['edges'], a.graph_nodes, a, len(X))
                else: supervised(a.out, task, X, y if task == 'classification' else yr, train, val, test, a, recalls)
            except Exception:
                failures.append(task)
                record(a.out, {'task': task, 'stage': 'task', 'status': 'error', 'error': traceback.format_exc()})
        if recalls:
            with (a.out / 'class_recall.csv').open('w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f); w.writerow(['model', 'label', 'test_count', 'recall']); w.writerows(recalls)
        if failures: raise RuntimeError('failed tasks: ' + ', '.join(failures))
    except Exception:
        record(a.out, {'task': 'run', 'stage': 'run', 'status': 'error', 'error': traceback.format_exc()})
        raise
    finally:
        summarize(a.out)


if __name__ == '__main__':
    main()
