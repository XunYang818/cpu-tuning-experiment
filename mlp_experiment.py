"""Controlled CPU MLP ablations using saved splits and label-free graph features."""
import argparse
import csv
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy import sparse
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info
from experiment import read_array, sha256, dump, record

ARMS = [('raw', 0, False), ('raw_ros', 0, True), ('hop1', 1, False),
        ('hop2', 2, False), ('hop2_ros', 2, True)]
MLP_PARAMS = dict(hidden_layer_sizes=(128,), learning_rate_init=3e-4,
                  alpha=1e-3, early_stopping=False, solver='adam', batch_size=200)


def aggregate_features(X, edges):
    """Only node attributes and edge IDs are accepted; no labels or years."""
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.size == 0:
        raise ValueError('invalid edge shape')
    if not np.isfinite(edges).all() or not np.equal(edges, np.floor(edges)).all():
        raise ValueError('edges must be integer node IDs')
    edges = edges.astype(np.int64)
    if edges.min() < 0 or edges.max() >= len(X): raise ValueError('edge ID out of range')
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(X), len(X)))
    A.sum_duplicates(); A.data.fill(1); A.setdiag(1); A.sort_indices()
    degree = np.asarray(A.sum(axis=1)).ravel()
    P = sparse.diags(1 / degree, format='csr') @ A
    if not np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1, atol=1e-6):
        raise ValueError('row normalization failed')
    H1 = np.asarray(P @ X, dtype=np.float32)
    H2 = np.asarray(P @ H1, dtype=np.float32)
    return H1, H2, {'nodes': len(X), 'adjacency_nnz_with_self': int(A.nnz),
                    'undirected_edges_without_self': int((A.nnz-len(X))//2),
                    'isolates_before_self_loops': int(np.count_nonzero(degree == 1))}


def graph_cache(cache, X, edge_path, feature_hash):
    signature = {'features_sha256': feature_hash, 'edges_sha256': sha256(edge_path),
                 'protocol': 'undirected-binary-self-loop-row-mean-v1', 'shape': list(X.shape)}
    if cache.exists():
        meta = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
        if meta['signature'] != signature: raise ValueError('graph cache signature mismatch')
        arrays = []
        for name in ('hop1', 'hop2'):
            path = cache/(name+'.npy')
            if sha256(path) != meta['array_sha256'][name]: raise ValueError('corrupted graph cache')
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            if a.shape != X.shape or a.dtype != np.float32: raise ValueError('cache shape/dtype mismatch')
            arrays.append(a)
        return arrays, meta
    cache.mkdir(parents=True, exist_ok=False)
    edges = read_array(edge_path, np.float64)
    tic = time.perf_counter(); H1, H2, info = aggregate_features(X, edges)
    compute_time = time.perf_counter()-tic
    tic = time.perf_counter()
    np.save(cache/'hop1.npy', H1); np.save(cache/'hop2.npy', H2)
    save_time = time.perf_counter()-tic
    meta = {'signature': signature, 'graph': info, 'compute_seconds': compute_time,
            'save_seconds': save_time, 'array_sha256': {n: sha256(cache/(n+'.npy')) for n in ('hop1','hop2')},
            'setting': 'transductive: all node features and edges, no labels/years'}
    dump(cache/'metadata.json', meta)
    return [H1,H2], meta


def resample_ids(ids, y, floor, seed):
    """Add copies from fitting IDs only; original samples are always retained."""
    rng = np.random.RandomState(seed)
    pieces = [np.asarray(ids)]
    for label in np.unique(y[ids]):
        members = np.asarray(ids)[y[ids] == label]
        if len(members) < floor:
            pieces.append(rng.choice(members, size=floor-len(members), replace=True))
    return np.concatenate(pieces)


def make_input(X, hops, ids, order):
    blocks = [X[ids]] + [hops[i][ids] for i in range(order)]
    return np.hstack(blocks).astype(np.float32, copy=False)


def check_splits(parts, sampled, node_count):
    if any(v.ndim!=1 or len(v)==0 or not np.issubdtype(v.dtype,np.integer) or
           len(np.unique(v))!=len(v) or v.min()<0 or v.max()>=node_count for v in parts):
        raise ValueError('invalid split IDs')
    combined=np.concatenate(parts)
    if len(combined)!=len(sampled) or len(np.unique(combined))!=len(combined) or set(combined)!=set(sampled):
        raise ValueError('overlapping/inconsistent splits')


def scores(y_true, pred, labels, rare):
    return {'accuracy': float(accuracy_score(y_true,pred)),
            'macro_f1': float(f1_score(y_true,pred,labels=labels,average='macro',zero_division=0)),
            'weighted_f1': float(f1_score(y_true,pred,labels=labels,average='weighted',zero_division=0)),
            'rare10_macro_f1': float(f1_score(y_true,pred,labels=rare,average='macro',zero_division=0))}


def prepare(X,hops,y,tr,ev,order,ros,args):
    started = time.perf_counter()
    fit_ids = resample_ids(tr,y,args.ros_floor,args.seed) if ros else np.asarray(tr)
    labels, counts = np.unique(y[tr],return_counts=True)
    rare = labels[np.lexsort((labels,counts))[:10]]
    scaler = StandardScaler()
    A = scaler.fit_transform(make_input(X,hops,fit_ids,order))
    B = scaler.transform(make_input(X,hops,ev,order))
    return A,y[fit_ids],B,rare,fit_ids,time.perf_counter()-started


def validate_arm(out,name,order,ros,X,hops,y,tr,val,args,labels):
    A,yt,B,rare,fit_ids,prep_time = prepare(X,hops,y,tr,val,order,ros,args)
    model = MLPClassifier(random_state=args.seed,**MLP_PARAMS)
    best_key = (-1.0,-1.0); best_epoch=0; stale=0; train_seconds=0; predict_seconds=0
    best_metrics = None; caught_messages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for epoch in range(1,args.max_epochs+1):
            tic=time.perf_counter(); model.partial_fit(A,yt,classes=labels); train_seconds+=time.perf_counter()-tic
            tic=time.perf_counter(); pred=model.predict(B); predict_seconds+=time.perf_counter()-tic
            metrics=scores(y[val],pred,labels,rare)
            key=(metrics['macro_f1'],metrics['accuracy'])
            if key > best_key:
                best_key=key; best_epoch=epoch; best_metrics=metrics; stale=0
            else: stale+=1
            record(out,{'stage':'learning_curve','name':name,'epoch':epoch,'loss':float(model.loss_),
                        'metrics':metrics,'status':'ok','seed':args.seed})
            if stale>=args.patience: break
        caught_messages=[str(w.message) for w in caught]
    r={'stage':'validation','name':name,'status':'ok','seed':args.seed,'order':order,'resampling':ros,
       'best_epoch':best_epoch,'epochs_run':epoch,'metrics':best_metrics,'prep_seconds':prep_time,
       'fit_seconds':train_seconds,'predict_seconds':predict_seconds,'n_train_original':len(tr),
       'n_train_fit':len(fit_ids),'rare10_labels':rare.tolist(),'warnings':caught_messages,
       'stop_reason':'patience' if stale>=args.patience else 'max_epochs',
       'hit_epoch_limit':epoch==args.max_epochs}
    record(out,r)
    return r


def final_arm(out,selection,X,hops,y,tr,test,args,labels,predictions,class_rows):
    name=selection['name']; order=selection['order']; ros=selection['resampling']
    A,yt,B,rare,fit_ids,prep_time=prepare(X,hops,y,tr,test,order,ros,args)
    model=MLPClassifier(random_state=args.seed,**MLP_PARAMS)
    tic=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(selection['best_epoch']): model.partial_fit(A,yt,classes=labels)
    fit_time=time.perf_counter()-tic
    tic=time.perf_counter(); pred=model.predict(B); pred_time=time.perf_counter()-tic
    predictions[name]=pred
    precision,recall,f1,support=precision_recall_fscore_support(y[test],pred,labels=labels,zero_division=0)
    for i,label in enumerate(labels):
        class_rows.append([name,int(label),int(np.count_nonzero(y[tr]==label)),int(np.count_nonzero(yt==label)),
                           int(support[i]),float(precision[i]),float(recall[i]),float(f1[i]),bool(label in rare)])
    r={'stage':'test','name':name,'status':'ok','seed':args.seed,'order':order,'resampling':ros,
       'epochs':selection['best_epoch'],'metrics':scores(y[test],pred,labels,rare),
       'prep_seconds':prep_time,'fit_seconds':fit_time,'predict_seconds':pred_time,
       'n_train_original':len(tr),'n_train_fit':len(fit_ids),'n_test':len(test),
       'rare10_labels':rare.tolist(),'warnings':[str(w.message) for w in caught]}
    record(out,r)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path,help='Completed v2 run with fixed splits')
    p.add_argument('--cache',required=True,type=Path,help='Shared label-free graph feature cache')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--max-epochs',type=int,default=200);p.add_argument('--patience',type=int,default=20)
    p.add_argument('--ros-floor',type=int,default=200)
    a=p.parse_args()
    if min(a.max_epochs,a.patience,a.ros_floor)<1:p.error('epoch, patience and floor must be positive')
    source=json.loads((a.run/'metadata.json').read_text(encoding='utf-8-sig'))
    a.seed=source['arguments']['seed']
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        meta={'protocol':'mlp-external-validation-v1','source_run':str(a.run.resolve()),'seed':a.seed,
              'arms':ARMS,'mlp_params':MLP_PARAMS,'max_epochs':a.max_epochs,'patience':a.patience,'ros_floor':a.ros_floor,
              'selection':'per-arm epoch: validation macro-F1 then accuracy; model: same criterion, ties in declared arm order',
              'setting':'raw is feature-only; hop1/hop2 use full unlabeled graph: transductive',
              'test_history':'these test splits were evaluated by earlier tree experiments; this is a predeclared follow-up, not a fresh blind test',
              'files':source['files'],'split_sha256':sha256(a.run/'split_indices.npz'),
              'source_sha256':sha256(Path(__file__)),'numpy':np.__version__,'scipy':scipy.__version__,
              'sklearn':sklearn.__version__,'threadpools':threadpool_info(),
              'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}}
        dump(a.out/'protocol.json',meta)  # Freeze all arms and selection rules before evaluation.
        for key in ('features','labels','edges'):
            if sha256(Path(source['files'][key]['path'])) != source['files'][key]['sha256']:raise ValueError('source data mismatch: '+key)
        X=read_array(Path(source['files']['features']['path']),np.float32)
        yf=read_array(Path(source['files']['labels']['path']),np.float64)
        if yf.shape!=(len(X),1) or not np.isfinite(X).all() or not np.isfinite(yf).all() or not np.equal(yf,np.floor(yf)).all():raise ValueError('invalid features or labels')
        y=yf[:,0].astype(np.int64);labels=np.unique(y)
        with np.load(a.run/'split_indices.npz',allow_pickle=False) as z:
            tr,val,test,sampled=[z[k].copy() for k in ('train','validation','test','sampled')]
        check_splits([tr,val,test],sampled,len(X))
        if any(not np.array_equal(np.unique(y[v]),labels) for v in parts):raise ValueError('every split must cover all classes')
        np.savez_compressed(a.out/'split_indices.npz',train=tr,validation=val,test=test,sampled=sampled)
        tic=time.perf_counter();hops,cache_meta=graph_cache(a.cache,X,Path(source['files']['edges']['path']),source['files']['features']['sha256'])
        meta['cache_access_seconds']=time.perf_counter()-tic;meta['graph_cache']=cache_meta
        dump(a.out/'protocol.json',meta)
        start=time.perf_counter()
        selections=[validate_arm(a.out,name,order,ros,X,hops,y,tr,val,a,labels) for name,order,ros in ARMS]
        chosen=max(selections,key=lambda r:(r['metrics']['macro_f1'],r['metrics']['accuracy']))
        record(a.out,{'stage':'selection','status':'ok','seed':a.seed,'selected':chosen['name'],
                      'criterion':'validation macro_f1, then accuracy','search_seconds':time.perf_counter()-start})
        # Test labels are accessed for scoring only after every validation choice is fixed.
        predictions={'node_ids':test,'truth':y[test]};class_rows=[]
        final=[final_arm(a.out,s,X,hops,y,np.concatenate([tr,val]),test,a,labels,predictions,class_rows) for s in selections]
        np.savez_compressed(a.out/'test_predictions.npz',**predictions)
        with (a.out/'per_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['name','label','original_train_count','fitted_train_count','test_count','precision','recall','f1','rare10']);w.writerows(class_rows)
        dump(a.out/'final_summary.json',{'validation':selections,'selected':chosen['name'],'test':final})
    except Exception:
        record(a.out,{'stage':'run','status':'error','error':traceback.format_exc()})
        raise


if __name__=='__main__':main()
