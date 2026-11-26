import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, auc as sk_auc, average_precision_score,
                             f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R)
from dse_model import VLM_CODE
sys.path.insert(0, VLM_CODE)

from unified_dataset_v2 import (UnifiedDtiDatasetV2,
                                unified_collate_fn_v2)
from dse_head import (BIND, NOBIND, UNKNOWN,
                      dse_loss)
from dse_model import DSENet, readout

PROTOCOL = dict(
    datasets=['Patent', 'celegans', 'human', 'Article'],
    splits=['cluster', 'cold_smiles', 'cold_protein', 'cold_both', 'random'],
    seeds=[42, 1234, 2025, 12345, 666],
    batch_size=16, accum_steps=8, gnn_hidden=128, k=5,
    lr=5e-5, weight_decay=1e-5, max_epochs=30, warmup_epochs=5,
    grad_clip=5.0, plateau_factor=0.5, plateau_patience=10, plateau_min_lr=1e-6,
    fusion_mode='VLM-XA', vlm_ablation='no_attn', cl_weight=1e-4,
    decision_threshold=0.5,
)

HEAD = dict(lam=0.1, m_min=1e-2, m_max=1e3, knn=8, bank_size=4096,
            head_hidden=64, dropout=0.1, gate_refit_batches=200)

def find_data_root(candidate=None):
    candidate = candidate or os.environ.get('DTI_DATA') or os.path.join(R, 'data')
    resolved = os.path.abspath(os.path.expanduser(candidate))
    if not os.path.isdir(resolved):
        raise FileNotFoundError('Prepared data root not found: ' + resolved)
    return resolved

DATA_DIR = os.environ.get('DTI_DATA', os.path.join(R, 'data'))

def set_all_seeds(seed):

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_data_paths(dataset, split):
    base = os.path.join(DATA_DIR, dataset, split)
    if split == 'random':
        return ([os.path.join(base, 'train', 'samples.csv')],
                os.path.join(base, 'valid', 'samples.csv'),
                os.path.join(base, 'test', 'samples.csv'))
    train_paths = [os.path.join(base, 'target_train.csv')]
    src = os.path.join(base, 'source_train.csv')
    if os.path.exists(src):
        train_paths.append(src)
    return (train_paths, os.path.join(base, 'target_valid.csv'),
            os.path.join(base, 'target_test.csv'))

def _normalize_columns(df):
    rename, drop = {}, [c for c in df.columns if c.lower().startswith('unnamed')]
    for col in df.columns:
        cl = col.strip().lower()
        if cl == 'smiles':
            rename[col] = 'SMILES'
        elif cl == 'protein':
            rename[col] = 'Protein'
        elif cl in ('label', 'y'):
            rename[col] = 'Y'
    if rename:
        df = df.rename(columns=rename)
    return df.drop(columns=drop) if drop else df

def load_dataframes(train_paths, valid_path, test_path):
    dfs = [_normalize_columns(pd.read_csv(p)) for p in train_paths if os.path.exists(p)]
    if not dfs:
        raise ValueError('no train files found')
    return (pd.concat(dfs).reset_index(drop=True),
            _normalize_columns(pd.read_csv(valid_path)),
            _normalize_columns(pd.read_csv(test_path)))

def build_datasets(dataset, tr_df, vl_df, ts_df):
    mod = os.path.join(DATA_DIR, dataset, 'Modalities')
    args = (os.path.join(mod, 'smiles_all.npy'), os.path.join(mod, 'p_LM.pkl'),
            os.path.join(mod, 'd_LM.pkl'),
            os.path.join(DATA_DIR, dataset, 'sequence_to_pdb.json'),
            os.path.join(DATA_DIR, dataset, 'CompoundGraph'),
            os.path.join(DATA_DIR, dataset, 'ProteinGraph', 'package_bin'))
    return [UnifiedDtiDatasetV2(df, *args, load_llm=True) for df in (tr_df, vl_df, ts_df)]

@torch.no_grad()
def infer(model, loader, device):

    model.eval()
    cols = {k: [] for k in ('p1', 'u', 'm', 'p_unknown', 'gate', 'dist',
                            'pi_bind', 'pi_nobind', 'pi_unknown', 'y')}
    for bg_d, bg_p, df_, pf_, labels, dl_, pl_ in loader:
        o = model(bg_d.to(device), bg_p.to(device), df_.to(device),
                  pf_.to(device), dl_.to(device), pl_.to(device))
        r = readout(o['pi'], o['m'])
        cols['p1'].append(r['p1'].cpu().numpy())
        cols['u'].append(r['u'].cpu().numpy())
        cols['m'].append(r['m'].cpu().numpy())
        cols['p_unknown'].append(r['p_unknown'].cpu().numpy())
        cols['gate'].append(o['gate'].cpu().numpy())
        cols['dist'].append(model.gate.distance(o['h']).cpu().numpy())

        for name, j in (('pi_nobind', NOBIND), ('pi_bind', BIND),
                        ('pi_unknown', UNKNOWN)):
            cols[name].append(o['pi'][:, j].cpu().numpy() if j < o['pi'].shape[1]
                              else np.zeros(o['pi'].shape[0], dtype=np.float32))
        cols['y'].append(labels.numpy())
    return {k: np.concatenate(v) for k, v in cols.items()}

def aurc(confidence, wrong):

    if len(wrong) == 0:
        return float('nan')
    order = np.argsort(-confidence, kind='stable')
    risk = np.cumsum(wrong[order]) / np.arange(1, len(wrong) + 1)
    return float(risk.mean())

def metrics_from_dump(d):
    y, p1, u = d['y'].astype(int), d['p1'], d['u']
    pred = (p1 >= PROTOCOL['decision_threshold']).astype(int)
    if len(np.unique(y)) < 2:
        return None
    pr, rc, _ = precision_recall_curve(y, p1)
    wrong = (pred != y).astype(float)
    conf = np.abs(p1 - 0.5)
    out = dict(
        AUC=roc_auc_score(y, p1), PRAUC=sk_auc(rc, pr),
        AUPRC=average_precision_score(y, p1),
        Precision=precision_score(y, pred, zero_division=0),
        Recall=recall_score(y, pred, zero_division=0),
        Accuracy=accuracy_score(y, pred),
        F1=f1_score(y, pred, zero_division=0), MCC=matthews_corrcoef(y, pred),
        U_Median=float(np.median(u)), Unknown_Mean=float(d['p_unknown'].mean()),
        Gate_Median=float(np.median(d['gate'])),

        AURC_u=aurc(-u, wrong), AURC_conf=aurc(conf, wrong))
    out['AURC_gain'] = out['AURC_conf'] - out['AURC_u']

    out['Degenerate'] = float(len(np.unique(pred)) < 2)
    for nm, s in (('ErrDet_u', u), ('ErrDet_conf', -conf)):
        out[nm] = float(roc_auc_score(wrong, s)) if wrong.sum() not in (0, len(wrong)) else float('nan')
    return {k: float(v) for k, v in out.items()}

def run_one(args, device, dataset, split, seed, out_root):
    set_all_seeds(seed)
    tag = '%s/%s/s%d' % (dataset, split, seed)
    mdir = os.path.join(out_root, 'models', dataset, split, 'seed_%d' % seed)
    ddir = os.path.join(out_root, 'dumps')
    os.makedirs(mdir, exist_ok=True)
    os.makedirs(ddir, exist_ok=True)

    tr_df, vl_df, ts_df = load_dataframes(*get_data_paths(dataset, split))
    from predict import verify_inputs
    verify_inputs(pd.concat([tr_df, vl_df, ts_df], ignore_index=True),
                  os.path.join(DATA_DIR, dataset))
    tr_ds, vl_ds, ts_ds = build_datasets(dataset, tr_df, vl_df, ts_df)
    print('  data: train=%d valid=%d test=%d' % (len(tr_ds), len(vl_ds), len(ts_ds)))

    bs, nw = PROTOCOL['batch_size'], args.num_workers
    mk = lambda ds, sh, w: DataLoader(ds, batch_size=bs, shuffle=sh,
                                      collate_fn=unified_collate_fn_v2,
                                      num_workers=w, pin_memory=True,
                                      prefetch_factor=(2 if w else None))
    tl, vl, tsl = mk(tr_ds, True, nw), mk(vl_ds, False, max(nw // 2, 0)), mk(ts_ds, False, max(nw // 2, 0))

    model = DSENet(gnn_hidden=PROTOCOL['gnn_hidden'], k=PROTOCOL['k'],
                   fusion_mode=PROTOCOL['fusion_mode'],
                   vlm_ablation=PROTOCOL['vlm_ablation'],
                   m_min=HEAD['m_min'], m_max=HEAD['m_max'],
                   bank_size=HEAD['bank_size'], knn=HEAD['knn'],
                   head_hidden=HEAD['head_hidden'], dropout=HEAD['dropout']).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=PROTOCOL['lr'],
                           weight_decay=PROTOCOL['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=PROTOCOL['plateau_factor'],
        patience=PROTOCOL['plateau_patience'], min_lr=PROTOCOL['plateau_min_lr'])

    json.dump(dict(tag=tag, dataset=dataset, split=split, seed=seed,
                   protocol=PROTOCOL, head=HEAD, n_params=int(n_params),
                   n_train=len(tr_ds), n_valid=len(vl_ds), n_test=len(ts_ds),
                   gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'),
                   host=platform.node(), torch=torch.__version__,
                   started=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(os.path.join(mdir, 'manifest.json'), 'w'), indent=2, default=str)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    epochs = 2 if args.smoke else PROTOCOL['max_epochs']
    best_val, best_ep, best_dump = -1.0, -1, None
    best_valid, best_valid_dump = None, None
    hist, t0 = [], time.time()
    opt.zero_grad(set_to_none=True)

    for ep in range(1, epochs + 1):
        if ep <= PROTOCOL['warmup_epochs']:
            for g in opt.param_groups:
                g['lr'] = PROTOCOL['lr'] * ep / PROTOCOL['warmup_epochs']
        model.train()
        run_loss, n_it = 0.0, 0
        bar = tqdm(tl, desc='  %s E%02d/%d' % (tag, ep, epochs), ncols=100,
                   leave=False, unit='b')
        for i, (bg_d, bg_p, df_, pf_, labels, dl_, pl_) in enumerate(bar):
            o = model(bg_d.to(device), bg_p.to(device), df_.to(device),
                      pf_.to(device), dl_.to(device), pl_.to(device))
            loss = dse_loss(o['alpha'], labels.to(device), lam=HEAD['lam'])
            loss = (loss + PROTOCOL['cl_weight'] * o['cl']) / PROTOCOL['accum_steps']
            loss.backward()
            run_loss += loss.item() * PROTOCOL['accum_steps']
            n_it += 1
            if (i + 1) % PROTOCOL['accum_steps'] == 0 or (i + 1) == len(tl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['grad_clip'])
                opt.step()
                opt.zero_grad(set_to_none=True)
            bar.set_postfix_str('loss %.4f' % (run_loss / n_it))

        model.refit_gate(tl, device, max_batches=HEAD['gate_refit_batches'])

        vd = infer(model, vl, device)
        vm = metrics_from_dump(vd)
        val_auc = vm['AUC'] if vm else float('nan')
        if ep > PROTOCOL['warmup_epochs'] and np.isfinite(val_auc):
            sched.step(val_auc)
        ep_loss = run_loss / max(n_it, 1)
        hist.append(dict(epoch=ep, train_loss=ep_loss,
                         val_auc=val_auc, val_u=vm['U_Median'] if vm else np.nan))

        recent = [h['train_loss'] for h in hist[-5:]]
        if len(recent) == 5 and max(recent) - min(recent) < 1e-3:
            print('  !! train_loss has not moved in 5 epochs (%.4f); the model '
                  'is stuck, not converged -- stop and diagnose' % ep_loss)

        if vm and vm['Degenerate']:
            print('  !! validation predictions are all one class -- thresholded '
                  'metrics (ACC/F1/MCC/Precision) are meaningless this epoch')
        print('  %s E%02d loss=%.4f valAUC=%.4f valU=%.4f%s'
              % (tag, ep, run_loss / max(n_it, 1), val_auc,
                 vm['U_Median'] if vm else np.nan,
                 '   * best' if np.isfinite(val_auc) and val_auc > best_val else ''))

        if np.isfinite(val_auc) and val_auc > best_val:
            best_val, best_ep = val_auc, ep
            best_valid = vm
            best_valid_dump = vd
            torch.save(model.state_dict(), os.path.join(mdir, 'best_model.pth'))

            best_dump = infer(model, tsl, device)
        pd.DataFrame(hist).to_csv(os.path.join(mdir, 'epochs.csv'), index=False)

    if best_dump is None:
        raise RuntimeError('no epoch produced a finite validation AUC')

    dump_path = os.path.join(ddir, '%s_%s_seed%d_test.csv' % (dataset, split, seed))
    pd.DataFrame(best_dump).assign(Dataset=dataset, Split=split, Seed=seed,
                                   BestEpoch=best_ep).to_csv(dump_path, index=False,
                                                             float_format='%.17g')
    reloaded = pd.read_csv(dump_path)
    best_test = metrics_from_dump({k: reloaded[k].values for k in
                                   ('y', 'p1', 'u', 'p_unknown', 'gate')})
    mem = metrics_from_dump(best_dump)
    drift = max(abs(mem[k] - best_test[k]) for k in best_test
                if np.isfinite(best_test[k]) and np.isfinite(mem[k]))

    print('  %s  in-memory vs written dump: max|delta| = %.2e  %s'
          % (tag, drift, 'OK' if drift < 1e-3 else '<-- INVESTIGATE'))

    peak_a = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == 'cuda' else 0.0
    peak_r = torch.cuda.max_memory_reserved() / 1024 ** 3 if device.type == 'cuda' else 0.0
    print('  %s  GPU peak: %.2f GB needed, %.2f GB reserved' % (tag, peak_a, peak_r))
    row = dict(Dataset=dataset, Split=split, Seed=seed, BestEpoch=best_ep,
               Valid_AUC=best_val, Num_Params=int(n_params),
               Training_Time=time.time() - t0, DumpDrift=drift,
               PeakGPU_Alloc_GB=peak_a, PeakGPU_Reserved_GB=peak_r,
               DumpPath=dump_path)
    row.update({'%s_Test' % k: v for k, v in best_test.items()})

    row.update({'%s_Valid' % k: v for k, v in (best_valid or {}).items()})
    pd.DataFrame(best_valid_dump).to_csv(
        os.path.join(ddir, '%s_%s_seed%d_valid.csv' % (dataset, split, seed)),
        index=False, float_format='%.17g')
    return row

def append_row(csv_path, row):

    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as f:
            rd = csv.reader(f)
            old = next(rd, None)
            rest = list(rd)
        if old and set(row.keys()) - set(old):
            cols = old + [c for c in row.keys() if c not in old]
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(cols)
                for r in rest:
                    w.writerow(r + [''] * (len(cols) - len(r)))
            print('  note: results.csv gained %d column(s); header rewritten'
                  % (len(cols) - len(old)))
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        cols = list(row.keys())
        if exists:
            with open(csv_path, newline='', encoding='utf-8') as g:
                cols = next(csv.reader(g), cols)
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerow(row)

def main():
    global DATA_DIR
    ap = argparse.ArgumentParser(description='Train the three-component DSE-DTI model')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--datasets', default=','.join(PROTOCOL['datasets']))
    ap.add_argument('--splits', default=','.join(PROTOCOL['splits']))
    ap.add_argument('--seeds', default=','.join(map(str, PROTOCOL['seeds'])))
    ap.add_argument('--out', default=os.path.join(R, 'runs'))
    ap.add_argument('--device', default='auto')
    ap.add_argument('--num-workers', '--num_workers', dest='num_workers', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    DATA_DIR = find_data_root(args.data_root)
    datasets = [x.strip() for x in args.datasets.split(',') if x.strip()]
    splits = [x.strip() for x in args.splits.split(',') if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    if not datasets or set(datasets) - set(PROTOCOL['datasets']):
        ap.error('Use dataset names from: ' + ', '.join(PROTOCOL['datasets']))
    if not splits or set(splits) - set(PROTOCOL['splits']):
        ap.error('Use split keys from: ' + ', '.join(PROTOCOL['splits']))
    if not seeds or args.num_workers < 0:
        ap.error('At least one seed and a non-negative worker count are required')
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA was requested but is unavailable')
    out_root = os.path.abspath(args.out)
    if args.smoke:
        out_root = os.path.join(out_root, 'smoke')
    os.makedirs(out_root, exist_ok=True)
    live = os.path.join(out_root, 'results.csv')
    schedule = [(d, s, seed) for d in datasets for s in splits for seed in seeds]
    if args.smoke:
        schedule = schedule[:1]
    done = set()
    if os.path.exists(live):
        with open(live, newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                done.add((row['Dataset'], row['Split'], int(row['Seed'])))
    print('Device:', device)
    print('Prepared data:', DATA_DIR)
    print('Output:', out_root)
    print('Tasks:', len(schedule))
    for dataset, split, seed in schedule:
        if (dataset, split, seed) in done:
            print('Skipping completed run:', dataset, split, seed)
            continue
        row = run_one(args, device, dataset, split, seed, out_root)
        append_row(live, row)
        print(dataset, split, seed, 'test AUROC:', row['AUC_Test'])
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
