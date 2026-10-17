#!/usr/bin/env python
# coding: utf-8
# copied from ~/analysis/run12/data/unfolding/good/final_0628/multifold.py
# improved by ChatGPT

import sys
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import random
import numpy as np
import pandas as pd
import uproot
import tensorrt
from energyflow import archs
import tensorflow as tf

# set seed
if len(sys.argv) < 2:
    raise SystemExit(f"Usage: {sys.argv[0]} SEED")

print('Number of arguments:', len(sys.argv), 'arguments')
try:
    seed = int(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"SEED must be an integer; received {sys.argv[1]!r}") from exc
print('Seed:', seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def applyCut(inputDataframe, cut, text=None):
    nbeforecut = inputDataframe.shape[0]
    cutDataframe = inputDataframe.query(cut).copy()
    if text is not None:
        print(f"{text}: kept {len(cutDataframe)} / {nbeforecut} rows using cut: {cut}")
    return cutDataframe


def getData(observables, filename, treename):
    with uproot.open(filename) as root_file:
        if treename not in root_file:
            raise KeyError(f"Tree {treename!r} was not found in {filename!r}")
        return root_file[treename].arrays(observables, library='pd')


def reweight(X, Y, w, model, filepath, fitargs, val_data=None):
    # Keep weight-clipping options out of model.fit(), which does not accept them.
    fit_kwargs = {
        key: value for key, value in fitargs.items()
        if key not in ('weight_clip_min', 'weight_clip_max')
    }
    val_dict = {'validation_data': val_data} if val_data is not None else {}
    history = model.fit(X, Y, sample_weight=w, **fit_kwargs, **val_dict)

    # Keras 3 requires weight-only checkpoints to end in '.weights.h5'.
    if filepath is not None:
        model.save_weights(filepath)

    # Avoid very large prediction batches when the training batch size is large.
    predict_batch_size = min(10 * fitargs.get('batch_size', 500), 8192)
    preds = np.asarray(
        model.predict(X, batch_size=predict_batch_size, verbose=0), dtype=float
    ).reshape(-1)

    # Concatenate validation predictions into training predictions, in the same order.
    if val_data is not None:
        preds_val = np.asarray(
            model.predict(val_data[0], batch_size=predict_batch_size, verbose=0), dtype=float
        ).reshape(-1)
        preds = np.concatenate((preds, preds_val))
        w = np.concatenate((w, val_data[2]))
    else:
        w = np.asarray(w, dtype=float).copy()

    if len(preds) != len(w):
        raise ValueError(
            f"Prediction/weight length mismatch: {len(preds)} predictions for {len(w)} weights"
        )
    if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(w)):
        raise ValueError("Non-finite predictions or weights encountered during reweighting")
    if np.any((preds < 0) | (preds > 1)):
        raise ValueError("Classifier predictions must lie between 0 and 1")

    weight_clip_min = fitargs.get('weight_clip_min', 0.)
    weight_clip_max = fitargs.get('weight_clip_max', np.inf)
    if weight_clip_min > weight_clip_max:
        raise ValueError("weight_clip_min must not exceed weight_clip_max")

    odds = preds / (1. - preds + 1.e-50)
    w *= np.clip(odds, weight_clip_min, weight_clip_max)
    if not np.all(np.isfinite(w)):
        raise ValueError("Non-finite reweighted values were produced")
    return w, history


def omnifold(X_gen_i, Y_gen_i, X_det_i, Y_det_i, wdata, winit, det_model, mc_model, fitargs1, fitargs2,
             val=0.2, it=10, weights_filename=None, trw_ind=0, delete_global_arrays=False):
    history_arr = []
    X_gen_arr = globals()[X_gen_i] if isinstance(X_gen_i, str) else X_gen_i
    Y_gen_arr = globals()[Y_gen_i] if isinstance(Y_gen_i, str) else Y_gen_i
    X_det_arr = globals()[X_det_i] if isinstance(X_det_i, str) else X_det_i
    Y_det_arr = globals()[Y_det_i] if isinstance(Y_det_i, str) else Y_det_i

    # Convert explicitly and check the assumptions made by the concatenations below.
    X_gen_arr = np.asarray(X_gen_arr)
    Y_gen_arr = np.asarray(Y_gen_arr).reshape(-1)
    X_det_arr = np.asarray(X_det_arr)
    Y_det_arr = np.asarray(Y_det_arr).reshape(-1)
    wdata = np.asarray(wdata, dtype=float).reshape(-1)
    winit = np.asarray(winit, dtype=float).reshape(-1)

    if not 0. < val < 1.:
        raise ValueError(f"val must be strictly between 0 and 1; received {val}")
    if it < 1:
        raise ValueError(f"it must be at least 1; received {it}")
    if len(wdata) == 0 or len(winit) == 0:
        raise ValueError("wdata and winit must both contain at least one event")
    if len(X_det_arr) != len(Y_det_arr) or len(X_det_arr) != len(wdata) + len(winit):
        raise ValueError("Detector arrays must match len(wdata) + len(winit)")
    if len(X_gen_arr) != len(Y_gen_arr) or len(X_gen_arr) != 2 * len(winit):
        raise ValueError("Generator arrays must contain two copies of each event in winit")
    if X_det_arr.ndim != 2 or X_gen_arr.ndim != 2:
        raise ValueError("X_det and X_gen must be two-dimensional feature arrays")
    if X_det_arr.shape[1] != X_gen_arr.shape[1]:
        raise ValueError("X_det and X_gen must have the same number of features")
    if (not np.all(np.isfinite(wdata))) or (not np.all(np.isfinite(winit))):
        raise ValueError("Input event weights must be finite")

    # The per-event and duplicated generator samples must be large enough to split.
    nval_det = int(val * (len(winit) + len(wdata)))
    nval = int(val * len(winit))
    if nval_det < 1 or nval_det >= len(X_det_arr):
        raise ValueError("The detector sample is too small for this validation fraction")
    if nval < 1 or nval >= len(winit):
        raise ValueError("The generator sample is too small for this validation fraction")

    ws = [winit]
    perm_det = np.random.permutation(len(winit) + len(wdata))
    invperm_det = np.argsort(perm_det)
    X_det_train, X_det_val = X_det_arr[perm_det[:-nval_det]], X_det_arr[perm_det[-nval_det:]]
    Y_det_train, Y_det_val = Y_det_arr[perm_det[:-nval_det]], Y_det_arr[perm_det[-nval_det:]]
    if delete_global_arrays:
        del X_det_arr, Y_det_arr
        if isinstance(X_det_i, str):
            del globals()[X_det_i]
        if isinstance(Y_det_i, str):
            del globals()[Y_det_i]

    baseperm0 = np.random.permutation(len(winit))
    baseperm1 = baseperm0 + len(winit)
    trainperm = np.concatenate((baseperm0[:-nval], baseperm1[:-nval]))
    valperm = np.concatenate((baseperm0[-nval:], baseperm1[-nval:]))
    np.random.shuffle(trainperm)
    np.random.shuffle(valperm)
    perm_gen = np.concatenate((trainperm, valperm))
    invperm_gen = np.argsort(perm_gen)
    # Use the actual validation partition length so duplicate truth copies stay together.
    nval_gen = len(valperm)
    X_gen_train, X_gen_val = X_gen_arr[perm_gen[:-nval_gen]], X_gen_arr[perm_gen[-nval_gen:]]
    Y_gen_train, Y_gen_val = Y_gen_arr[perm_gen[:-nval_gen]], Y_gen_arr[perm_gen[-nval_gen:]]
    if delete_global_arrays:
        del X_gen_arr, Y_gen_arr
        if isinstance(X_gen_i, str):
            del globals()[X_gen_i]
        if isinstance(Y_gen_i, str):
            del globals()[Y_gen_i]

    model_det_fp = det_model[1].get('filepath', None)
    model_mc_fp = mc_model[1].get('filepath', None)
    if it > 1 and (model_det_fp is None or model_mc_fp is None):
        raise ValueError("Model 'filepath' patterns are required when it > 1")

    for i in range(it):
        model_det_fp_i = model_det_fp.format(i) if model_det_fp is not None else None
        model_mc_fp_i = model_mc_fp.format(i) if model_mc_fp is not None else None
        if model_det_fp_i is not None:
            det_model[1]['filepath'] = model_det_fp_i + '_Epoch-{epoch}.weights.h5'
        if model_mc_fp_i is not None:
            mc_model[1]['filepath'] = model_mc_fp_i + '_Epoch-{epoch}.weights.h5'

        model_det = det_model[0](**det_model[1])
        model_mc = mc_model[0](**mc_model[1])
        if i > 0:
            model_det.load_weights(model_det_fp.format(i - 1) + '.weights.h5')
            model_mc.load_weights(model_mc_fp.format(i - 1) + '.weights.h5')

        # step 1: reweight sim to look like data
        w = np.concatenate((wdata, ws[-1]))
        w_train, w_val = w[perm_det[:-nval_det]], w[perm_det[-nval_det:]]
        det_weights_path = model_det_fp_i + '.weights.h5' if model_det_fp_i is not None else None
        rw1, history = reweight(
            X_det_train, Y_det_train, w_train, model_det, det_weights_path, fitargs1,
            val_data=(X_det_val, Y_det_val, w_val)
        )
        rw = rw1[invperm_det]
        history_arr.append(history)
        ws.append(rw[len(wdata):])

        # step 2: reweight the prior to the learned weighting
        w = np.concatenate((ws[-1], ws[trw_ind]))
        w_train, w_val = w[perm_gen[:-nval_gen]], w[perm_gen[-nval_gen:]]
        mc_weights_path = model_mc_fp_i + '.weights.h5' if model_mc_fp_i is not None else None
        rw1, history = reweight(
            X_gen_train, Y_gen_train, w_train, model_mc, mc_weights_path, fitargs2,
            val_data=(X_gen_val, Y_gen_val, w_val)
        )
        rw = rw1[invperm_gen]
        history_arr.append(history)
        ws.append(rw[len(ws[-1]):])

        # save the weights if specified
        if weights_filename is not None:
            np.save(weights_filename, np.asarray(ws))
    return ws, history_arr


# prepare dataframe for PYTHIA6+embedding
observables2 = ['pt', 'm', 'q', 'rg', 'zg', 'mg', 'w', 'mcevi']  # truth
observabless2 = ['pts', 'ms', 'qs', 'rgs', 'zgs', 'mgs']  # reco
inputFiles = ['/nfs/roberts/home/ys668/output/run12/all050124_rc_notrueptcons.root']
treeName = ["MatchedTree"]
df1 = applyCut(
    getData(observables2 + observabless2, inputFiles[0], treeName[0]),
    'pt>0 and pts>0 and w<5e-9'
)  # matched jets
# pt > 0 means not missed, pts > 0 means not fake, w < 5e-9 means pTHat > 9 GeV
if df1.empty:
    raise ValueError("No PYTHIA6+embedding rows survived the matched-jet selection")
if not np.all(np.isfinite(df1['w'])) or df1['w'].max() <= 0:
    raise ValueError("The selected PYTHIA event weights must be finite and have a positive maximum")
df1['relw'] = np.asarray(df1['w'], dtype=float) / df1['w'].max()  # normalize weights

# prepare dataframe for data
observables1 = ['pt', 'm', 'q', 'rg', 'zg', 'mg']
observabless1 = ['pts', 'ms', 'qs', 'rgs', 'zgs', 'mgs']
with uproot.open('/nfs/roberts/home/ys668/output/run12/data062824.root') as root_file:
    tree = root_file["ResultTree"].arrays(observables1, library='np')
# In the data ROOT file the branches are called pt, m, ..., but the reco
# embedding uses pts, ms, ..., so map the names while creating the DataFrame.
df2 = pd.DataFrame({
    output_name: np.concatenate(tree[input_name])
    for input_name, output_name in zip(observables1, observabless1)
})
if df2.empty:
    raise ValueError("The data ROOT tree contains no rows")

# reweight data according to (1-fake rate)
ptedges = [15, 20, 25, 30, 40, 65]
scaling = [0.97, 0.98, 0.99, 0.99, 0.99]  # new for 20235003, same for notrueptcons
if len(scaling) != len(ptedges) - 1:
    raise ValueError("scaling must have one value for each pT bin")
pt_values = df2['pts'].to_numpy()
w2 = np.ones(df2.shape[0], dtype=float)
for j in range(len(ptedges) - 1):
    # Preserve the original strict bin boundaries: values exactly at an edge are unchanged.
    in_bin = (ptedges[j] < pt_values) & (pt_values < ptedges[j + 1])
    w2[in_bin] *= scaling[j]
df2['w'] = w2

# set up dictionary and arrays for multifold
datasets = {'pythia': df1, 'data': df2}
obs_multifold = ['pt', 'm', 'q', 'rg', 'zg', 'mg']
obs = {}
obs.setdefault('pt', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['pt' + s]),
})
obs.setdefault('q', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['q' + s]),
})
obs.setdefault('rg', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['rg' + s]),
})
obs.setdefault('zg', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['zg' + s]),
})
obs.setdefault('m', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['m' + s]),
})
obs.setdefault('mg', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['mg' + s]),
})

for obkey, ob in obs.items():
    ob['genobs'], ob['simobs'] = ob['func']('pythia', ''), ob['func']('pythia', 's')
    ob['dataobs'] = ob['func']('data', 's')
    print('Done with', obkey)

# concatenate data (label Y = 1) and MC reco (label Y = 0). matrix shape is (N_data + N_MC, 6).
X_det = np.asarray([
    np.concatenate((obs[obkey]['dataobs'], obs[obkey]['simobs']))
    for obkey in obs_multifold
]).T
Y_det = np.concatenate((np.ones(len(obs['pt']['dataobs'])), np.zeros(len(obs['pt']['simobs']))))
# concatenate MC truth (label Y = 1) and duplicated MC truth (label Y = 0).
X_gen = np.asarray([
    np.concatenate((obs[obkey]['genobs'], obs[obkey]['genobs']))
    for obkey in obs_multifold
]).T
Y_gen = np.concatenate((np.ones(len(obs['pt']['genobs'])), np.zeros(len(obs['pt']['genobs']))))

# Check the inputs before standardization; don't silently divide by zero for a constant observable.
if not np.all(np.isfinite(X_det)) or not np.all(np.isfinite(X_gen)):
    raise ValueError("Feature arrays contain NaN or infinite values")
det_mean, det_std = np.mean(X_det, axis=0), np.std(X_det, axis=0)
gen_mean, gen_std = np.mean(X_gen, axis=0), np.std(X_gen, axis=0)
if np.any(det_std == 0) or np.any(gen_std == 0):
    raise ValueError("At least one feature has zero standard deviation and cannot be standardized")
X_det = (X_det - det_mean) / det_std
X_gen = (X_gen - gen_mean) / gen_std

# normalize such that total data weight = total MC weight
sum_data_weights = np.sum(df2['w'])
sum_mc_weights = np.sum(df1['relw'])
if (not np.isfinite(sum_data_weights)) or (not np.isfinite(sum_mc_weights)) or sum_data_weights <= 0 or sum_mc_weights <= 0:
    raise ValueError("The total data and MC weights must be finite and positive")
norm = sum_data_weights / sum_mc_weights
wdata = df2['w'].to_numpy(dtype=float)
winit = norm * df1['relw'].to_numpy(dtype=float)

# set up multifold parameters
itnum = 8
valnum = 0.5
batch_size = 50000  # changed on 060425
model_layer_sizes = [50, 100, 50]
det_args = {'input_dim': len(obs_multifold), 'dense_sizes': model_layer_sizes,
            'patience': 50, 'filepath': 'Step1_{}', 'save_weights_only': True,
            'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}
mc_args = {'input_dim': len(obs_multifold), 'dense_sizes': model_layer_sizes,
           'patience': 50, 'filepath': 'Step2_{}', 'save_weights_only': True,
           'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}
fitargs1 = {'batch_size': batch_size, 'epochs': 100, 'verbose': 1}
fitargs2 = {'batch_size': batch_size, 'epochs': 100, 'verbose': 1}

# multifold
print('Available GPUs:', tf.config.list_physical_devices('GPU'))
multifold_ws, history = omnifold(
    X_gen, Y_gen, X_det, Y_det, wdata, winit,
    (archs.DNN, det_args), (archs.DNN, mc_args), fitargs1, fitargs2,
    val=valnum, it=itnum, trw_ind=-2, weights_filename=str(seed)
)

# save results
path = '/nfs/roberts/scratch/pi_hlc7/ys668/test_101326/'
os.makedirs(path, exist_ok=True)

baseline_path = os.path.join(path, 'jet_noweight.csv')
if not os.path.isfile(baseline_path):
    df3 = df1[obs_multifold]
    df3.to_csv(baseline_path, index=False)

for i in range(itnum):
    iteration_weights = np.asarray(multifold_ws[2 * (i + 1)], dtype=float)
    weight_sum = np.sum(iteration_weights)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError(f"Invalid total unfolding weight for iteration {i + 1}: {weight_sum}")
    df = pd.DataFrame(data=iteration_weights / weight_sum, columns=['weight'])
    output_file = os.path.join(path, f'it{i + 1}_seed{seed}.csv')
    df.to_csv(output_file, index=False)
