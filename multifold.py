#!/usr/bin/env python
# coding: utf-8

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
print('Number of arguments:', len(sys.argv), 'arguments')
print('Seed:', sys.argv[1])
seed = (int)(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def applyCut(inputDataframe, cut, text=None):
    dataframe = inputDataframe
    nbeforecut = dataframe.shape[0]
    cutDataframe = dataframe.query(cut)
    return cutDataframe

def getData(observables, filename, treename):
    return (uproot.open(filename)[treename]).arrays(observables, library='pd')

def reweight(X, Y, w, model, filepath, fitargs, val_data=None):
    val_dict = {'validation_data': val_data} if val_data is not None else {}
    history = model.fit(X, Y, sample_weight=w, **fitargs, **val_dict)
    model.save_weights(filepath)
    preds = model.predict(X, batch_size=10*fitargs.get('batch_size', 500))[:,0]  
    # concatenate validation predictions into training predictions
    if val_data is not None:
        preds_val = model.predict(val_data[0], batch_size=10*fitargs.get('batch_size', 500))[:,0]
        preds = np.concatenate((preds, preds_val))
        w = np.concatenate((w, val_data[2]))
    w *= np.clip(preds/(1 - preds + 10**-50), fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
    return w, history

def omnifold(X_gen_i, Y_gen_i, X_det_i, Y_det_i, wdata, winit, det_model, mc_model, fitargs1, fitargs2,
             val=0.2, it=10, weights_filename=None, trw_ind=0, delete_global_arrays=False):
    history_arr = []
    X_gen_arr = globals()[X_gen_i] if isinstance(X_gen_i, str) else X_gen_i
    Y_gen_arr = globals()[Y_gen_i] if isinstance(Y_gen_i, str) else Y_gen_i
    X_det_arr = globals()[X_det_i] if isinstance(X_det_i, str) else X_det_i
    Y_det_arr = globals()[Y_det_i] if isinstance(Y_det_i, str) else Y_det_i
    ws = [winit]
    perm_det = np.random.permutation(len(winit) + len(wdata))
    invperm_det = np.argsort(perm_det)
    nval_det = int(val*len(perm_det))
    X_det_train, X_det_val = X_det_arr[perm_det[:-nval_det]], X_det_arr[perm_det[-nval_det:]]
    Y_det_train, Y_det_val = Y_det_arr[perm_det[:-nval_det]], Y_det_arr[perm_det[-nval_det:]]
    if delete_global_arrays:
        del X_det_arr, Y_det_arr
        if isinstance(X_det_i, str):
            del globals()[X_det_i]
        if isinstance(Y_det_i, str):
            del globals()[Y_det_i]
    nval = int(val*len(winit))
    baseperm0 = np.random.permutation(len(winit))
    baseperm1 = baseperm0 + len(winit)
    trainperm = np.concatenate((baseperm0[:-nval], baseperm1[:-nval]))
    valperm = np.concatenate((baseperm0[-nval:], baseperm1[-nval:]))
    np.random.shuffle(trainperm)
    np.random.shuffle(valperm)
    perm_gen = np.concatenate((trainperm, valperm))
    invperm_gen = np.argsort(perm_gen)
    nval_gen = int(val*len(perm_gen))
    X_gen_train, X_gen_val = X_gen_arr[perm_gen[:-nval_gen]], X_gen_arr[perm_gen[-nval_gen:]]
    Y_gen_train, Y_gen_val = Y_gen_arr[perm_gen[:-nval_gen]], Y_gen_arr[perm_gen[-nval_gen:]]
    if delete_global_arrays:
        del X_gen_arr, Y_gen_arr
        if isinstance(X_gen_i, str):
            del globals()[X_gen_i]
        if isinstance(Y_gen_i, str):
            del globals()[Y_gen_i]
    model_det_fp, model_mc_fp = det_model[1].get('filepath', None), mc_model[1].get('filepath', None)
    for i in range(it):
        if model_det_fp is not None:
            model_det_fp_i = model_det_fp.format(i)
            det_model[1]['filepath'] = model_det_fp_i + '_Epoch-{epoch}.weights.h5'
        if model_mc_fp is not None:
            model_mc_fp_i = model_mc_fp.format(i)
            mc_model[1]['filepath'] = model_mc_fp_i + '_Epoch-{epoch}.weights.h5'
        model_det = det_model[0](**det_model[1])
        model_mc = mc_model[0](**mc_model[1])
        if i > 0:
            model_det.load_weights(model_det_fp.format(i-1))
            model_mc.load_weights(model_mc_fp.format(i-1))
        # step 1: reweight sim to look like data
        w = np.concatenate((wdata, ws[-1]))
        w_train, w_val = w[perm_det[:-nval_det]], w[perm_det[-nval_det:]]
        rw1,history = reweight(X_det_train, Y_det_train, w_train, model_det, model_det_fp_i, fitargs1, val_data=(X_det_val, Y_det_val, w_val))
        rw = rw1[invperm_det]
        history_arr.append(history)
        ws.append(rw[len(wdata):])
        # step 2: reweight the prior to the learned weighting
        w = np.concatenate((ws[-1], ws[trw_ind]))
        w_train, w_val = w[perm_gen[:-nval_gen]], w[perm_gen[-nval_gen:]]
        rw1,history = reweight(X_gen_train, Y_gen_train, w_train, model_mc, model_mc_fp_i,fitargs2, val_data=(X_gen_val, Y_gen_val, w_val))
        rw = rw1[invperm_gen]
        history_arr.append(history)
        ws.append(rw[len(ws[-1]):])
        # save the weights if specified
        if weights_filename is not None:
            np.save(weights_filename, ws)
    return ws, history_arr

# prepare dataframe for PYTHIA6+embedding
observables2 = ['pt','m','q','rg','zg','mg','w','mcevi']
observabless2 = ['pts','ms','qs','rgs','zgs','mgs']
inputFiles = ['/nfs/roberts/home/ys668/output/run12/all050124_rc_hadcorr05_notrueptcons.root']
treeName = ["MatchedTree"]
df1 = applyCut(getData(observables2+observabless2, inputFiles[0], treeName[0]), 'pt>0 and pts>0 and w<5e-9')
df1['relw'] = np.asarray(df1['w'])/max(df1['w'])
    
# prepare dataframe for data
observables1 = ['pt','m','q','rg','zg','mg']
observabless1 = ['pts','ms','qs','rgs','zgs','mgs']
file = uproot.open('/nfs/roberts/home/ys668/output/run12/data062824.root')
tree = file["ResultTree"].arrays(observables1, library='np')
df2 = pd.DataFrame(np.transpose(np.array((np.concatenate(tree['pt']), np.concatenate(tree['m']), np.concatenate(tree['q']), np.concatenate(tree['rg']), np.concatenate(tree['zg']), np.concatenate(tree['mg'])))), columns=observabless1)

# reweight data according to (1-fake rate)
ptedges=[15, 20, 25, 30, 40, 65]
scaling=[0.97, 0.98, 0.99, 0.99, 0.99] # new for 20235003, same for notrueptcons
w2 = np.ones(df2.shape[0])
for i in range(0,df2.shape[0]):
    for j in range(0, len(ptedges)-1):
        if (ptedges[j] < df2.iloc[i]['pts'] < ptedges[j+1]):
            w2[i] *= scaling[j]
            break
df2['w'] = w2

# set up dictionary and arrays for multifold
datasets = {'pythia': df1, 'data': df2}
obs_multifold = ['pt', 'm', 'q', 'rg', 'zg', 'mg']
obs = {}
obs.setdefault('pt', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['pt'+s]),
})
obs.setdefault('q', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['q'+s]),
})
obs.setdefault('rg', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['rg'+s]),
})
obs.setdefault('zg', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['zg'+s]),
})
obs.setdefault('m', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['m'+s]),
})
obs.setdefault('mg', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['mg'+s]),
})

for obkey, ob in obs.items():
    ob['genobs'], ob['simobs'] = ob['func']('pythia', ''), ob['func']('pythia', 's')
    ob['dataobs'] = ob['func']('data', 's')
    print('Done with', obkey)

X_det = np.asarray([np.concatenate((obs[obkey]['dataobs'], obs[obkey]['simobs'])) for obkey in obs_multifold]).T
Y_det = np.concatenate((np.ones(len(obs['pt']['dataobs'])), np.zeros(len(obs['pt']['simobs']))))
X_gen = np.asarray([np.concatenate((obs[obkey]['genobs'], obs[obkey]['genobs'])) for obkey in obs_multifold]).T
Y_gen = np.concatenate((np.ones(len(obs['pt']['genobs'])), np.zeros(len(obs['pt']['genobs']))))
X_det = (X_det - np.mean(X_det, axis=0))/np.std(X_det, axis=0)
X_gen = (X_gen - np.mean(X_gen, axis=0))/np.std(X_gen, axis=0)
norm = sum(df2['w'])/sum(df1['relw'])
wdata = df2['w']
winit = norm*np.asarray(df1['relw'])

# set up multifold parameters
itnum = 8 
valnum = 0.5
batch_size = 50000 # changed on 060425
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
tf.config.list_physical_devices('GPU')
multifold_ws,history = omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (archs.DNN, det_args), (archs.DNN, mc_args),
                                fitargs1, fitargs2, val=valnum, it=itnum, trw_ind=-2, weights_filename=str(seed))

# save results
path='/nfs/roberts/scratch/pi_hlc7/ys668/test_101326/'

if (os.path.isfile("jet_noweight.csv")==False):
    df3 = df1[obs_multifold]
    df3.to_csv(path+'jet_noweight.csv',index=False)

for i in range(0, itnum):   
    df = pd.DataFrame(data=multifold_ws[2*(i+1)]/sum(multifold_ws[2*(i+1)]), columns=['weight'])
    df.to_csv(path+'it'+str(i+1)+'_seed'+str(seed)+'.csv',index=False)
