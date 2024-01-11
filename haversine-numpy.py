# import pandas as pd

import argparse
import sys
import time

import numpy as np
import pandas as pd
import os
from numba import njit
import jax
import jax.numpy as jnp
import torch
import tensorflow as tf

def gen_data(size):
    sys.stdout.write("generating data...")
    sys.stdout.flush()
    lats = np.ones(size, dtype="float64") * 0.0698132
    lons = np.ones(size, dtype="float64") * 0.0698132
    sys.stdout.write("done.")
    sys.stdout.flush()
    return lats, lons


# Haversine definition
def haversine(lat2, lon2):
    miles_constant = 3959.0
    lat1 = 0.70984286
    lon1 = 1.2389197

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    mi = miles_constant * c
    return mi

@njit(parallel=True)
def haversine_numba(lat2, lon2):
    miles_constant = 3959.0
    lat1 = 0.70984286
    lon1 = 1.2389197

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    mi = miles_constant * c
    return mi

@jax.jit
def haversine_jax(lat2, lon2):
    miles_constant = 3959.0
    lat1 = 0.70984286
    lon1 = 1.2389197

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = jnp.sin(dlat / 2.0) ** 2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon / 2.0) ** 2
    c = 2.0 * jnp.arcsin(jnp.sqrt(a))
    mi = miles_constant * c
    return mi

@torch.jit.script
def haversine_pytorch(lat2, lon2):
    miles_constant = torch.tensor(3959.0, dtype=torch.float64)
    lat1 = torch.tensor(0.70984286, dtype=torch.float64)
    lon1 = torch.tensor(1.2389197)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
    c = 2.0 * torch.arcsin(torch.sqrt(a))
    mi = miles_constant * c
    return mi

def haversine_tensorflow(lat2, lon2):
    miles_constant = 3959.0
    lat1 = 0.70984286
    lon1 = 1.2389197

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = tf.sin(dlat / 2.0) ** 2 + tf.cos(lat1) * tf.cos(lat2) * tf.sin(dlon / 2.0) ** 2
    c = 2.0 * tf.asin(tf.sqrt(a))
    mi = miles_constant * c

    return mi

class haversine_torch_model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lat2, lon2):
        miles_constant = torch.tensor(3959.0, dtype=torch.float64)
        lat1 = torch.tensor(0.70984286, dtype=torch.float64)
        lon1 = torch.tensor(1.2389197)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            torch.sin(dlat / 2.0) ** 2
            + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * torch.arcsin(torch.sqrt(a))
        mi = miles_constant * c
        return mi


def run_haversine_with_scalar(args):
    lat2, lon2 = gen_data(args.scale)
    dist2 = haversine(lat2, lon2)

    print('num rows in lattitudes: ', len(lat2))
    # lat2 = weldarray(lat2)
    # lon2 = weldarray(lon2)

    start = time.time()
    dist2 = haversine(lat2, lon2)
    # dist2 = dist2.evaluate()
    end = time.time()

    print('****************************')
    print('numpy took {} seconds'.format(end - start))
    print('****************************')
    print(dist2[0:5])

    start = time.time()
    dist2 = haversine_numba(lat2, lon2)
    # dist2 = dist2.evaluate()
    end = time.time()

    print('****************************')
    print('numba took {} seconds'.format(end - start))
    print('****************************')
    print(dist2[0:5])

    start = time.time()
    dist2 = haversine_jax(lat2, lon2)
    # dist2 = dist2.evaluate()
    end = time.time()

    print('****************************')
    print('jax took {} seconds'.format(end - start))
    print('****************************')
    print(dist2[0:5])

    lat2_torch = torch.tensor(lat2, dtype=torch.float64)
    lon2_torch = torch.tensor(lon2, dtype=torch.float64)
    torch.set_num_threads(args.threads)

    start = time.time()
    dist2 = haversine_pytorch(lat2_torch, lon2_torch)
    # dist2 = dist2.evaluate()
    end = time.time()

    print('****************************')
    print('torch took {} seconds'.format(end - start))
    print('****************************')
    print(dist2[0:5])

    torch.reshape(lat2_torch, (1 << 10, -1))
    torch.reshape(lon2_torch, (1 << 10, -1))

    #start = time.time()
    #dist2 = haversine_pytorch(lat2_torch, lon2_torch)
    # dist2 = dist2.evaluate()
    #end = time.time()

    #print('****************************')
    #print('torch + reshape took {} seconds'.format(end - start))
    #print('****************************')
    #print(dist2[0:5])

    start = time.time()
    dist2 = haversine(lat2, lon2)
    # dist2 = dist2.evaluate()
    end = time.time()

    print('****************************')
    print('numpy took {} seconds'.format(end - start))
    print('****************************')
    print(dist2[0:5])

    model = haversine_torch_model()
    model.eval()
    traced = torch.jit.trace(model, (lat2_torch, lon2_torch))
    traced = torch.jit.freeze(traced)

    start = time.time()
    dist2 = traced(lat2_torch, lon2_torch)
    # dist2 = dist2.evaluate()
    end = time.time()

    print('****************************')
    print('torch model took {} seconds'.format(end - start))
    print('****************************')
    print(dist2[0:5])

    # lat2_tf = tf.convert_to_tensor(lat2, dtype=tf.float64)
    # lon2_tf = tf.convert_to_tensor(lon2, dtype=tf.float64)
    # start = time.time()
    # dist2 = haversine_tensorflow(lat2_tf, lon2_tf)
    # dist2 = dist2.evaluate()
    #end = time.time()

    # print('****************************')
    # print('tensorflow took {} seconds'.format(end - start))
    # print('****************************')
    # print(dist2[0:5])



parser = argparse.ArgumentParser(
    description="give num_els of arrays used for nbody"
)
parser.add_argument('-s', "--scale", type=int, default=29,
                    help=("Data size"))
parser.add_argument('-t', "--threads", type=int, default=15,
                    help=("Threads"))

args = parser.parse_args()
os.environ["h"] = str(args.threads)
print("threads",args.threads)
args.scale = (1 << args.scale)
run_haversine_with_scalar(args)
