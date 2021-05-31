import numpy as np

def construct_features(seqs):
    X = []
    for i in range(0, len(seqs), 1):
        arr = np.array(seqs[i])
        data_mean = np.mean(arr, axis=0)
        min = np.min(arr, axis=0)
        max = np.max(arr, axis=0)
        std = np.std(arr, axis=0)
        joined = np.concatenate((data_mean, min, max, std), axis=0)
        X.append(joined.tolist())
    
    return X