import numpy as np
from scipy.integrate import dblquad, tplquad
from joblib import Parallel, delayed

# Perform integration in parallel
def dblquad_(func, a, b, gfun, hfun, num_jobs = 8):

    ab_ranges = np.linspace(a, b, num_jobs + 1)
    ab_ranges = [(ab_ranges[i], ab_ranges[i + 1]) for i in range(num_jobs)]

    results = np.array(Parallel(n_jobs=num_jobs)(delayed(dblquad)(func, a_, b_, gfun, hfun) for (a_, b_) in ab_ranges))
    return results.sum(axis=0)

def tplquad_(func, a, b, gfun, hfun, qfun, rfun, num_jobs = 8):

    ab_ranges = np.linspace(a, b, num_jobs + 1)
    ab_ranges = [(ab_ranges[i], ab_ranges[i + 1]) for i in range(num_jobs)]

    results = np.array(Parallel(n_jobs=num_jobs)(delayed(tplquad)(func, a_, b_, gfun, hfun, qfun, rfun) for (a_, b_) in ab_ranges))
    return results.sum(axis=0)