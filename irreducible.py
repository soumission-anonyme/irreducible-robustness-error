import numpy as np
from scipy.signal import fftconvolve
from scipy.integrate import simps

def zeta_(densities, vicinity, x1, x2):

    densities = np.stack(densities)
    S = densities.max(axis=0, keepdims=True) == densities
    S_ = fftconvolve(S, np.repeat(vicinity[np.newaxis, ...], repeats=len(densities), axis=0), mode='same', axes=np.arange(vicinity.ndim)+1).round().astype(int)
    E = (~((S_ == 0) | (S_ == vicinity.sum())) | ~S) * densities
    # robustness_values = Z_p0 * (S_ == 0) + Z_p1 * (S_ == V.sum())
    return simps(simps(E.sum(0), x2), x1)

# print(f"Simpson's rule Integral Result: {zeta:6f}, rough estimate: {1 - robustness_values.sum()* _d**2:6f}", )