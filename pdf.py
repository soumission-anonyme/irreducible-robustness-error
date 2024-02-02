import numpy as np
import scipy.integrate as integrate

# Moons
def moon_upper_(x1, x2, sigma = 0.3):
    constant = 1 / (2 * np.pi**2 * sigma**2)
    exp_term = np.exp(-(x1**2 + x2**2 + 1) / (2 * sigma**2))
    integral = numerical_part_moon_upper(x1, x2, sigma)
    return constant * exp_term * integral

moon_upper = np.vectorize(moon_upper_)

def moon_lower_(x1, x2, sigma = 0.3):
    constant = 1 / (2 * np.pi**2 * sigma**2)
    exp_term = np.exp(-((x1 - 1)**2 + (x2 - 0.5)**2 + 1) / (2 * sigma**2))
    integral = numerical_part_moon_lower(x1, x2, sigma)
    return constant * exp_term * integral

moon_lower = np.vectorize(moon_lower_)

def numerical_part_moon_upper(x1, x2, sigma):
    integrand = lambda t: np.exp((x1*np.cos(t) + x2*np.sin(t)) / sigma**2)
    result, _ = integrate.quad(integrand, 0, np.pi)
    return result

def numerical_part_moon_lower(x1, x2, sigma):
    integrand = lambda t: np.exp((- (x1 - 1) * np.cos(t) - (x2 - 0.5) * np.sin(t)) / sigma**2)
    result, _ = integrate.quad(integrand, 0, np.pi)
    return result


# Moons convolved
from scipy.integrate import tplquad

def integrand_moon_upper_conv(u, v, t, sigma):
    return np.exp(-(u**2 + v**2 + 1) / (2*sigma**2)) * np.exp((u*np.cos(t) + v*np.sin(t)) / sigma**2)

def integrand_moon_lower_conv(u, v, t, sigma):
    return np.exp(-((u - 1)**2 + (v - 0.5)**2 + 1) / (2*sigma**2)) * np.exp((- (u - 1) * np.cos(t) - (v - 0.5) * np.sin(t)) / sigma**2)

from multiprocessing import Pool
from functools import partial
from itertools import repeat

def moon_upper_conv(x1, x2, sigma = 0.3, epsilon = 0.15):

    s = np.array(x1).shape
    x1, x2 = np.array(x1).flatten(), np.array(x2).flatten()

    with Pool() as pool:
        results = pool.starmap(partial(tplquad, args=(sigma,)), zip(repeat(integrand_moon_upper_conv), repeat(0), repeat(np.pi), x2 - epsilon, x2 + epsilon, x1 - epsilon, x1 + epsilon))
    
    return np.array(results)[:, 0].reshape(s) / (4 * epsilon**2) / (2*np.pi**2*sigma**2)

def moon_lower_conv(x1, x2, sigma = 0.3, epsilon = 0.15):

    s = np.array(x1).shape
    x1, x2 = np.array(x1).flatten(), np.array(x2).flatten()

    with Pool() as pool:
        results = pool.starmap(partial(tplquad, args=(sigma,)), zip(repeat(integrand_moon_lower_conv), repeat(0), repeat(np.pi), x2 - epsilon, x2 + epsilon, x1 - epsilon, x1 + epsilon))
    
    return np.array(results)[:, 0].reshape(s) / (4 * epsilon**2) / (2*np.pi**2*sigma**2)