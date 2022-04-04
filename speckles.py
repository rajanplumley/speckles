import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import psana_utility
import argparse
from psana import *
from pulse_fit_class3 import *
from ImgAlgos.PyAlgos import photons as photonFunc
from mpi4py import MPI
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.special import gamma, factorial

def speckle(k, kbar, beta):
    """
    Speckle statistics function [*]
    
    * J..W. Goodman, Speckle Phenomena in Optics: Theory and Appli-
    cations (Roberts & Company, Englewood, 2007).
    """
    return (gamma(k+1./beta**2)/gamma(k+1.)/gamma(1./beta**2)) * (1+1./kbar/beta**2)**(-k) * (1 + kbar*beta**2)**(-1./beta**2)

def fit_beta(kbar, hist, p0=0.3):
    """
    Use speckle statistics function to fit contrasts.  Returns contasts
    and errors for 1 photon, 2 photons, 3 photons.
    """
    fits = []
    betas = []
    pcovs = []

    def curvefn(kn):
        def curry(kbar, beta):
            return speckle(kn, kbar, beta)
        return curry
    
    # Remove any negative-valued points
    mask1 = hist[:, 0]>-1
    mask2 = hist[:, 1]>-1
    mask = np.logical_and(mask1, mask2)

    # Fit contrasts
    for k in range(1,4):
        fit, pcov = curve_fit(curvefn(i), kbar[mask], hist[:, 1][mask], p0=p0)
        fits.append(fit)
        betas.append(fits[0])
        pcovs.append(pcov)

    return betas, fits, pcovs
