#------------------------------------------------------
import h5py as h5
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.special import gamma, factorial
import skbeam.core.correlation as corr
import skbeam.core.roi as sk_roi
import scipy.ndimage as ndimage

#------------------------------------------------------
# Utility Functions
#------------------------------------------------------

def load_h5(f, name):
    """
    Load dataset with name `name` from HDF5 file `f`. 
    """
    return np.asarray(f[name])

def overlay_rois(ax, image, label_array):
    """
    This will plot the reqiured roi's on the image
    """
    tmp = np.array(label_array, dtype='float')
    tmp[label_array==0] = np.nan
    
    im_data = ax.imshow(image, interpolation='none', norm=LogNorm())
    im_overlay = ax.imshow(tmp, cmap='Paired', 
                   interpolation='nearest', alpha=.5,)
    
    return im_data, im_overlay

def gaus_2d(x, y, cx, wx, cy, wy):
    """
    Return the value of 2D Gaussian at location (x, y).

    Parameters:
    -----------
    `x` : x-coordinate
    `y` : y-coordinate
    `cx` : center in x
    `cy` : center in y
    `wx` : width in x
    `wy` : width in y
    """
    gx = np.power(x-cx,2.)/2./np.power(wx,2.)
    gy = np.power(y-cy,2.)/2./np.power(wy,2.)
    return np.exp(-gx-gy)
#------------------------------------------------------
# XPFS/XSVS Functions
#------------------------------------------------------

def speckle(k, kb, beta):
    """
    Speckle probability function for photon/pixel `k`, given average photons/pixel `kb` and contrast beta.

    * J..W. Goodman, Speckle Phenomena in Optics: Theory and Appli-
    cations (Roberts & Company, Englewood, 2007).
    """
    return (gamma(k+1./beta**2)/gamma(k+1.)/gamma(1./beta**2)) * (1+1./kb/beta**2)**(-k) * (1 + kb*beta**2)**(-1./beta**2)

def curvefn(kn):
    """
    Use to curve fit photon statistics to the speckle probability function for contrast estimation.
    """
    def curry(kb, beta):
        return speckle(kn, kb, beta)
    return curry

def contrast_from_droplet(p, kb=0.05):
    """
    Approximation for low kbar contrast estimation.  

    Sun, Yanwen, et al. 
    "Accurate contrast determination for X-ray speckle visibility spectroscopy."
    Journal of Synchrotron Radiation 27.4 (2020): 999-1007.
    """
    return (p[1]-2*p[2]/kb)/(2*p[2]-p[1])

def contast_ebars(beta0, kb, n_pixels, n_frames):
    """
    Photon statistics-based error bar estimation.

    Sun, Yanwen, et al. 
    "Accurate contrast determination for X-ray speckle visibility spectroscopy."
    Journal of Synchrotron Radiation 27.4 (2020): 999-1007.
    """
    return (1/(kb))*np.power((2*(1+beta0)/(n_pixels*n_frames)),0.5)

def unsparsify(idx, det='epix_3', frame_size=(704, 768)):
    """
    Use to unsparsify smalldata HDF5 saved as sparse arrays. 
    This will recreate the original photon map.

    Parameters:
    ----------------
    `idx` : event index of desired frame
    `det` : name of the detector (ex: 'epix_3')
    `frame_size` : tuple of detector sensor idth/height in pixels

    """
    idx_non_zero = h5file[det]['photon_sparse_row'][idx] !=0
    x = h5file[det]['photon_sparse_col'][idx][idx_non_zero].astype(int)
    #print(x)
    y = h5file[det]['photon_sparse_row'][idx][idx_non_zero].astype(int)
    z = h5file[det]['droplet_sparse_data'][idx][idx_non_zero].astype(int)

    xyz = np.stack((x, y, z), axis=1)

    droplet_img = np.zeros(frame_size)

    for element in xyz:
        droplet_img[element[1], element[0]] = element[2]
    return droplet_img

#------------------------------------------------------
# XPCS Functions
#------------------------------------------------------

def correct_illumination(imgs, roi, w, kernel_size=5):
    """ Correct the detector images for non-uniform illumination.
    This implementaion follows Part II in Duri et al. PHYS. REV. E 72, 051401 (2005).
    
    Args:
        imgs: stack of detector images
        roi: region of interest to consider. Important so that the normalization of the correction 
            is ~unity for the specific roi
        kernel_size: size of the kernel for box-average. Can be None, in which case no kernel is 
            applied. The kernel is used to smooth out remaining speckly structure in the intensity correction.
        w: width of the gaussian filter
        
    Returns:
        imgs: corrected images, cropped to an extended box aroung the roi
        roi: new roi for the cropped image
        bp: correction factor
    """
    if kernel_size is None:
        extend = 10
    else:
        extend=2*kernel_size

    bp = np.mean(imgs, axis=0)
    if kernel_size is not None:
        kernel = np.zeros((2*kernel_size,2*kernel_size))
        x = np.arange(-kernel_size+0.5,kernel_size,1.0)
        y = np.arange(-kernel_size+0.5,kernel_size,1.0)
        for i in range(2*kernel_size):
            for j in range(2*kernel_size):
                kernel[i,j] = gaus_2d(x[i],y[j],0.,w,0.,w) # gaussian filter
        bp = ndimage.convolve(bp, kernel)
    bp = bp / bp[roi].mean()
    zero = bp==0
    bp[zero] = 1e-6
    imgs_corr = imgs/bp
    return imgs_corr, roi, bp
