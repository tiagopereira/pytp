"""
Collection of cython-ized fast math or image processing libraries.
"""
import numpy as np
cimport numpy as np
cimport cython

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t
DTYPEi = np.int32
ctypedef np.int32_t DTYPEi_t
DTYPEl = np.int64
ctypedef np.int64_t DTYPEl_t


@cython.boundscheck(False)  # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn of bounds-checking for entire function
def replace_nans(np.ndarray[DTYPEf_t, ndim=2] array, int max_iter, float tol,
                 int kernel_size=1, str method='localmean'):
    """
    Replace NaN elements in an array using an iterative image inpainting algorithm.
    
    The algorithm is the following:
    
    1) For each element in the input array, replace it by a weighted average
       of the neighbouring elements which are not NaN themselves. The weights depends
       of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )
       
    2) Several iterations are needed if there are adjacent NaN elements.
       If this is the case, information is "spread" from the edges of the missing
       regions iteratively, until the variation is below a certain threshold. 
    
    Parameters
    ----------
    
    array : 2d np.ndarray
        an array containing NaN elements that have to be replaced
    
    max_iter : int
        the number of iterations

   tol : float
        tolerance
    
    kernel_size : int
        the size of the kernel, default is 1
        
    method : str
        the method used to replace invalid values. Valid options are
        `localmean`.
        
    Returns
    -------
    
    filled : 2d np.ndarray
        a copy of the input array, where NaN elements have been replaced.
        
    """
    cdef int i, j, I, J, it, n, k, l
    cdef int n_invalids
    cdef np.ndarray[DTYPEf_t, ndim=2] filled = np.empty([array.shape[0],
                                                         array.shape[1]],
                                                         dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] kernel = np.empty((2*kernel_size+1,
                                                         2*kernel_size+1),
                                                         dtype=DTYPEf ) 
    cdef np.ndarray[np.int_t, ndim=1] inans
    cdef np.ndarray[np.int_t, ndim=1] jnans
    # indices where array is NaN
    inans, jnans = np.nonzero( np.isnan(array) )
    # number of NaN elements
    n_nans = len(inans)
    # arrays which contain replaced values to check for convergence
    cdef np.ndarray[DTYPEf_t, ndim=1] replaced_new = np.zeros(n_nans,
                                                              dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] replaced_old = np.zeros(n_nans,
                                                              dtype=DTYPEf)
    # depending on kernel type, fill kernel array
    if method == 'localmean':
        for i in range(2*kernel_size+1):
            for j in range(2*kernel_size+1):
                kernel[i,j] = 1.0
    else:
        raise ValueError( 'method not valid. Should be one of `localmean`.')
    # fill new array with input elements
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            filled[i,j] = array[i,j]
    # make several passes
    # until we reach convergence
    for it in range(max_iter):
        # for each NaN element
        for k in range(n_nans):
            i = inans[k]
            j = jnans[k]
            # initialize to zero
            filled[i,j] = 0.0
            n = 0
            # loop over the kernel
            for I in range(2*kernel_size+1):
                for J in range(2*kernel_size+1):
                    # if we are not out of the boundaries
                    if i+I-kernel_size < array.shape[0] and i+I-kernel_size >= 0:
                        if j+J-kernel_size < array.shape[1] and j+J-kernel_size >= 0:
                                                
                            # if the neighbour element is not NaN itself.
                            if filled[i+I-kernel_size, j+J-kernel_size] == \
                                filled[i+I-kernel_size, j+J-kernel_size] :
                                
                                # do not sum itself
                                if I-kernel_size != 0 and J-kernel_size != 0:
                                    
                                    # convolve kernel with original array
                                    filled[i,j] = filled[i,j] + \
                                        filled[i+I-kernel_size, j+J-kernel_size] * kernel[I, J]
                                    n = n + 1
            # divide value by effective number of added elements
            if n != 0:
                filled[i,j] = filled[i,j] / n
                replaced_new[k] = filled[i,j]
            else:
                filled[i,j] = np.nan
        # check if mean square difference between values of replaced 
        # elements is below a certain tolerance
        if np.mean( (replaced_new-replaced_old)**2 ) < tol:
            break
        else:
            for l in range(n_nans):
                replaced_old[l] = replaced_new[l]
    return filled


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)  # turn of bounds-checking for entire function
def peakdetect(np.ndarray[DTYPEf_t, ndim=1] y_axis,
               np.ndarray[DTYPEf_t, ndim=1] x_axis=None,
               float delta_max=0., float delta_min=0.,
               int lookahead_max=300, int lookahead_min=300):
    """
    peakdetect(y_axis, x_axis=None, delta_max=0., delta_min=0.,
               lookahead_max=300, lookahead_min=300)
    
    Function for detecting local maximas and minimas in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively
    
    Parameters
    ----------
    y_axis : 1-D ndarray (double type)
        Array containg the signal over which to find peaks
    x_axis : 1-D ndarray (double type), optional
        Array whose values correspond to the y_axis array and is used
        in the return to specify the postion of the peaks. If omitted an
        index of the y_axis is used. (default: None)
    lookahead_max, lookahead_min : int, optional
        Distance to look ahead from a peak candidate to determine if it
        is the actual peak. '(sample / period) / f' where '4 >= f >= 1.25'
        might be a good value. lookahead_max is for maximum peaks,
        lookahead_min for minimum peaks.
    delta_max, delta_min : float, optional
        Specifies a minimum difference between a peak and the following
        points, before a peak may be considered a peak. Useful to hinder
        the function from picking up false peaks towards to end of the
        signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
        delta function causes a 20% decrease in speed, when omitted.
        Correctly used it can double the speed of the function
        delta_max is for maximum peaks, delta_min for minimum peaks.
    
    Returns
    -------
    max_peaks, min_peaks : 2-D ndarrays
        Two arrays containing the maxima and minima location. Each array
        has a shape of (2, len(y_axis)). First dimension is for peak position
        (first index) or peak value (second index). Second dimension is
        for the number of peaks (maxima or minima).
        
    Notes
    ----- 
    Downloaded and adapted from https://gist.github.com/1178136
    
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html
    """
    cdef int length = len(y_axis)
    cdef np.ndarray[DTYPEf_t, ndim=2] max_peaks = np.zeros((2, length),
                                                         dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] min_peaks = np.zeros((2, length),
                                                         dtype=DTYPEf)
    cdef int nmax = 0
    cdef int nmin = 0
    cdef int dump = 0
    cdef float mxpos, mnpos
    cdef float mn = np.inf
    cdef float mx = -np.inf
    cdef int i, lookahead
    
    lookahead = max(lookahead_max, lookahead_min)
    assert lookahead > 0
    for i in range(length - lookahead):
        if y_axis[i] > mx:
            mx = y_axis[i]
            mxpos = x_axis[i]
        if y_axis[i] < mn:
            mn = y_axis[i]
            mnpos = x_axis[i]
        #### look for max ####
        if (y_axis[i] < mx - delta_max) and (mx != np.Inf):
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[i:i + lookahead_max].max() < mx:
                max_peaks[0, nmax] = mxpos
                max_peaks[1, nmax] = mx
                nmax += 1
                if mx == y_axis[0]:
                    dump = 1
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if i + lookahead_max >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue
        #### look for min ####
        if (y_axis[i] > mn + delta_min) and (mn != -np.Inf):
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[i:i + lookahead_min].min() > mn:
                min_peaks[0, nmin] = mnpos
                min_peaks[1, nmin] = mn
                nmin += 1
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if i + lookahead_min >= length:
                    # end is within lookahead no more peaks can be found
                    break
    # Remove the false hit on the first value of the y_axis
    if dump == 1:
        max_peaks = max_peaks[:, 1:]
        nmax -= 1
    else:
        min_peaks = min_peaks[:, 1:]
        nmin -= 1
    return max_peaks[:, :nmax], min_peaks[:, :nmin]