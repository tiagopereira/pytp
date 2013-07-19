"""
Set of math related tools.
"""
import numpy as np


def lclxtrem(vec, in_width, maxima=False):
    """
    Finds peaks in data. Converted from lclxtrem.
    """
    width = abs(in_width)
    # First derivative
    vecp = np.diff(vec)
    # Collapse the derivative to just +1, 0, or -1
    vecps = np.zeros(vecp.shape, dtype='i')
    vecps[vecp > 0.] = 1
    vecps[vecp < 0.] = -1
    # Derivative of the sign vectors
    vecpps = np.diff(vecps)
    # Keep the appropriate extremum
    if maxima:
        zids = np.where(vecpps < 0)[0]
    else:
        zids = np.where(vecpps > 0)[0]
    nidx = len(zids)
    flags = np.ones(nidx, dtype=np.bool)
    # Create an index vector with just the good points.
    if nidx == 0:
        if maxima:
            idx = (vec == np.max(vec))
        else:
            idx = (vec == np.min(vec))
    else:
        idx = zids + 1
    # Sort the extrema (actually, the absolute value)
    sidx = idx[np.argsort(np.abs(vec[idx]))[::-1]]
    # Scan down the list of extrema, start with the brightest and take out
    #   all extrema within width of the position.  Any that are too close should
    #   be removed from further consideration.
    if width > 1:
        i = 0      
        for i in range(nidx - 1):
            if flags[i]:
                flags[i + 1:][np.abs(sidx[i + 1:] - sidx[i]) <= width] = False
    #  The ones that survive are returned.
    return np.sort(sidx[flags])


def peakdetect_lcl(y_axis, x_axis=None, lookahead=300):
    """
    Wrapper to lclxtrem to mimic the behaviour of peakdetect.
    """
    maxima = lclxtrem(y_axis, lookahead, maxima=True)
    minima = lclxtrem(y_axis, lookahead, maxima=False)
    return np.array([x_axis[maxima], y_axis[maxima]]), \
           np.array([x_axis[minima], y_axis[minima]])
    
