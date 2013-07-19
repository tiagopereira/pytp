"""
Set of tools to analyse IRIS spectra.
"""
import numpy as np
import scipy.constants as const
from scipy import interpolate as interp
from scipy import ndimage
from pytp.math.utilsmath import peakdetect_lcl
from pytp.math.utilsfast import peakdetect, replace_nans


# pylint: disable=E1101
def iris_get_spec_features(wave_in, spec_in, wave_ref, vrange=(-60, 60),
                           lookahead=2, delta=2e-11, pmin=5, pfit=2,
                           gsigma=6, gnk=3):
    """
    Extracts the line properties (blue peak, line centre, red peak)
    of a Mg II line (h or k).

    Parameters
    ----------
    wave - 1D array
        array of wavelengths/velocities
    spec - nD array
        spectral array, wavelength in last index
    vrange - tuple or list
        Doppler velocity limits around wave_ref, determing the range that
        will be analysed
    lookahead - integer
        Parameter for peak finding, how many points to look ahead for peaks
    delta - float
        Minimum threshold for peak in spec units
    pmin - integer
        Number of points to find around the minimum
    pfit - integer
        Number of points to fit a parabola around the minimum
    gsigma - integer
        Gaussian sigma (in pixels) to smooth velocities and find outliers
    gnk - integer
        Kernel size for replace_nans

    Returns
    -------
        results - tuple with results, bp, lc, rp
    """
    # flatten spectrum
    shape_in = spec_in.shape[:-1]
    spec = np.reshape(spec_in, (np.prod(shape_in), spec_in.shape[-1]))
    # Determine wavelength range to use
    vaxis = const.c / 1e3 * (wave_in - wave_ref) / wave_ref
    wvi = max(np.where(vaxis >= vrange[0])[0][0] - 1, 0)
    wvf = min(np.where(vaxis <= vrange[1])[0][-1] + 1, vaxis.shape[0] - 1)
    vel = vaxis[wvi:wvf]
    spec = spec[:, wvi:wvf]
    nspec = len(spec)
    lc = np.ma.masked_all((2, nspec))   # line core (central reversal, k3/h3)
    bp = np.ma.masked_all((2, nspec))   # blue peak
    rp = np.ma.masked_all((2, nspec))   # red peak

    # LINE CORE, main loop
    for i in range(nspec):
        guess = 5.
        pts_min = pmin
        lc[0, i], lc[1, i] = mg_single(vel, spec[i], guess, delta=delta,
                                       lookahead=lookahead, ppts=pfit,
                                       pts_min=pts_min, use_deriv=False)
    # two iterations for cleaning up
    for _nidx in range(2):
        # put back in initial shape
        lc = np.reshape(lc, (2,) + shape_in)
        # find outliers,distant more than 3 km/s from a smoothed Gaussian
        xpto = lc[0].data.copy()
        xpto[lc[0].mask] = 0.
        diff = np.abs(xpto - ndimage.gaussian_filter(xpto, gsigma,
                                                     mode='wrap'))
        lc[:, diff > 3] = np.nan
        lc.data[:, lc[0].mask] = np.nan   # so that replace_nans will work
        # perform inpainting on masked image to get next best guess
        lc_guess = replace_nans(lc[0].data, 10, .5, gnk, 'localmean')
        # put back into flat shape, redo loop with new guess
        lc = np.reshape(lc, (2, np.prod(shape_in)))
        lc_guess = np.reshape(lc_guess, (np.prod(shape_in)))
        for i in range(nspec):
            if not np.isfinite(lc[0, i]):
                lc[0, i], lc[1, i] = \
                    mg_single(vel, spec[i], lc_guess[i], ppts=pfit,
                              lookahead=lookahead, delta=delta, pts_min=pmin,
                              force_guess=True)
    print('Last run with use_deriv')
    for i in range(nspec):
        if not np.isfinite(lc[0, i]):
            lc[0, i], lc[1, i] = \
                    mg_single(vel, spec[i], guess, delta=0., ppts=pfit,
                              lookahead=lookahead, pts_min=pts_min,
                              use_deriv=True)
    lc = np.reshape(lc, (2,) + shape_in)
    # clean up some isolated problem pixels
    xpto = lc[0].data.copy()
    xpto[lc[0].mask] = 40.
    idx = np.abs(xpto - ndimage.gaussian_filter(xpto, 1., mode='wrap')) > 10
    lc[:, idx] = np.nan
    lc[0] = replace_nans(lc[0].data, 10, .5, gnk, 'localmean')
    lc[1] = replace_nans(lc[1].data, 10, .5, gnk, 'localmean')
    # get mask for bad values
    bad_mask = ndimage.binary_fill_holes(np.abs(xpto -
                          ndimage.gaussian_filter(xpto, 1.7, mode='wrap')) > 6)
    lc[:, bad_mask] = np.ma.masked
    # get better lc estimate for peak detection
    lc = np.reshape(lc, (2,) + shape_in)
    xpto = lc[0].data.copy()
    xpto[lc[0].mask] = 0.
    diff = np.abs(xpto - ndimage.gaussian_filter(xpto, 10. * gsigma / 6,
                                                mode='wrap'))
    xpto[(diff > 4) | lc[0].mask] = np.nan
    # perform inpainting on masked image to get next best guess
    lc_guess = replace_nans(xpto, 10, .5, 2, 'localmean')
    lc_guess = np.reshape(lc_guess, (np.prod(shape_in)))

    # PEAKS, main loop
    print('Peaks main loop...')
    for i in range(nspec):
        bp[:, i], rp[:, i] = mg_peaks_single(vel, spec[i], lc_guess[i],
                                             lookahead=lookahead)
    lc = np.reshape(lc, (2,) + shape_in)
    bp = np.reshape(bp, (2,) + shape_in)
    rp = np.reshape(rp, (2,) + shape_in)
    return bp, lc, rp


# pylint: disable=E1101
def mg_single(vel, spec, guess, lookahead=2, pts_min=5, margin=15, ppts=2,
              delta=5e-12, force_guess=False, use_deriv=False):
    """
    Calculates the properties for a single Mg II spectrum. Similar to
    specquant.linecentre, but also does peakdetect and everything more
    streamlined.

    For the conv spectra.
    """
    if hasattr(spec[0], 'mask') or np.isnan(guess):
        return np.ma.masked, np.ma.masked
    deriv_flag = False
    pmax, pmin = peakdetect(spec, vel, delta_max=delta, delta_min=delta,
                            lookahead_max=lookahead, lookahead_min=lookahead)
    # Use peak detect unless forcing the use of supplied guess
    if not force_guess:
        if np.any(pmax) and np.any(pmin):
            pmax = pmax[:, (pmax[0] > vel[0] + 10) & (pmax[0] < vel[-1] - 10.)]
            pmin = pmin[:, (pmin[0] > vel[0] + 10) & (pmin[0] < vel[-1] - 10.)]
            lpmax = len(pmax[0])
            lpmin = len(pmin[0])
            if (lpmin, lpmax) in [(1, 2), (3, 1), (3, 2), (3, 4), (5, 4),
                                  (7, 6)]:
                # most straightforward case: take the middle one
                guess = pmin[0][lpmin // 2]
                pts_min = pts_min // 2
            elif (lpmin, lpmax) in [(2, 2), (2, 3), (3, 3), (4, 2), (4, 3),
                                    (4, 4)]:
                pts_min = pts_min // 2
                # take the lowest minimum between the two largest maxima
                # locations of two largest maxima
                lmax = np.sort(pmax[0][np.argsort(pmax[1])[-2:]])
                # minima inside the above window
                idx = (pmin[0] > lmax[0]) & (pmin[0] < lmax[1])
                if np.any(idx):
                    guess = pmin[0, idx][np.argmin(pmin[1, idx])]
            elif lpmin == 2 and lpmax == 1:  # was [1, 3]
                if np.max(pmin[1]) / np.min(pmin[1]) > 1.3:
                    # use lowest minimum
                    guess = pmin[0, np.argmin(pmin[1])]
                    pts_min = pts_min // 2
                else:
                    deriv_flag = True
            elif lpmax == 1:
                deriv_flag = True
            elif lpmin == 1:
                guess = pmin[0, 0]
                pts_min = pts_min // 2
        elif np.any(pmax):
            lpmax = len(pmax[0])
            if lpmax == 1:
                deriv_flag = True
    else:
        if np.any(pmax):
            pmax = pmax[:, (pmax[0] > vel[0] + 20) & (pmax[0] < vel[-1] - 20.)]
        if np.any(pmax):
            lpmax = len(pmax[0])
            if lpmax == 1:
                deriv_flag = True
    if force_guess:
        deriv_flag = False

    if deriv_flag:   # special case to use the derivatives
        if not use_deriv:
            return np.ma.masked, np.ma.masked
        dd = np.abs(np.diff(spec))
        # define the boundaries to inspect for peak asymmetry and derivative
        incr = int(15. / (vel[1] - vel[0]))
        idxm = np.argmin(np.abs(vel - pmax[0, 0]))
        vv = [max(0, idxm - incr), min(idxm + incr, spec.shape[0] - 1)]
        vv2 = [idxm - incr * 1.3, idxm + incr * 1.3]
        vv2 = vv
        if spec[vv[0]] > spec[vv[1]]:   # k3/h3 peak on left side
            try:
                der = dd[vv2[0] + margin:idxm - margin]
                pidx = vv2[0] + np.argmin(der) + margin + 1
            except IndexError:
                return np.ma.masked, np.ma.masked
        else:                           # k3/h3 peak on right side
            der = dd[idxm + margin:vv2[1] - margin]
            pidx = idxm + np.argmin(der) + margin + 1
        lc = vel[pidx]
        lc_int = spec[pidx]
    else:
        # Approximate index of guess and of spectral minimum around it
        try:
            idg = np.argmin(np.abs(vel - guess))
            ini = np.argmin(spec[idg - pts_min:idg + pts_min]) + idg - pts_min
        except ValueError:
            return np.ma.masked, np.ma.masked
        # if no points, return masked
        if len(vel[ini - ppts:ini + ppts + 1]) == 0:
            return np.ma.masked, np.ma.masked
        # Fit parabola
        fit = np.polyfit(vel[ini - ppts:ini + ppts + 1],
                        spec[ini - ppts:ini + ppts + 1], 2)
        # Convert poly to parabola coefficients
        lc = -fit[1] / (2 * fit[0])
        lc_int = fit[2] - fit[0] * lc ** 2
        # If fitted minimum is furthen than 4 wavelenght points, mask
        if np.abs(lc - vel[ini]) > 4 * (vel[1] - vel[0]):
            lc = np.ma.masked
            lc_int = np.ma.masked
    return lc, lc_int


def mg_peaks_single(vel, spec, lc, lookahead=2):
    """
    Calculate the intensity and velocity of the two peaks (h2v and h2r,
    or k2v and k2r)
    """
    bp = np.ma.masked_all(2)
    rp = np.ma.masked_all(2)
    if hasattr(spec[0], 'mask'):
        return bp, rp
    pmax, pmin = peakdetect_lcl(spec, vel, lookahead=lookahead)
    if np.any(pmax):
        # remove peaks more than 30 km/s from line centre
        if np.isfinite(lc):
            pmax = pmax[:, np.abs(pmax[0] - lc) < 30]
        else:
            pmax = pmax[:, np.abs(pmax[0] - 0) < 30]
        lpmax = len(pmax[0])
        lpmin = len(pmin[0])
        # SELECTION OF TWO PEAKS
        if lpmax > 4:   # too many maxima, take inner 4
            pmax = pmax[:, lpmax // 2 - 2: lpmax // 2 + 2]
            lpmax = 4
        if lpmax == 1:
            if pmax[0, 0] > lc:
                rp = pmax[:, 0]
            else:   # by default assign single max to blue peak if no core
                bp = pmax[:, 0]
        elif lpmax == 2:
            bp = pmax[:, 0]
            rp = pmax[:, 1]
        elif lpmax == 3:
            # take the one close to the line core, and from the
            # remaining two chose the strongest
            if pmax[0, 0] < lc < pmax[0, 1]:
                bp = pmax[:, 0]
                rp = pmax[:, np.argmax(pmax[1, 1:]) + 1]
            elif pmax[0, 1] < lc < pmax[0, 2]:
                bp = pmax[:, np.argmax(pmax[1, :-1])]
                rp = pmax[:, -1]
            else:
                aa = pmax[:, pmax[1] != np.min(pmax[1])]
                bp = aa[:, 0]
                rp = aa[:, 1]
        elif lpmax == 4:
            # first look for special case when two close weak inner peaks
            # are taken as peaks. In this case take the outer peaks.
            sep_out = pmax[0, 3] - pmax[0, 0]
            sep_in = pmax[0, 2] - pmax[0, 1]
            rtt = (pmax[1, 0] > 1.06 * pmax[1, 1]) and \
                  (pmax[1, 3] > 1.06 * pmax[1, 2])
            if (sep_out < 40) and (sep_in < 13) and rtt:
                bp = pmax[:, 0]
                rp = pmax[:, 3]
            # if line core in the middle of the four, or outside the
            # maxima region, take inner two maxima
            elif (lc > pmax[0, 1]) and (lc < pmax[0, 2]):
                bp = pmax[:, 1]
                rp = pmax[:, 2]
            elif (lc > pmax[0, 3]) or (lc < pmax[0, 0]):
                # for now, just take inner two
                bp = pmax[:, 1]
                rp = pmax[:, 2]
            elif lc < pmax[0, 1]:  # proceed like for 3 maxima
                bp = pmax[:, 0]
                rp = pmax[:, np.argmax(pmax[1, 1:]) + 1]
            elif lc < pmax[0, 3]:
                bp = pmax[:, np.argmax(pmax[1, :3])]
                rp = pmax[:, 3]
    # interpolation for more precise peaks
    if bp[0]:
        idg = np.argmin(np.abs(vel - bp[0]))
        llim = max(0, idg - 2)
        hlim = min(idg + 2, spec.shape[0] - 1)
        nvel = np.linspace(vel[llim], vel[hlim], 45)
        llim = max(0, idg - 3)
        hlim = min(idg + 4, spec.shape[0])
        spl = interp.splrep(vel[llim:hlim], spec[llim:hlim], k=3, s=0)
        nspec = interp.splev(nvel, spl, der=0)
        midx = np.argmax(nspec)
        bp[0] = nvel[midx]
        bp[1] = nspec[midx]
    if rp[0]:
        idg = np.argmin(np.abs(vel - rp[0]))
        llim = max(0, idg - 2)
        hlim = min(idg + 2, spec.shape[0] - 1)
        nvel = np.linspace(vel[llim], vel[hlim], 45)
        llim = max(0, idg - 3)
        hlim = min(idg + 4, spec.shape[0])
        spl = interp.splrep(vel[llim:hlim], spec[llim:hlim], k=3, s=0)
        nspec = interp.splev(nvel, spl, der=0)
        midx = np.argmax(nspec)
        rp[0] = nvel[midx]
        rp[1] = nspec[midx]
    return bp, rp


def getvel_centroid(wguess, wave, spec, spec_guess, pts=5):
    """
    Calculates the centroid of the spectrum in 5 points around the line centre.
    No intensity is obtained (perhaps use linear interpolation?)
    """
    iwv = np.argmin(np.abs(wave - wguess))
    iii = np.arange(iwv - pts, iwv + pts + 1)
    iwv = spec_guess[..., iii].argmin(-1) + iwv - pts
    vaxis = const.c / 1e3 * (wave - wguess) / wguess
    vel = np.zeros(spec.shape[:-1], dtype='f')
    for idx, value in np.ndenumerate(iwv):
        nspec = spec[idx][value - pts:value + pts + 1]
        nvaxis = vaxis[value - pts:value + pts + 1]
        nspec = 1 - nspec / nspec[0]
        vel[idx] = np.sum(nvaxis * nspec) / np.sum(nspec)
    return vel
