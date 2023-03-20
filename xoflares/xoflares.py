import numpy as np

from scipy import integrate

def _before_flare(time, tpeak, fwhm, ampl):
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    fout = (
        _fr[0]
        + _fr[1] * ((time - tpeak) / fwhm)
        + _fr[2] * ((time - tpeak) / fwhm) ** 2.0
        + _fr[3] * ((time - tpeak) / fwhm) ** 3.0
        + _fr[4] * ((time - tpeak) / fwhm) ** 4.0
    ) * ampl
    return fout

# reference implementation in numpy
def get_light_curvenp(time, tpeaks, fwhms, ampls, texp=None, oversample=7):
    time = np.asarray(time, dtype=float)

    tpeaks = np.atleast_1d(tpeaks)
    fwhms = np.atleast_1d(fwhms)
    ampls = np.atleast_1d(ampls)

    if texp is None:
        tgrid = time
    if texp is not None:
        # taking this oversample code from
        # https://github.com/dfm/exoplanet
        # and https://github.com/lkreidberg/batman
        texp = float(texp)
        oversample = int(oversample)
        oversample += 1 - oversample % 2
        dt = np.linspace(-texp / 2.0, texp / 2.0, oversample)
        tgrid = (dt + time.reshape(time.size, 1)).flatten()

    multiflare_lc = multiflaremodelnp(tgrid, tpeaks, fwhms, ampls)

    if texp is not None:
        multiflare_lc = np.mean(multiflare_lc.reshape(-1, oversample), axis=1)

    return multiflare_lc


def multiflaremodelnp(time, tpeaks, fwhms, ampls):
    time = np.asarray(time, dtype=float)
    tpeaks = np.atleast_1d(tpeaks)
    fwhms = np.atleast_1d(fwhms)
    ampls = np.atleast_1d(ampls)
    multiflare_lc = np.zeros_like(time)
    npeaks = tpeaks.shape[0]
    for i in range(npeaks):
        flare_lc = _flaremodelnp(time, tpeaks[i], fwhms[i], ampls[i])
        multiflare_lc = multiflare_lc + flare_lc
    return multiflare_lc


def _flaremodelnp(time, tpeak, fwhm, ampl):
    # reuses some code from AltaiPony and Apaloosa
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]
    flare_lc = np.zeros_like(time)
    flare_lc = np.where(
        (time <= tpeak) * ((time - tpeak) / fwhm > -1.0),
        (
            _fr[0]
            + _fr[1] * ((time - tpeak) / fwhm)
            + _fr[2] * ((time - tpeak) / fwhm) ** 2.0
            + _fr[3] * ((time - tpeak) / fwhm) ** 3.0
            + _fr[4] * ((time - tpeak) / fwhm) ** 4.0
        )
        * ampl,
        flare_lc,
    )
    flare_lc = np.where(
        (time > tpeak) * ((time - tpeak) / fwhm < 20.0),
        (
            _fd[0] * np.exp(((time - tpeak) / fwhm) * _fd[1])
            + _fd[2] * np.exp(((time - tpeak) / fwhm) * _fd[3])
        )
        * ampl,
        flare_lc,
    )
    return flare_lc


def get_flare_integral_numerical(
    time, tpeak, fwhm, ampl, texp=None, oversample=7
):
    """
    used in testing flare integral
    """
    feval = get_light_curvenp(time, tpeak, fwhm, ampl, texp, oversample)
    tstart, tend = time[feval > 0][0], time[feval > 0][-1]
    integral = integrate.quad(
        get_light_curvenp,
        tstart,
        tend,
        points=tpeak,
        args=(tpeak, fwhm, ampl, texp, oversample),
    )
    return integral


def _flareintegralnp(fwhm, ampl):
    t0, t1, t2 = -1, 0, 20
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    def get_int_before(x):
        integral = (
            _fr[0] * x
            + (_fr[1] * x ** 2 / 2)
            + (_fr[2] * x ** 3 / 3)
            + (_fr[3] * x ** 4 / 4)
            + (_fr[4] * x ** 5 / 5)
        )
        return integral

    def get_int_after(x):
        integral = (_fd[0] / _fd[1] * np.exp(_fd[1] * x)) + (
            _fd[2] / _fd[3] * np.exp(_fd[3] * x)
        )
        return integral

    before = get_int_before(t1) - get_int_before(t0)
    after = get_int_after(t2) - get_int_after(t1)
    return (before + after) * ampl * fwhm


def multiflareintegralnp(fwhms, ampls):
    fwhms = np.atleast_1d(fwhms)
    ampls = np.atleast_1d(ampls)
    npeaks = fwhms.shape[0]
    multiintegral = np.zeros_like(fwhms)
    for i in range(npeaks):
        multiintegral[i] = _flareintegralnp(fwhms[i], ampls[i])
    return multiintegral

