from numpy import var, mean, sum
from scipy import stats

__all__ = ['intraclass']


def intraclass(y, icc_type, alpha=0.05, r0=0):
    """
    Intraclass correlation

    Parameters
    ----------
    y : numpy.ndarray
        (N, K) array of N samples/targets or K raters/measurements.
    icc_type : str
        The type of ICC to perform. See notes
    alpha : float, optional
        Statistical significance level. Default is 0.05
    r0 : float, optional
        Value to test the intraclass correlation coefficient against. Default is 0

    Returns
    -------
    r : float
        Intraclass correlation coefficient
    LB : float
        Lower bound of the confidence interval with alpha level of significance
    UB : float
        Upper bound of the confidence interval with alpha level of significance
    F : float
        F-statistic for the test that r = r0
    df1 : float
        1st degrees of freedom for the test that r = r0
    df2 : float
        2nd degrees of freedom for the test that r = r0
    p : float
        p-value for the test that r = r0

    Notes
    -----
    ICC types:
        1-1 : One-way random effects, absolute agreement, single rater/measurement, or ICC(1, 1)
        1-k : One-way random effects, absolute agreement, multiple raters/measurements, or ICC(1, k)
        C-1 : Two-way mixed effects, consistency, single rater/measurement. Either no notation, or ICC(3, 1)
        C-k : Two-way mixed effects, consistency, multiple raters/measurements. Either no notation, or ICC(3, k)
        A-1 : Two-way mixed effects, absolute agreement, single rater/measurement. Either ICC(2, 1) or no notation
        A-k : Two-way mixed effects, absolute agreement, multiple raters/measurements. Either ICC(2, k) or no notation.

    For more on the typics and designations, see `Koo and Li, 2015`

    References
    ----------
    McGraw, K. O., S. P. Wong. "Forming Inferences About Some Intraclass Correlation Coefficients".
        Psychological Methods. Vol. 1, No. 1. pp. 30-46. 1996
    Koo, T. K., M. Y. Li. "A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability
        Research". Journal of Chiropractic Medicine. Vol. 15, pp. 155-163. 2015.
    """
    n, k = y.shape

    ss_total = var(y, ddof=1) * (n * k - 1)
    msr = var(mean(y, axis=1), ddof=1) * k
    msw = sum(var(y, ddof=1, axis=1)) / n
    msc = var(mean(y, axis=0), ddof=1) * n
    mse = (ss_total - msr * (n - 1) - msc * (k - 1)) / ((n - 1) * (k - 1))

    if icc_type == '1-1':
        result = _icc_11(msr, msw, n, k, alpha=alpha, r0=r0)
    elif icc_type == '1-k':
        result = _icc_1k(msr, msw, n, k, alpha=alpha, r0=r0)
    elif icc_type == 'C-1':
        result = _icc_c1(msr, mse, n, k, alpha=alpha, r0=r0)
    elif icc_type == 'C-k':
        result = _icc_ck(msr, mse, n, k, alpha=alpha, r0=0)
    elif icc_type == 'A-1':
        result = _icc_a1(msr, mse, msc, n, k, alpha=alpha, r0=r0)
    elif icc_type == 'A-k':
        result = _icc_ak(msr, mse, msc, n, k, alpha=alpha, r0=r0)
    else:
        raise ValueError('icc_type is not recognized')

    return result


def _icc_11(ms_r, ms_w, n_target, n_rater, alpha=0.05, r0=0):
    r = (ms_r - ms_w) / (ms_r + (n_rater - 1) * ms_w)
    F = (ms_r / ms_w) * (1 - r0) / (1 + (n_rater - 1) * r0)
    df1 = n_target - 1
    df2 = n_target * (n_rater - 1)
    p = 1 - stats.f.cdf(F, df1, df2)

    FL = (ms_r / ms_w) / stats.f.ppf(1 - alpha / 2, df1, df2)
    FU = (ms_r / ms_w) * stats.f.ppf(1 - alpha / 2, df2, df1)

    LB = (FL - 1) / (FL + (n_rater - 1))
    UB = (FU - 1) / (FU + (n_rater - 1))

    return r, LB, UB, F, df1, df2, p


def _icc_1k(ms_r, ms_w, n_target, n_rater, alpha=0.05, r0=0):
    r = (ms_r - ms_w) / ms_r
    F = (ms_r / ms_w) * (1 - r0)
    df1 = n_target - 1
    df2 = n_target * (n_rater - 1)
    p = 1 - stats.f.cdf(F, df1, df2)

    FL = (ms_r / ms_w) / stats.f.ppf(1 - alpha / 2, n_target - 1, n_target * (n_rater - 1))
    FU = (ms_r / ms_w) * stats.f.ppf(1 - alpha / 2, n_target * (n_rater - 1), n_target - 1)
    LB = 1 - 1 / FL
    UB = 1 - 1 / FU
    return r, LB, UB, F, df1, df2, p


def _icc_c1(ms_r, ms_e, n_target, n_rater, alpha=0.05, r0=0):
    r = (ms_r - ms_e) / (ms_r + (n_rater - 1) * ms_e)
    F = (ms_r / ms_e) * (1 - r0) / (1 + (n_rater - 1) * r0)
    df1 = n_target - 1
    df2 = (n_target - 1) * (n_rater - 1)
    p = 1 - stats.f.cdf(F, df1, df2)
    FL = (ms_r / ms_e) / stats.f.ppf(1 - alpha / 2, n_target - 1, (n_target - 1) * (n_rater - 1))
    FU = (ms_r / ms_e) * stats.f.ppf(1 - alpha / 2, (n_target - 1) * (n_rater - 1), n_target - 1)
    LB = (FL - 1) / (FL + (n_rater - 1))
    UB = (FU - 1) / (FU + (n_rater - 1))
    return r, LB, UB, F, df1, df2, p


def _icc_ck(ms_r, ms_e, n_target, n_rater, alpha=0.05, r0=0):
    r = (ms_r - ms_e) / ms_r
    F = (ms_r / ms_e) * (1 - r0)
    df1 = n_target - 1
    df2 = (n_target - 1) * (n_rater - 1)
    p = 1 - stats.f.cdf(F, df1, df2)
    FL = (ms_r / ms_e) / stats.f.ppf(1 - alpha / 2, n_target - 1, (n_target - 1) * (n_rater - 1))
    FU = (ms_r / ms_e) * stats.f.ppf(1 - alpha / 2, (n_target - 1) * (n_rater - 1), n_target - 1)
    LB = 1 - 1 / FL
    UB = 1 - 1 / FU
    return r, LB, UB, F, df1, df2, p


def _icc_a1(ms_r, ms_e, ms_c, n_target, n_rater, alpha=0.05, r0=0):
    r = (ms_r - ms_e) / (ms_r + (n_target - 1) * ms_e + n_target * (ms_c - ms_e) / n_target)

    a = (n_rater * r0) / (n_target * (1 - r0))
    b = 1 + (n_target * r0 * (n_target - 1)) / (n_target * (1 - r0))
    F = ms_r / (a * ms_c + b * ms_e)

    a = n_rater * r / (n_target * (1 - r))
    b = 1 + n_rater * r * (n_target - 1) / (n_target * (1 - r))
    df1 = n_target - 1
    df2 = (a * ms_c + b * ms_e)**2 / ((a * ms_c)**2 / (n_rater - 1) + (b * ms_e)**2 / ((n_target - 1) * (n_rater - 1)))

    p = 1 - stats.f.cdf(F, df1, df2)

    Fs = stats.f.ppf(1 - alpha / 2, n_target - 1, df2)
    LB = n_target * (ms_r - Fs * ms_e) / (Fs * (n_rater * ms_c + (n_rater * n_target - n_rater - n_target) * ms_e)
                                          + n_target * ms_r)
    Fs = stats.f.ppf(1 - alpha / 2, df2, n_target - 1)
    UB = n_target * (Fs * ms_r - ms_e) / (n_rater * ms_c + (n_rater * n_target - n_rater - n_target) * ms_e
                                          + n_target * Fs * ms_r)
    return r, LB, UB, F, df1, df2, p


def _icc_ak(ms_r, ms_e, ms_c, n_target, n_rater, alpha=0.05, r0=0):
    r = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n_target)

    c = r0 / (n_target * (1 - r0))
    d = 1 + (r0 * (n_target - 1)) / (n_target * (1 - r0))
    F = ms_r / (c * ms_c + d * ms_e)

    a = n_rater * r / (n_target * (1 - r))
    b = 1 + n_rater * r * (n_target - 1) / (n_target * (1 - r))
    df1 = n_target - 1
    df2 = (a * ms_c + b * ms_e)**2 / ((a * ms_c)**2 / (n_rater - 1) + (b * ms_e)**2 / ((n_target - 1) * (n_rater - 1)))

    p = 1 - stats.f.cdf(F, df1, df2)

    Fs = stats.f.ppf(1 - alpha / 2, df1, df2)
    LB = n_target * (ms_r - Fs * ms_e) / (Fs * (ms_c - ms_e) + n_target * ms_r)
    Fs = stats.f.ppf(1 - alpha / 2, df2, df1)
    UB = n_target * (Fs * ms_r - ms_e) / (ms_c - ms_e + n_target * Fs * ms_r)
    return r, LB, UB, F, df1, df2, p
