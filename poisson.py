#!/usr/bin/env python
"""Function to compute Poission probability mass function (pmf), cummulative
distribution function (cdf) and their derivatives.

Notes
-----
I am not using the Poisson function in the Standard scipy.stats library
because that version of the poission function give
poisson.pmf(k=0, mu=0.0) = nan.

For our proposes, poisson.pmf(k=0, mu=0.0) should be 1.

"""

import math

def pmf(k, mu):
    """Poission probability mass function.

    """
    if not isinstance(k, int) or k < 0:
        raise TypeError("k %s must be an non-negative int." % k)
    if not isinstance(mu, float):
        raise TypeError("mu %s must be an float." % mu)


    if mu <= 0.0: 
        if k == 0:
            return 1.0 # use limit value as mu -> 0.0, k = 0
        else:
            return 0.0 # use limit value as mu -> 0.0, k > 0

    pmf = mu ** k
    pmf *= math.exp(-mu) 
    pmf /= math.factorial(k)

    return pmf

def dpmf_dmu(k, mu):
    """Derivation of the Poisson probability mass function with respect to
    mu.

        dP(X = k)
        --------- = (k/mu - 1) * P(X = k)
           dmu
    """
    if not isinstance(k, int) or k < 0:
        raise TypeError("k %s must be an non-negative int." % k)
    if not isinstance(mu, float):
        raise TypeError("mu %s must be an float." % mu)

    if mu <= 0.0:
        if k == 0:
            return -1.0 # use limit value as mu -> 0.0, k =0
        elif k == 1:
            return 1.0 # use limit value as mu -> 0.0, k = 1
        else:
            return 0.0 # use limit value as mu -> 0.0, k > 1

    deriv = ((float(k) / mu) - 1) * pmf(k, mu)

    return deriv

def cdf(kmax, mu):
    """Poisson cummulative distribution function.

    Sum of Poisson probiability mass function from k = 0, 1, 2 to kmax.
    """
    if not isinstance(kmax, int) or kmax < 0:
        raise TypeError("kmax %s must be an non-negative int." % kmax)
    if not isinstance(mu, float):
        raise TypeError("mu %s must be an float." % mu)

    cdf = 0
    for k in xrange(kmax + 1):
        cdf += pmf(k, mu)

    return cdf

def dcdf_dmu(kmax, mu):
    """Derivation of the Poisson cumulative distribution function mass 
    function with respect to mu.

    Derivative is sum from k = 0, 1, 2, 3, to kmax of:

        dP(X = k)
        --------- = (k/mu - 1) * P(X = k)
           dmu
    """
    if not isinstance(kmax, int) or kmax < 0:
        raise TypeError("kmax %s must be an non-negative int." % kmax)
    if not isinstance(mu, float):
        raise TypeError("mu %s must be an float." % mu)

    deriv = 0.0
    for k in xrange(kmax + 1):
        deriv += dpmf_dmu(k, mu)

    return deriv
