"""
Astro 599 Homework 2: Fully implement an IMF concept
====================================================

Assignment
----------

**The requirements are **
    * Computing the expected number of stars per mass bin
    * Computing the mass enclosed in a given mass range
    * Being able to draw random masses from an IMF
    * What is the average mass predicted by an IMF?

Plot at least a couple of mass functions on the same figure

Draw a random sample of N masses from one mass function and show that the
sample follows the desired distribution


File Content
------------

This python file implements *one* solution to the IMF class creation and using
inheritance to define multiple functions from the literature.

Note: the implementation is of course not unique.

:author: Morgan Fouesneau
:email: mfouesn@uw.edu
"""
import numpy as np


class IMF(object):
    """
    Here is one possible implementation, which
    let you define an IMF as multiple power-laws
    """
    def __init__(self, nI, x, a, massMin=0.1, massMax=120., name=None):
        """__init__ - constructor of the class

        keywords
        --------
        nI: int
            number of definition intervals

        x:  iterable
            interval edges (of len nI + 1)

        a:  iterable
            power-law indexes (of len nI) in units of dN/dM (Salpeter corresponds to -2.35)

        massMin: float
            minimal mass

        massMax:
            maximal mass

        name: string
            optional name

        notes
        -----
        1 - the mass range can be restricted by massMin, massMax while keeping
            the official definition.
        2 - indexes of the different power-laws are assumed to be in units of
            dN/dM, ie, -2.35 corresponds to a Salpeter IMF. However, sometimes
            you could find -1.35, which corresponds to an index defined in
            terms of dN/dlog(M).
        """
        # Store relevant information
        # ==========================
        self.nIMFbins  = nI
        self.massinf   = np.asarray(x)
        self.slope     = np.asarray(a)
        self.name      = name
        self.coeffcont = np.zeros(np.size(a))
        self.massMin   = massMin
        self.massMax   = massMax

        # Make functional form
        # ====================
        # the first step is to build the functional form of the IMF:
        # i.e., make a continuous function and compute the normalization

        # continuity
        # ----------
        # given by:  c[i-1] * x[i] ^ a[i-1] = c[i] * x[i] ^ a[i]
        # arbitrary choice for the first point, which will be corrected by the
        # normalization step.
        self.coeffcont[0] = 1.
        for i in range(1, nI):
            self.coeffcont[i]  = (self.coeffcont[i - 1])
            self.coeffcont[i] *= (x[i] ** (a[i - 1] - a[i]))

        # normalize
        # ---------
        # depends on the adpoted definition of the IMF indexes:
        # either dN/dM or dN/dlog(M). In this example we consider that indexes
        # are given in units of dN/dM.
        # get the norm : integral(imf(M) dM) = 1 seen as a prob dist. funct."""
        self.norm = 1.  # will be updated at the next line
        self.norm = self.get_enclosed_Nstar(self.massMin, self.massMax)

        # Compute the average mass
        self.avg = self.get_avg_mass(self.massMin, self.massMax)

    def get_enclosed_Nstar(self, Mmin, Mmax):
        """get_enclosed_Nstar - Get the enclosed dN over a given mass range.
        Analytic integration, Sum(imf(m) dm)
        Note: no extrapolation outside the original mass range definition of the IMF.

        keywords
        --------
        Mmin, Mmax: float, float
            lower and upper masses

        returns
        -------
        r: float
            enclosed dN within [Mmin, Mmax]
        """
        x = self.massinf
        a = self.slope
        c = self.coeffcont

        # will be useful in the integration
        b = a + 1.

        val = 0.
        # analytical integration of a power law
        for i in range(0, self.nIMFbins):
            if (Mmin < x[i + 1]) & (Mmax > x[i]):
                if x[i] <= Mmin:
                    x0 = Mmin
                else:
                    x0 = x[i]
                if x[i + 1] <= Mmax:
                    x1 = x[i + 1]
                else:
                    x1 = Mmax
                # careful if the index is 1
                if a[i] == 1:
                    S = c[i] * (np.log(x1) - np.log(x0))
                else:
                    S = c[i] / b[i] * ( (x1) ** (b[i]) - (x0) ** (b[i]) )
                val += S

        return val * self.norm

    def get_enclosed_mass(self, Mmin, Mmax):
        """get_enclosed_mass - Get the enclosed mass over a given mass range.
        Analytic integration, integral(m * imf(m) dm)

        Note: no extrapolation outside the original mass range definition of the IMF.

        keywords
        --------
        Mmin, Mmax: float, float
            lower and upper masses

        returns
        -------
        r: float
            enclosed mass within [Mmin, Mmax]
        """
        x = self.massinf
        a = self.slope
        c = self.coeffcont

        # will be useful in the integration
        b = a + 2.

        val = 0.
        # analytical integration of a power law
        for i in range(0, self.nIMFbins):
            if (Mmin < x[i + 1]) & (Mmax > x[i]):
                if x[i] <= Mmin:
                    x0 = Mmin
                else:
                    x0 = x[i]
                if x[i + 1] <= Mmax:
                    x1 = x[i + 1]
                else:
                    x1 = Mmax

                # careful if the index is 1
                if a[i] == 2:
                    S = c[i] * (np.log(x1) - np.log(x0))
                else:
                    S = c[i] / b[i] * ( (x1) ** (b[i]) - (x0) ** (b[i]) )
                val += S

        return val * self.norm

    def getValue(self, m):
        """getValue - returns the value of the normalized IMF at a given mass m
            IMF(m) = 1 / norm * c * m ** a
            and integral( IMF(m) dm ) = 1

        keywords
        --------
        m: float or iterable of floats
            masses at which evaluate the function

        returns
        -------
        r: float or ndarray(dtype=float)
            evaluation of the function (normalized imf)
        """
        # if m is iterable
        if getattr(m, '__iter__', False):
            return np.asarray([ self.getValue(mk) for mk in m])
        else:
            # extrapolation
            if (m > self.massMax) or (m < self.massMin):
                return 0.
            # exact value exists
            elif m in self.massinf[:-1]:
                ind = np.where(m == self.massinf)
                return float(float(self.coeffcont[ind]) / self.norm_M * m ** self.slope[ind])
            # otherwise do the evaluation with the correct interval
            else:
                i = 0
                if self.nIMFbins > 1:
                    while m > self.massinf[i]:
                        i += 1
                    i -= 1
                if self.massinf[i] > self.massMax:
                    return 0.
                else:
                    return float(float(self.coeffcont[i]) / self.norm * m ** self.slope[i])

    def get_avg_mass(self, Mmin, Mmax):
        """ get the avg mass over a given range
              < M > = integral(M * imf * dM) / integral(imf * dM)

        :param Mmin: float, lower mass
        :param Mmax: float, upper mass
        """
        return self.get_enclosed_mass(Mmin, Mmax) / self.get_enclosed_Nstar(Mmin, Mmax)

    def random(self, N, massMin=None, massMax=None):
        """random - Draw mass samples from this distribution
        Samples are distributed over the interval [massMin, massMax]
        Interval is truncated to the IMF range definition if it extents beyond it. (taken as is otherwise)

        keywords
        --------

        N: int
            size of the sample

        massMin: float
            lower mass (default self.massMin)

        massMax: float
            upper mass (default self.massMax)

        returns
        -------

        r: ndarray(dtype=float)
            returns an array of random masses

        method
        ------

        drawing random numbers from a given distribution can be done using
        multiple methods. When you can compute or accurately estimate the
        integral of your function, the optimal way is to use it, aka "Inverse
        transform sampling". Briefly:

        let's call F(x) = integral( imf(m) dm, m=massMin..x)

        Since we use power laws, the intregral is trivial. if x such as
        M[i] <= x < M[i+1], with M the masses used to define the IMF:

        F(x) = F(M[i]) + 1 / norm * 1 / (a[i] + 1) ( x ** (a[i] + 1) - M[i] ** (a[i] + 1) )

        The trick is to observe that F(x) varies between 0 and 1, and that
        there is a unique mapping between F(x) and x (F is a bijective
        function).
        I spare the proof but we can demonstrate that drawing uniform F(x)
        numbers between 0 and 1 will then give us x values that exactly follow
        the imf.

        In our case, the previous equation is inversible, and once you extract
        x as a function of F(x) you're done.
        """

        # check keyword values, default is the IMF definition
        massMin = massMin or self.massMin
        massMax = massMax or self.massMax

        beta = self.slope + 1.

        # compute the cumulative distribution values at each mass interval edge
        F = np.zeros(self.nIMFbins + 1)
        F[-1] = 1.0
        for i in range(1, self.nIMFbins):
            F[i] = F[i - 1] + 1. / self.norm * (self.coeffcont[i - 1] / (beta[i - 1])) * ( self.massinf[i] ** (beta[i - 1]) - self.massinf[i - 1] ** (beta[i - 1]) )

        #find intervals of massMin and massMax
        for k in range(self.nIMFbins):
            if massMin >= self.massinf[k]:
                mink = k
            if massMax >= self.massinf[k]:
                maxk = k
        if massMin < self.massMin:
            Fmin = 0.
        elif massMin >= self.massMax:
            return
        else:
            i = mink
            Fmin = F[i] + 1. / self.norm * ( self.coeffcont[i] / (beta[i]) ) * ( massMin ** (beta[i]) - self.massinf[i] ** (beta[i]) )

        if massMax >= self.massMax:
            Fmax = 1.0
        elif massMax < self.massMin:
            return
        else:
            i = maxk
            Fmax = F[i] + 1. / self.norm * ( self.coeffcont[i] / (beta[i]) ) * ( massMax ** (beta[i]) - self.massinf[i] ** (beta[i]) )

        x = np.random.uniform(Fmin, Fmax, np.int64(N))
        y = np.zeros(np.int64(N))
        for k in range(self.nIMFbins):
            ind = np.where((x >= F[k]) & (x < F[k + 1]))
            if len((ind)[0]) > 0:
                y[ind] = self.massinf[k] * ( 1. + (x[ind] - F[k]) / (self.massinf[k] ** (beta[k])) * ( beta[k] ) * self.norm / self.coeffcont[k]) ** (1. / (beta[k]))
        return y

    def __call__(self, m=None):
        """__call__ - make a callable object (function like)

        keywords
        --------

        m: float or iterable or None
            if None calls self.info()
            else calls self.getValue(m)
        """
        if m is None:
            return self.info()
        return self.getValue(m)

    def info(self):
        """ prints a quick summary of the functional object """
        txt = """IMF: {s.name}, IMF(m) = 1./norm * c * m ** a
        nI = {s.nIMFbins},
        norm = {s.norm}, 1/norm = {invnorm},
        Average mass = {s.avg},
        m[] = {s.massinf},
        a[] = {s.slope},
        c[] = {s.coeffcont}"""
        print(txt.format(s=self, invnorm=1. / self.norm))


# Deriving common IMF from the literature
#=========================================

# in the definitions below, power-law indexes are given for the dN/dlog(M)
# definition and accordingly converted before usage

class Kennicutt(IMF):
    def __init__(self):
        nI = 2
        x = [0.1, 1., 120.]
        a = [-0.4, -1.5]
        a = np.asarray(a) - 1
        massMin = 0.1
        massMax = 120.
        IMF.__init__(self, nI, x, a, massMin, massMax, name='Kennicutt')


class Kroupa2001(IMF):
    def __init__(self):
        nI = 4
        x = [0.01, 0.08, 0.5, 1., 120.]
        a = [0.7, -0.3, -1.3, -1.3]
        a = np.asarray(a) - 1
        massMin = 0.01
        massMax = 120.
        IMF.__init__(self, nI, x, a, massMin, massMax, name='Kroupa 2001')


class Kroupa93(IMF):
    def __init__(self):
        nI = 3
        x = [0.1, 0.5, 1., 120.]
        a = [-0.3, -1.2, -1.7]
        a = np.asarray(a) - 1
        massMin = 0.1
        massMax = 120.
        IMF.__init__(self, nI, x, a, massMin, massMax, name='Kroupa 1993')


class Salpeter(IMF):
    def __init__(self):
        nI = 1
        x = [0.1, 120.]
        a = [-1.35]
        a = np.asarray(a) - 1
        massMin = 0.1
        massMax = 120.
        IMF.__init__(self, nI, x, a, massMin, massMax, name='Salpeter')


class MillerScalo(IMF):
    def __init__(self):
        nI = 3
        x = [0.1, 1., 10., 120.]
        a = [-0.4, -1.5, -2.3]
        a = np.asarray(a) - 1
        massMin = 0.1
        massMax = 120.
        IMF.__init__(self, nI, x, a, massMin, massMax, name='Miller & Scalo')


class Scalo98(IMF):
    def __init__(self):
        nI = 3
        x = [0.1, 1., 10., 120.]
        a = [-0.2, -1.7, -1.3]
        a = np.asarray(a) - 1
        massMin = 0.1
        massMax = 120.
        IMF.__init__(self, nI, x, a, massMin, massMax, name='Scalo 1998')


class Scalo86(IMF):
    def __init__(self):
        nI = 24
        x = [    1.00000000e-01,   1.10000000e-01,   1.40000000e-01,
                 1.80000000e-01,   2.20000000e-01,   2.90000000e-01,
                 3.60000000e-01,   4.50000000e-01,   5.40000000e-01,
                 6.20000000e-01,   7.20000000e-01,   8.30000000e-01,
                 9.80000000e-01,   1.17000000e+00,   1.45000000e+00,
                 1.86000000e+00,   2.51000000e+00,   3.47000000e+00,
                 5.25000000e+00,   7.94000000e+00,   1.20200000e+01,
                 1.82000000e+01,   2.69200000e+01,   4.16900000e+01,
                 1.20000000e+02     ]
        a = [  3.2   ,  2.455,  2.   ,  0.3  ,  0.   ,  0.   , -0.556, -1.625,
               -1.833, -1.286,  1.5  , -1.857,  0.   , -2.333, -3.455, -1.692,
               -2.571, -1.722, -1.611, -1.667, -2.333, -1.353, -0.947, -1.778]
        a = np.asarray(a) - 1
        massMin = 0.1
        massMax = 120.
        IMF.__init__(self, nI, x, a, massMin, massMax, name='Scalo 1986')
