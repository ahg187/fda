import numpy as np
import scipy.sparse as sp
from fda.utils.validation import check_array


class basis(object):
    """Base class for basis objects. Use specific subclasses to create an actual basis object."""

    def __init__(self, basistype=None, rangeval=[0, 1], nbasis=2, dropind=None, quadvals=None, values=None,
                 basisvalues=None):
        """

        :param basistype: Basis type.
        :param rangeval: Array of length 2 containing the lower and upper boundaries for the rangeval of argument values.
            If a single value is input, it must be positive and the lower limit is set to 0.
        :param nbasis: Number of basis functions.
        :param dropind: A set of indices in 1:NBASIS to drop when basis functions are arguments. .
        :param quadvals: A NQUAD by 2 matrix.  The first column contains quadrature points to be used in a fixed
            point quadrature. The second contains quadrature weights.  For example, for Simpson's rule for
            NQUAD = 7, the points are equally spaced and the weights are delta.*[1, 4, 2, 4, 2, 4, 1]/3.
            DELTA is the spacing between quadrature points. The default is QUADVALS = None.
        :param values: A cell array, with cells containing the values of the basis function derivatives starting
            with 0 and going up to the highest derivative needed.  The values correspond to quadrature points in
            QUADVALS. It is up to the user to multiply the derivative values by the square roots of the quadrature
            weights so as to make numerical integration a simple matrix multiplication. Values are checked against
            QUADVALS to ensure the correct number of rows, and against NBASIS to ensure the correct number of columns.
            The default is VALUES = None.
        :param basisvalues: A cell array.  The cell array must be 2-dimensional, with a variable number of rows and
            two or more columns. This field is designed to avoid evaluation of a basis system repeatedly at a set of
            argument values. Each row corresponds to a specific set of argument values. The first cell in that row
            contains the argument values. The second cell in that row contains a matrix of values of the basis
            functions. The third and subsequent cells contain matrices of values their derivatives up to a maximum
            derivative order. Whenever function getbasismatrix is called, it checks the first cell in each row to see,
            first, if the number of argument values corresponds to the size of the first dimension, and if this test
            succeeds, checks that all of the arguments match. This takes time, of course, but is much faster than
            re-evaluation of the basis system. Even this time can be avoided by direct retrieval of the desired array.

        :return: Object of class basis()
        """

        self.type = basistype
        self.rangeval = rangeval
        self.nbasis = nbasis
        self.dropind = dropind
        self.quadvals = quadvals
        self.values = values
        self.basisvalues = basisvalues


class bspline_basis(basis):
    """Creates a basis object of type B-spline."""

    def __init__(self, nbasis=5, norder=4, nbreaks=None, breaks=None, rangeval=[0, 1], **kwargs):
        """
        :param nbasis: Number of basis functions.
        :param norder: Order of B-spline basis (one higher than their degree). Default of norder=4 gives cubic B-splines.
        :param breaks: Piecewise increasing sequence of junction points between piecewise polynomial segments.
        :param kwargs: Other arguments to be passed to basis.__init__()

        :return: Object of class bspline_basis()
        """

        super().__init__(basistype='bspline', **kwargs)

        # check RANGEVAL
        if type(rangeval) == int:
            if rangeval <= 0:
                raise ValueError("RANGEVAL is a single value that is not positive.")
            rangeval = [0, rangeval]
        elif len(rangeval) == 2:
            if rangeval[1] <= rangeval[0]:
                raise ValueError("RANGEVAL is not a legitimate range.")
        else:
            raise ValueError("RANGEVAL is not a legitimate range.")

        # convert to column array
        rangeval = check_array(rangeval)

        # check BREAKS
        if breaks is not None:
            if np.min(np.diff(breaks)) < 0:
                raise ValueError("One or more BREAKS differences are negative.")

        # case of empty NBASIS and empty BREAKS: set up splines with a single interior knot (nbreaks = 3)
        if nbasis is None and breaks is None:
            nbasis = 5
            norder = 4
            nbreaks = 3
            breaks = np.linspace(rangeval[0, 0], rangeval[0, 1], nbreaks)

        # deal with inconsistencies of arguments
        if breaks is not None:
            nbreaks = len(breaks)
            nbasis = nbreaks + norder - 2

        if breaks is None and nbasis is not None:
            nbreaks = nbasis - norder + 2
            breaks = np.linspace(rangeval[0, 0], rangeval[0, 1], nbreaks)

        # convert to column array
        breaks = check_array(breaks)

        # check the compatibility of NBASIS, NBREAKS and RANGEVAL
        if nbreaks < 2:
            raise ValueError("Number of values in BREAKS less than 2.")

        if nbasis < nbreaks - 1:
            raise ValueError("NBASIS is less than number of values=BREAKS.")

        if breaks[0, 0] != rangeval[0, 0]:
            raise ValueError("Smallest value in BREAKS not equal to RANGEVAL[0].")

        if breaks[0, nbreaks - 1] != rangeval[0, 1]:
            raise ValueError("Largest value in BREAKS not equal to RANGEVAL[1].")

        self.nbasis = nbasis
        self.norder = norder
        self.nbreaks = nbreaks
        self.breaks = breaks
        self.rangeval = rangeval

    def _getbasismatrix(self, evalargs, nderiv=0, sparsewrd=0):
        """
        :param evalargs: Array of values at which basis matrix BASISMAT is to be evaluated.
        :param nderiv: Order of derivative required. Default is 0.
        :param sparsewrd: If 1, return in sparse form.

        :return: Basis matrix BASISMAT (len(evalargs) x nbasis) evaluated at EVALARGS
        """

        # check for stored basis matrix
        # tbd

        breaks = self.breaks
        nbreaks = self.nbreaks
        norder = self.norder
        dropind = self.dropind

        # check NDERIV
        if nderiv < 0:
            raise ValueError('NDERIV is negative.')
        if nderiv >= norder:
            raise ValueError('NDERIV cannot be as large as order of B-Spline.')

        # check EVALARGS
        evalargs = check_array(evalargs)
        n = evalargs.shape[1]

        if evalargs.shape[0] != 1:
            raise ValueError('EVALARGS is not a vector.')

        if np.min(np.diff(evalargs)) < 0:
            isrt = np.argsort(evalargs)
            evalargs = evalargs[:, isrt.ravel()]
            sortwrd = 1
        else:
            sortwrd = 0

        if evalargs[0, 0] - breaks[0, 0] < -1e-10 or evalargs[0, n - 1] - breaks[0, nbreaks - 1] > 1e-10:
            raise ValueError('EVALARGS out of range.')

        if norder == 1:
            bsplinemat = np.zeros((n, nbreaks - 1))

            for ibreaks in range(1, nbreaks):
                ind = np.where((evalargs >= breaks[0, ibreaks - 1]) & (evalargs <= breaks[0, ibreaks]))
                bsplinemat[ind[1], ibreaks - 1] = 1

            if sparsewrd:
                bsplinemat = sp.coo_matrix(bsplinemat)

        # set abbriviations
        k = norder  # order of splines
        x = evalargs
        km1 = k - 1
        nb = nbreaks  # number of break points
        nx = n  # number of argument values
        nd = nderiv + 1  # ND is order of derivative plus one
        ns = nb - 2 + k  # number splines to compute
        if ns < 1:
            bsplinemat = None
            self.basismat = bsplinemat
            raise ValueError('There are no B-splines for the given input.')

        onenx = np.ones((1, nx))
        onenb = np.ones((1, k))
        onens = np.ones((1, ns))

        # augment break sequence to get knots by adding a K-fold knot at each end
        knots = np.hstack((breaks[0, 0] * np.ones((1, km1)), breaks, breaks[0, nb - 1] * np.ones((1, km1))))
        nbasis = knots.shape[1] - k

        # for each i, determine left(i) so that K <= left(i) < nbasis+1, and, within that restriction,
        # knots(left(i)) <= pts(i) < knots(left(i)+1)
        knotslower = knots[:, :nbasis]
        index = np.argsort(np.hstack((knotslower, x)))
        pointer = check_array(np.where(index > nbasis - 1)[1] - np.arange(1, nx + 1, 1))
        left = np.max(np.vstack((pointer, onenx * km1)), axis=0).astype(int)

        # compute bspline values and derivatives, if needed:

        # initialize the b array
        temp = check_array([1] + km1 * [0])
        b = np.resize(temp, (nd * nx, k)).astype(float)
        nxs = nd * (np.arange(nx) + 1) - 1

        # run the recurrence simultaneously for all x(i)

        # bring it up to the intended level

        for j in range(k - nd):
            saved = np.zeros((1, nx))
            for r in range(j + 1):
                leftpr = left + r + 1
                tr = knots[0, leftpr] - x
                tl = x - knots[0, leftpr - j - 1]
                term = b[nxs, r] / (tr + tl)
                b[nxs, [r] * len(nxs)] = (saved + tr * term)
                saved = tl * term
            b[nxs, [j + 1] * len(nxs)] = saved

        # save the bspline values in successive blocks in b

        for jj in range(nd - 1):
            j = k - nd + jj
            saved = np.zeros((1, nx))
            nxn = nxs - 1
            for r in range(j + 1):
                leftpr = left + r + 1
                tr = knots[0, leftpr] - x
                tl = x - knots[0, leftpr - j - 1]
                term = b[nxs, r] / (tr + tl)
                b[nxn, [r] * len(nxn)] = (saved + tr * term)
                saved = tl * term
            b[nxn, [j + 1] * len(nxn)] = saved
            nxs = nxn

        # now use the fact that derivative values can be obtained by differencing

        for jj in range(nd - 2, -1, -1):
            j = k - jj - 1
            temp = np.dot(check_array(np.arange(jj+1, nd)).T, onenx) + np.dot(np.ones((nd - jj - 1, 1)), nxn.reshape(1, nx))
            nxs = np.asarray(temp.reshape(((nd - 1 - jj) * nx, 1), order='F').ravel(), dtype=int)
            for r in range(j - 1, -1, -1):
                leftpr = left + r + 1
                temp = np.ones((nd - jj - 1, 1)) * (knots[0,leftpr] - knots[0,leftpr-j]) / j
                b[nxs,[r]*len(nxs)] = -b[nxs,r]/temp.ravel()
                b[nxs,[r+1]*len(nxs)] = b[nxs,r+1] - b[nxs,r]


        # finally, zero out all rows in b that correspond to x outside the basic interval [breaks[0,0],...,breaks[0,nb-1]]

        index = np.where((x < breaks[0,0]) | (x > breaks[0,nb-1]))[1]
        if index.size != 0:
            temp = np.dot(check_array(np.arange(1-nd,1)).T, np.ones((1,len(index)))) + np.dot((nd)*np.ones((nd,1)), check_array(index+1)) - 1
            temp = np.asarray(temp.reshape((nd*len(index),1)).ravel(), dtype=int)
            b[temp,:] = np.zeros((nd * len(index), k))


        # setup output matrix bsplinemat

        width = max(ns, nbasis) + km1 + km1
        cc = np.zeros((nx*width,))
        index = np.dot(check_array(np.arange(1-nx,1)).T, onenb) + nx*(np.dot(check_array(left+1).T, onenb) + np.dot(onenx.T, check_array(np.arange(-km1,1))) ) - 1
        index = np.asarray(index, dtype=int)
        cc[index] = b[nd * np.arange(1,nx+1) - 1,:]
        index = np.dot(check_array(np.arange(1-nx,1)).T, onens) + nx * np.dot(onenx.T, check_array(np.arange(1,ns+1))) - 1
        index = np.asarray(index, dtype=int)
        bsplinemat = cc[index].reshape(nx,ns)

        if sortwrd:
            bsplinemat = bsplinemat[isrt.ravel(),:]

        # store in sparse mode if required

        if sparsewrd:
            bsplinemat = sp.coo_matrix(bsplinemat)

        self.evalargs = evalargs
        self.basismat = bsplinemat

    #def eval(self, evalargs, nderiv=0, sparsewrd=0):
