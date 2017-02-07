import numpy as np
from fda.basis.basis import constant_basis

class fd(object):
    """
    Base class for functional data objects.

    A functional data object consists of a basis for expanding a functional observation and
    a set of coefficients defining this expansion.
    """

    def __init__(self, coef=None, basisobj=None, fdnames=None):
        """
        :param coef:  An array containing coefficient values for the expansion of each set of function values, i.e.
            terms of a set of basis functions.
            If COEF is a 3d-array, then the first dimension corresponds to basis functions, the second to replications,
            and the third to variables.
            If COEF is a 2d array, it is assumed that there is only one variable per replication and then rows
            correspond to basis functions and columns correspond to replications.
            If COEF is a 1d array, it is assumed that there is only one replication and one variable.

        :param basisobj: An object of class 'basis'. If this argument is missing, then a B-spline basis is created
            with the number of basis functions equal to the size of the first dimension of COEF.

        :param fdnames: A list of length 3 with elements:
            1. a name for the argument domain, such as 'Time'
            2. a name for the replications ro cases
            3. a name for the function
            If this argument is missing, the list ['arguments', 'replications', 'functions'] is used.

            Each of the list elements may itself be a list of length 2 in which case the first element contains the
            name as above for the dimension of the data, and the second element contains a character array of names
            for each index value. Note that the rows must be of the same length
        """

        # set default fdnames

        defaultfdnames = []
        defaultfdnames.append('arguments')
        defaultfdnames.append('replications')
        defaultfdnames.append('functions')

        # define the default fd object

        if coef is None and basisobj is None and fdnames is None:
            self.coef = np.zeros((1,1))
            self.basisobj = constant_basis(rangeval=[0,1])
            self.fdnames = defaultfdnames

        # set default arguments

        if fdnames is None:
            fdnames = defaultfdnames

        # check COEF

        if not isinstance(coef, np.ndarray):
            raise ValueError("Coefficients are not stored as an array.")

        if coef is None or coef.size = 0:
            raise ValueError("Coefficient array is empty.")
        else:
            coefd = coef.shape
            ndim = length(coefd)
            if ndim > 3:
                raise ValueError("Coefficient array has more than three dimensions.")

        # check BASISOBJ

        if not isinstance(basisobj, basis):
            raise ValueError("BASISOBJ is not of class basis.")

        nbasis = basisobj.nbasis

        if coefd[0] != nbasis:
            raise ValueError("First dimension of coefficient array is not equal to number of basis functions.")

        # check FDNAMES

        if not isinstance(fdnames, list):
            raise ValueError("FDNAMES is not a list object.")

        if len(fdnames) > 3:
            raise ValueError("FDNAMES has length greater 3.")

        # fill missing fdnames elements with default labels

        if len(fdnames) == 2:
            fdnames.append('functions')
        if len(fdnames) == 1:
            fdnames.append('replications')
            fdnames.append('functions')

        self.coef = coef
        self.basisobj = basisobj
        self.fdnames = fdnames



