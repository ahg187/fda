import numpy as np

class Lfd(object):
    """Class for Linear Functional Differential Operator objects.
    
    Lfd creates a linear differential operator object of the form

    Lx(t) = w_0(t) x(t) + ... + w_{m-1}(t) D^{m-1}x(t) + \exp[w_m(t) D^m x(t)

    where nderiv = nderiv.

    Function x(t) is operated on by this operator L, and the operator computes a linear combination of the
    function and its first nderiv derivatives.  The function x(t) must be scalar.

    The linear combination of derivatives is defined by the weight or coefficient functions w_j(t),
    and these are assumed to vary over t, although of course they may also be constant as a special case.

    The weight coefficient for D^m is special in that it must be positive to properly identify the operator.
    This is why it is exponentiated.  In most situations, it will be 0, implying a weight of one, and this is the default.

    The inner products of the linear differential operator L applied to basis functions is evaluated in the functions
    called in function EVAL_PENALTY().

    Some important functions also have the capability of allowing the argument that is an LFD object be an integer.
    They convert the integer internally to an LFD object by INT2LFD().  These are:
         EVAL_FD()
         EVAL_MON()
         EVAL_POS()
         EVAL_BASIS()
         EVAL_PENALTY()


    Simple cases:

    All this generality may not be needed, and, for example, often the linear differential operator will be simply
    L = D^m, defining Lx(t) = D^mx(t).
    Or the weights and forcing functions may all have the same bases, in which case it is simpler to use a
    functional data objects to define them. These situations cannot be accommodated within Lfd(), but there is
    the unction int2Lfd(m) that converts a nonnegative integer nderiv into an Lfd object equivalent to D^m.
    There is also fd2cell(fdobj) and that converts a functional data object into cell object, which can then be used as
    an argument of Lfd().
    """

    def __init__(self, nderiv=0, bwtcell=None):
        """

        :param nderiv: The highest order of the derivative in operator L.
        :param bwtcell: A list object with either NDERIV or NDERIV+1 cells
            If there are NDERIV cells, then the coefficient of D^NDERIV is set to 1;
            otherwise, cell NEDRIV+1 contains a function that is exponentiated to define the actual coefficient.
        """

        # check NDERIV

        if not isinstance(nderiv, int):
            raise ValueError("Order of operator is not an integer.")

        if nderiv < 0:
            raise ValueError("Order of operator is negative.")

        # check BWTCELL

        if not isinstance(bwtcell, list):
            raise ValueError("BWTCELL is not a list object.")

        if bwtcell is None:
            if nderiv > 0:
                raise ValueError("Positive operator order is accompanied by empty BWTCELL.")
        else:

