# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import quad, quad_vec, dblquad
from scipy.special import erf, erfc

from .boreholes import Borehole


def finite_line_source(
        time, alpha, borehole1, borehole2, reaSource=True, imgSource=True):
    """
    Evaluate the Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution, as proposed by Claesson and Javed [#FLS-ClaJav2011]_
    and extended to boreholes with different vertical positions by Cimmino and
    Bernier [#FLS-CimBer2014]_. The FlS solution is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{1}{2H_2}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
            e^{-d_{12}^2s^2}(I_{real}(s)+I_{imag}(s))ds


            I_{real}(s) &= erfint((D_2-D_1+H_2)s) - erfint((D_2-D_1)s)

            &+ erfint((D_2-D_1-H_1)s) - erfint((D_2-D_1+H_2-H_1)s)

            I_{imag}(s) &= erfint((D_2+D_1+H_2)s) - erfint((D_2+D_1)s)

            &+ erfint((D_2+D_1+H_1)s) - erfint((D_2+D_1+H_2+H_1)s)


            erfint(X) &= \\int_{0}^{X} erf(x) dx

                      &= Xerf(X) - \\frac{1}{\\sqrt{\\pi}}(1-e^{-X^2})

        .. Note::
            The reciprocal thermal response factor
            :math:`h_{2\\rightarrow1}(t)` can be conveniently calculated by:

                .. math::
                    h_{2\\rightarrow1}(t) = \\frac{H_2}{H_1}
                    h_{1\\rightarrow2}(t)

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    borehole1 : Borehole object or list of Borehole objects, length (N)
        Borehole object of the borehole extracting heat.
    borehole2 : Borehole object or list of Borehole objects, length (M)
        Borehole object for which the FLS is evaluated.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool
        True if the image part of the FLS solution is to be included.
        Default is True.

    Returns
    -------
    h : float or array, shape (M, N, K), (M, N) or (K)
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h

    Notes
    -----
    The function returns a float if time is a float and borehole1 and borehole2
    are Borehole objects. If time is a float and any of borehole1 and borehole2
    are lists, the function returns an array, shape (M, N), If time is an array
    and borehole1 and borehole2 are Borehole objects, the function returns an
    array, shape (K).If time is an array and any of borehole1 and borehole2 are
    are lists, the function returns an array, shape (M, N, K).

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> h = gt.heat_transfer.finite_line_source(4*168*3600., 1.0e-6, b1, b2)
    h = 0.0110473635393

    References
    ----------
    .. [#FLS-ClaJav2011] Claesson, J., & Javed, S. (2011). An analytical
       method to calculate borehole fluid temperatures for time-scales from
       minutes to decades. ASHRAE Transactions, 117(2), 279-288.
    .. [#FLS-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.

    """
    if isinstance(borehole1, Borehole) and isinstance(borehole2, Borehole):
        if borehole1.is_vertical() and borehole2.is_vertical():
            # Unpack parameters
            dis = borehole1.distance(borehole2)
            H1, D1 = borehole1.H, borehole1.D
            H2, D2 = borehole2.H, borehole2.D
            # Integrand of the finite line source solution
            f = _finite_line_source_integrand(
                dis, H1, D1, H2, D2, reaSource, imgSource)

            # Evaluate integral
            if isinstance(time, (np.floating, float)):
                # Lower bound of integration
                a = 1.0 / np.sqrt(4.0*alpha*time)
                h = quad(f, a, np.inf)[0]
            else:
                h = np.stack(
                    [quad(f, 1.0 / np.sqrt(4.0*alpha*t), np.inf)[0] for t in time],
                    axis=-1)
        elif borehole1.is_parallel(borehole2) and reaSource:
            dx = borehole2.x - borehole1.x
            dy = borehole2.y - borehole1.y
            dis = np.sqrt(dx**2 + dy**2)
            beta = borehole1.tilt
            theta = np.arctan2(dy, dx) - borehole1.orientation
            deltaD = (borehole2.D - borehole1.D)
            b11 = Borehole(borehole1.H, 0., borehole1.r_b, 0., 0.)
            D2 = dis*np.cos(theta)*np.sin(beta) + deltaD*np.cos(beta)
            x2 = dis*np.cos(theta)*np.cos(beta) - deltaD*np.sin(beta)
            y2 = dis*np.sin(theta)
            b22 = Borehole(borehole2.H, D2, borehole2.r_b, x2, y2)
            h = finite_line_source(time, alpha, b11, b22, imgSource=False) + finite_line_source(time, alpha, borehole1, borehole2, reaSource=False, imgSource=imgSource)
        elif borehole1.image().is_parallel(borehole2) and imgSource:
            h = finite_line_source(time, alpha, borehole1, borehole2, reaSource=reaSource, imgSource=False) - finite_line_source(time, alpha, borehole1.image(), borehole2, imgSource=False)
        else:
            # Unpack parameters
            sameSource = borehole1.is_equal(borehole2)
            dis = borehole1.distance(borehole2)
            x1, y1 = borehole1.position()
            H1, D1 = borehole1.H, borehole1.D
            beta1, theta1 = borehole1.tilt, borehole1.orientation
            x2, y2 = borehole2.position()
            H2, D2 = borehole2.H, borehole2.D
            beta2, theta2 = borehole2.tilt, borehole2.orientation
            # Integrand of the finite line source solution
            gamma = 0.5/np.sqrt(alpha*time)

            # Evaluate integral
            if isinstance(time, (np.floating, float)):
                f = _finite_line_source_inclined_integrand(
                    dis, x1, y1, H1, D1, beta1, theta1, x2, y2, H2, D2, beta2, theta2,
                    gamma, reaSource, imgSource, sameSource)
                h = dblquad(f, 0., H1, 0., H2)[0]
            else:
                h = np.stack(
                    [dblquad(
                        _finite_line_source_inclined_integrand(
                            dis, x1, y1, H1, D1, beta1, theta1,
                            x2, y2, H2, D2, beta2, theta2,
                            g, reaSource, imgSource, sameSource),
                        0., H1, 0., H2)[0] for g in gamma],
                    axis=-1)
    else:
        if isinstance(borehole1, Borehole): borehole1 = [borehole1]
        if isinstance(borehole2, Borehole): borehole2 = [borehole2]
        if np.all([b1.is_vertical() for b1 in borehole1]) and np.all([b2.is_vertical() for b2 in borehole2]):
            # Unpack parameters
            dis = np.array(
                [[b1.distance(b2) for b1 in borehole1] for b2 in borehole2])
            D1 = np.array([b.D for b in borehole1]).reshape(1, -1)
            H1 = np.array([b.H for b in borehole1]).reshape(1, -1)
            D2 = np.array([b.D for b in borehole2]).reshape(-1, 1)
            H2 = np.array([b.H for b in borehole2]).reshape(-1, 1)

            # Evaluate integral
            h = finite_line_source_vectorized(
                time, alpha, dis, H1, D1, H2, D2,
                reaSource=reaSource, imgSource=imgSource)
        else:
            h = np.array([[finite_line_source(time, alpha, b1, b2, reaSource=reaSource, imgSource=imgSource) for b1 in borehole1] for b2 in borehole2])
    return h


def finite_line_source_vectorized(
        time, alpha, dis, H1, D1, H2, D2, reaSource=True, imgSource=True):
    """
    Evaluate the Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution, as proposed by Claesson and Javed
    [#FLSVec-ClaJav2011]_ and extended to boreholes with different vertical
    positions by Cimmino and Bernier [#FLSVec-CimBer2014]_. The FlS solution
    is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{1}{2H_2}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
            e^{-d_{12}^2s^2}(I_{real}(s)+I_{imag}(s))ds


            I_{real}(s) &= erfint((D_2-D_1+H_2)s) - erfint((D_2-D_1)s)

            &+ erfint((D_2-D_1-H_1)s) - erfint((D_2-D_1+H_2-H_1)s)

            I_{imag}(s) &= erfint((D_2+D_1+H_2)s) - erfint((D_2+D_1)s)

            &+ erfint((D_2+D_1+H_1)s) - erfint((D_2+D_1+H_2+H_1)s)


            erfint(X) &= \\int_{0}^{X} erf(x) dx

                      &= Xerf(X) - \\frac{1}{\\sqrt{\\pi}}(1-e^{-X^2})

        .. Note::
            The reciprocal thermal response factor
            :math:`h_{2\\rightarrow1}(t)` can be conveniently calculated by:

                .. math::
                    h_{2\\rightarrow1}(t) = \\frac{H_2}{H_1}
                    h_{1\\rightarrow2}(t)

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool
        True if the image part of the FLS solution is to be included.
        Default is True.

    Returns
    -------
    h : float
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h

    Notes
    -----
    This is a vectorized version of the :func:`finite_line_source` function
    using scipy.integrate.quad_vec to speed up calculations. All arrays
    (dis, H1, D1, H2, D2) must follow numpy array broadcasting rules. If time
    is an array, the integrals for different time values are stacked on the
    last axis.
    

    References
    ----------
    .. [#FLSVec-ClaJav2011] Claesson, J., & Javed, S. (2011). An analytical
       method to calculate borehole fluid temperatures for time-scales from
       minutes to decades. ASHRAE Transactions, 117(2), 279-288.
    .. [#FLSVec-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.

    """
    # Integrand of the finite line source solution
    f = _finite_line_source_integrand(
        dis, H1, D1, H2, D2, reaSource, imgSource)

    # Evaluate integral
    if isinstance(time, (np.floating, float)):
        # Lower bound of integration
        a = 1.0 / np.sqrt(4.0*alpha*time)
        h = quad_vec(f, a, np.inf)[0]
    else:
        h = np.stack(
            [quad_vec(f, 1.0 / np.sqrt(4.0*alpha*t), np.inf)[0]
             for t in time],
            axis=-1)
    return h


def _erfint(x):
    """
    Integral of the error function.

    Parameters
    ----------
    x : float or array
        Argument.

    Returns
    -------
    float or array
        Integral of the error function.

    """
    return x * erf(x) - 1.0/np.sqrt(np.pi) * (1.0-np.exp(-x**2))


def _finite_line_source_integrand(dis, H1, D1, H2, D2, reaSource, imgSource):
    """
    Integrand of the finite line source solution.

    Parameters
    ----------
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
    imgSource : bool
        True if the image part of the FLS solution is to be included.

    Returns
    -------
    f : callable
        Integrand of the finite line source solution. Can be vector-valued.

    Notes
    -----
    All arrays (dis, H1, D1, H2, D2) must follow numpy array broadcasting
    rules.

    """
    if reaSource and imgSource:
        # Full (real + image) FLS solution
        p = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        q = np.stack([D2 - D1 + H2,
                      D2 - D1,
                      D2 - D1 - H1,
                      D2 - D1 + H2 - H1,
                      D2 + D1 + H2,
                      D2 + D1,
                      D2 + D1 + H1,
                      D2 + D1 + H2 + H1],
                     axis=-1)
        f = lambda s: 0.5 / (H2*s**2) * np.exp(-dis**2*s**2) * np.inner(p, _erfint(q*s))
    elif reaSource:
        # Real FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.stack([D2 - D1 + H2,
                      D2 - D1,
                      D2 - D1 - H1,
                      D2 - D1 + H2 - H1],
                     axis=-1)
        f = lambda s: 0.5 / (H2*s**2) * np.exp(-dis**2*s**2) * np.inner(p, _erfint(q*s))
    elif imgSource:
        # Image FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.stack([D2 + D1 + H2,
                      D2 + D1,
                      D2 + D1 + H1,
                      D2 + D1 + H2 + H1],
                     axis=-1)
        f = lambda s: 0.5 / (H2*s**2) * np.exp(-dis**2*s**2) * np.inner(p, _erfint(q*s))
    else:
        # No heat source
        f = lambda s: 0.
    return f

def _finite_line_source_inclined_integrand(
        dis, x1, y1, H1, D1, beta1, theta1, x2, y2, H2, D2, beta2, theta2,
        gamma, reaSource, imgSource, sameSource):

    dx = lambda v, u: (x1 + u*np.sin(beta1)*np.cos(theta1)) - (x2 + v*np.sin(beta2)*np.cos(theta2))
    dy = lambda v, u: (y1 + u*np.sin(beta1)*np.sin(theta1)) - (y2 + v*np.sin(beta2)*np.sin(theta2))
    if reaSource and imgSource:
        # Full (real + image) FLS solution
        dzImg = lambda v, u: (D1 + u*np.cos(beta1)) + (D2 + v*np.cos(beta2))
        disImg = lambda v, u: np.sqrt(dx(v,u)**2 + dy(v,u)**2 + dzImg(v,u)**2)
        if sameSource:
            disRea = lambda v, u: np.sqrt(dis**2 + (u - v)**2)
        else:
            dzRea = lambda v, u: (D1 + u*np.cos(beta1)) - (D2 + v*np.cos(beta2))
            disRea = lambda v, u: np.sqrt(dx(v,u)**2 + dy(v,u)**2 + dzRea(v,u)**2)
        f = lambda v, u: (0.5*erfc(gamma*disRea(v,u))/disRea(v,u) - 0.5*erfc(gamma*disImg(v,u))/disImg(v,u))/H2
    elif reaSource:
        # Real FLS solution
        if sameSource:
            disRea = lambda v, u: np.sqrt(dis**2 + (u - v)**2)
        else:
            dzRea = lambda v, u: (D1 + u*np.cos(beta1)) - (D2 + v*np.cos(beta2))
            disRea = lambda v, u: np.sqrt(dx(v,u)**2 + dy(v,u)**2 + dzRea(v,u)**2)
        f = lambda v, u: 0.5/H2*erfc(gamma*disRea(v,u))/disRea(v,u)
    elif imgSource:
        # Image FLS solution
        dzImg = lambda v, u: (D1 + u*np.cos(beta1)) + (D2 + v*np.cos(beta2))
        disImg = lambda v, u: np.sqrt(dx(v,u)**2 + dy(v,u)**2 + dzImg(v,u)**2)
        f = lambda v, u: -0.5/H2*erfc(gamma*disImg(v,u))/disImg(v,u)
    else:
        # No heat source
        f = lambda v, u: 0.
    return f
