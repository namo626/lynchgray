"""
Analytic solutions for the Lynch and Gray (1978) test cases of the 2D
depth-averaged linear shallow water equations.

Python port of LG2D_Solutions.m (written by Ethan Kubatko).

References
----------
[1] Lynch, D. R. and Gray, W. G. (1978). Analytic Solutions for Computer
    Flow Model Testing. Journal of the Hydraulics Division, 104(HY10),
    1409-1428.
[2] Chen, C. L. (1989). Analytic Solutions for Tidal Model Testing.
    Journal of Hydraulic Engineering, 115(12), 1707-1714.

Note: [2] provides corrected solutions for the reverse quadratic
bathymetry cases (n = -2); the solutions originally published in [1]
are in error.
"""

import numpy as np
from scipy.special import jv, yv

_POLAR = {"Polar", "polar", "radial"}
_CARTESIAN = {"Cartesian", "cartesian", "Cart."}


def _get(data, key):
    """Read a parameter from a dict or from an object with attributes."""
    if isinstance(data, dict):
        return data[key]
    return getattr(data, key)


def lg2d_solutions(coords, t, data):
    """
    Compute the surface elevation and depth-averaged velocities for the
    Lynch and Gray test cases at the points ``coords`` and time ``t``.

    Parameters
    ----------
    coords : array_like, shape (N, 2) or (2,)
        x and y coordinates at which to compute the solution.
    t : float
        Time at which to compute the solution.
    data : dict or object
        Problem parameters, with keys/attributes:
          geometry : 'Cartesian' or 'Polar'
          geom1    : x1 or r1
          geom2    : x2 or r2
          h0       : bathymetric coefficient
          n        : bathymetric power (0, 1, 2 or -2)
          amp      : tidal amplitude of forcing
          freq     : tidal (angular) frequency of forcing
          phase    : tidal phase of forcing (radians)
          tau      : linear bottom friction factor
          g        : gravitational constant

    Returns
    -------
    SE : ndarray, shape (N,)
        Surface elevation.
    Ubar : ndarray, shape (N,)
        Depth-averaged velocity in the x direction.
    Vbar : ndarray, shape (N,)
        Depth-averaged velocity in the y direction.
    """
    SE_c, U_c, V_c = lg2d_complex_amplitudes(coords, data)
    eiwt = np.exp(1j * _get(data, "freq") * t)
    return np.real(SE_c * eiwt), np.real(U_c * eiwt), np.real(V_c * eiwt)


def lg2d_complex_amplitudes(coords, data):
    """
    Complex spatial amplitudes of the solution, i.e. the fields such that
    SE(x, t) = Re[SE_c(x) * exp(i*freq*t)], and likewise for Ubar, Vbar.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))

    geometry = _get(data, "geometry")
    g1 = float(_get(data, "geom1"))
    g2 = float(_get(data, "geom2"))
    h0 = _get(data, "h0")
    n = _get(data, "n")
    freq = _get(data, "freq")
    tau = _get(data, "tau")
    g = _get(data, "g")

    # Complex forcing amplitude, the constant beta, and i*omega
    amp = _get(data, "amp") * np.exp(-1j * _get(data, "phase"))
    beta = np.sqrt((freq**2 - 1j * freq * tau) / (g * h0) + 0j)
    iw = 1j * freq

    if geometry in _POLAR:
        # ------------------------------------------------------------
        # Radial coordinate test cases
        # ------------------------------------------------------------
        r1, r2 = g1, g2
        x, y = coords[:, 0], coords[:, 1]
        r = np.hypot(x, y)

        if n == 0:
            D = jv(0, beta*r2)*yv(1, beta*r1) - jv(1, beta*r1)*yv(0, beta*r2)
            A = amp * yv(1, beta*r1) / D
            B = -amp * jv(1, beta*r1) / D
            SE = A*jv(0, beta*r) + B*yv(0, beta*r)
            Ur = (-A*jv(1, beta*r) - B*yv(1, beta*r)) * iw / (beta*h0)

        elif n == 1:
            z1, z2, z = 2*beta*np.sqrt(r1), 2*beta*np.sqrt(r2), 2*beta*np.sqrt(r)
            D = jv(1, z2)*yv(2, z1) - jv(2, z1)*yv(1, z2)
            A = amp * np.sqrt(r2) * yv(2, z1) / D
            B = -amp * np.sqrt(r2) * jv(2, z1) / D
            SE = (A*jv(1, z) + B*yv(1, z)) / np.sqrt(r)
            Ur = (-A*jv(2, z) - B*yv(2, z)) / r * iw / (beta*h0)

        elif n == 2:
            s1 = -1 + np.sqrt(1 - beta**2)
            s2 = -1 - np.sqrt(1 - beta**2)
            D = s2*r2**s1*r1**s2 - s1*r1**s1*r2**s2
            A = amp * s2 * r1**s2 / D
            B = -amp * s1 * r1**s1 / D
            SE = A*r**s1 + B*r**s2
            Ur = (s1*A*r**(s1 - 1) + s2*B*r**(s2 - 1)) * iw / (beta**2*h0)

        elif n == -2:
            denom = np.cos(beta/2 * (r2**2 - r1**2))
            SE = amp * np.cos(beta/2 * (r**2 - r1**2)) / denom
            Ur = -amp * r * np.sin(beta/2 * (r**2 - r1**2)) / denom * iw / (beta*h0)

        else:
            raise ValueError(f"Unsupported bathymetric power n = {n}; use 0, 1, 2 or -2.")

        # Project radial velocity onto x and y (cos(theta) = x/r, sin(theta) = y/r)
        Ubar = Ur * x / r
        Vbar = Ur * y / r

    elif geometry in _CARTESIAN:
        # ------------------------------------------------------------
        # Cartesian coordinate test cases
        # ------------------------------------------------------------
        x1, x2 = g1, g2
        x = coords[:, 0]

        if n == 0:
            denom = np.cos(beta * (x2 - x1))
            SE = amp * np.cos(beta * (x - x1)) / denom
            Ubar = -iw * amp / (beta*h0) * np.sin(beta * (x - x1)) / denom

        elif n == 1:
            z1, z2, z = 2*beta*np.sqrt(x1), 2*beta*np.sqrt(x2), 2*beta*np.sqrt(x)
            D = jv(0, z2)*yv(1, z1) - yv(0, z2)*jv(1, z1)
            A = amp * yv(1, z1) / D
            B = -amp * jv(1, z1) / D
            SE = A*jv(0, z) + B*yv(0, z)
            Ubar = (-A*jv(1, z) - B*yv(1, z)) / np.sqrt(x) * iw / (beta*h0)

        elif n == 2:
            s1 = -0.5 + np.sqrt(0.25 - beta**2)
            s2 = -0.5 - np.sqrt(0.25 - beta**2)
            D = s2*x1**s2*x2**s1 - s1*x1**s1*x2**s2
            A = amp * s2 * x1**s2 / D
            B = -amp * s1 * x1**s1 / D
            SE = A*x**s1 + B*x**s2
            Ubar = (A*s1*x**(s1 - 1) + B*s2*x**(s2 - 1)) * iw / (beta**2*h0)

        elif n == -2:
            z1, z2, z = beta*x1**2/2, beta*x2**2/2, beta*x**2/2
            D = x2**1.5 * (jv(0.25, z1)*jv(0.75, z2) + jv(-0.75, z2)*jv(-0.25, z1))
            A = amp * jv(0.25, z1) / D
            B = amp * jv(-0.25, z1) / D
            SE = x**1.5 * (A*jv(0.75, z) + B*jv(-0.75, z))
            Ubar = x**2.5 * (A*jv(-0.25, z) - B*jv(0.25, z)) * iw / (beta*h0)

        else:
            raise ValueError(f"Unsupported bathymetric power n = {n}; use 0, 1, 2 or -2.")

        Vbar = np.zeros_like(Ubar)

    else:
        raise ValueError(f"Unknown geometry '{geometry}'; use 'Cartesian' or 'Polar'.")

    return SE, Ubar, Vbar


if __name__ == "__main__":
    # Example: quadratic bathymetry in a polar (quarter-annulus) domain
    data = dict(geometry="Polar", geom1=60960.0, geom2=152400.0,
                h0=1e-9, n=2,
                amp=0.3048, freq=2*np.pi/44712.0, phase=0.0,
                tau=1e-4, g=9.81)
    pts = np.column_stack([np.linspace(61000, 152400, 5), np.zeros(5)])
    SE, U, V = lg2d_solutions(pts, 0.0, data)
    print("SE  :", SE)
    print("Ubar:", U)
    print("Vbar:", V)
