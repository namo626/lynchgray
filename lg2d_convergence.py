"""
Mesh-convergence study for the Lynch and Gray test cases.

For each simulation output (NetCDF file containing `zeta` with dimensions
(points, time steps) and nodal coordinates `x`, `y`), this script computes
the L2 error of the surface elevation at the final snapshot against the
analytic solution from lg2d_solutions.py, then plots error versus mesh
spacing h on log-log axes and reports the observed convergence rates.

Usage: edit the CONFIGURATION block at the bottom of this file, or import
`run_convergence_study` from another script.
"""

import numpy as np
import matplotlib.pyplot as plt

from lg2d_solutions import lg2d_solutions


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def read_final_elevation(path, var="zeta"):
    """
    Return (x, y, zeta_final) from a NetCDF file, where zeta_final is the
    elevation at the last time step. Uses netCDF4 if installed (NetCDF3
    and NetCDF4/HDF5 files); otherwise falls back to scipy (NetCDF3 only).
    """
    try:
        from netCDF4 import Dataset
        with Dataset(path, "r") as nc:
            nc.set_auto_mask(True)
            x = np.ma.filled(nc.variables["x"][:].astype(float), np.nan)
            y = np.ma.filled(nc.variables["y"][:].astype(float), np.nan)
            z = nc.variables[var]
            npts = x.size
            # Expected (points, time); also accept (time, points)
            if z.shape[0] == npts:
                zf = z[:, -1]
            elif z.shape[-1] == npts:
                zf = z[-1, :]
            else:
                raise ValueError(f"{path}: '{var}' shape {z.shape} does not "
                                 f"match {npts} points")
            zf = np.ma.filled(np.ma.asarray(zf).astype(float), np.nan)
    except ImportError:
        from scipy.io import netcdf_file
        with netcdf_file(path, "r", mmap=False) as nc:
            x = np.array(nc.variables["x"][:], dtype=float)
            y = np.array(nc.variables["y"][:], dtype=float)
            zvar = nc.variables[var]
            z = np.array(zvar[:], dtype=float)
            fill = getattr(zvar, "_FillValue", None)
            npts = x.size
            if z.shape[0] == npts:
                zf = z[:, -1]
            elif z.shape[-1] == npts:
                zf = z[-1, :]
            else:
                raise ValueError(f"{path}: '{var}' shape {z.shape} does not "
                                 f"match {npts} points")
            if fill is not None:
                zf = np.where(zf == fill, np.nan, zf)

    return x.ravel(), y.ravel(), np.asarray(zf).ravel()


# ----------------------------------------------------------------------
# Error computation
# ----------------------------------------------------------------------
def l2_error(zeta_num, zeta_exact):
    """
    Discrete L2 (root-mean-square) error over the mesh nodes:

        e = sqrt( (1/N) * sum_i (zeta_num_i - zeta_exact_i)^2 )

    On quasi-uniform meshes this scales like the continuous L2 norm
    divided by sqrt(area), so it gives the same convergence rate.
    Nodes with missing values (NaN, e.g. dry or masked) are ignored.
    """
    diff = zeta_num - zeta_exact
    valid = np.isfinite(diff)
    if not np.any(valid):
        raise ValueError("No valid points to compute the error.")
    return np.sqrt(np.mean(diff[valid] ** 2))


def observed_rates(h, err):
    """Convergence rates between successive meshes: log(e_i/e_{i+1}) / log(h_i/h_{i+1})."""
    h, err = np.asarray(h, float), np.asarray(err, float)
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run_convergence_study(files, h, data, t_final, var="zeta",
                          ref_slopes=(1, 2), output="convergence_zeta.png",
                          show=True):
    """
    Parameters
    ----------
    files : list of str
        NetCDF result files, one per mesh.
    h : list of float
        Mesh spacing corresponding to each file.
    data : dict
        Problem parameters passed to lg2d_solutions.
    t_final : float
        Simulation time of the final snapshot stored in each file.
    var : str
        Name of the elevation variable in the NetCDF files.
    ref_slopes : tuple of float
        Slopes of the reference lines drawn on the plot.
    output : str or None
        Path to save the figure (None to skip saving).
    show : bool
        Whether to display the figure.

    Returns
    -------
    h, errors, rates : sorted mesh spacings (coarse to fine), L2 errors,
                       and observed rates between successive meshes.
    """
    if len(files) != len(h):
        raise ValueError("`files` and `h` must have the same length.")

    # Sort from coarsest to finest mesh
    order = np.argsort(h)[::-1]
    files = [files[i] for i in order]
    h = np.asarray(h, float)[order]

    errors = []
    for path, hi in zip(files, h):
        x, y, zeta_num = read_final_elevation(path, var)
        zeta_exact, _, _ = lg2d_solutions(np.column_stack([x, y]), t_final, data)
        errors.append(l2_error(zeta_num, zeta_exact))
    errors = np.array(errors)
    rates = observed_rates(h, errors)

    # Least-squares fit of log(e) = p*log(h) + c over all meshes
    p_fit, c_fit = np.polyfit(np.log(h), np.log(errors), 1)

    # ---- Report ----
    print(f"{'Mesh':>4} {'h':>14} {'L2 error':>14} {'Rate':>8}")
    for i, (hi, ei) in enumerate(zip(h, errors)):
        rate = f"{rates[i - 1]:8.3f}" if i > 0 else f"{'-':>8}"
        print(f"{i + 1:>4} {hi:14.6e} {ei:14.6e} {rate}")
    print(f"Least-squares fitted rate: {p_fit:.3f}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(h, errors, "o-", lw=2, ms=7,
              label=f"$L_2$ error (fitted rate = {p_fit:.2f})")

    # Annotate observed rates between successive meshes
    for i, r in enumerate(rates):
        hm = np.sqrt(h[i] * h[i + 1])
        em = np.sqrt(errors[i] * errors[i + 1])
        ax.annotate(f"{r:.2f}", (hm, em), textcoords="offset points",
                    xytext=(6, -12), fontsize=9)

    # Reference slopes anchored at the finest-mesh error
    linestyles = ["--", ":", "-."]
    for k, s in enumerate(ref_slopes):
        ref = errors[-1] * (h / h[-1]) ** s
        ax.loglog(h, ref, linestyles[k % len(linestyles)], color="gray",
                  label=f"$h^{s}$")

    ax.set_xlabel("Mesh spacing $h$")
    ax.set_ylabel(r"$L_2$ error in $\zeta$")
    ax.set_title(f"Elevation convergence at t = {t_final:g}")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300)
        print(f"Figure saved to {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return h, errors, rates


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION — edit these for your study
    # ------------------------------------------------------------------
    DATA = dict(
        geometry="Cartesian",    # 'Cartesian' or 'Polar'
        geom1=60000.0,       # x1 or r1
        geom2=150000.0,      # x2 or r2
        h0=3,             # bathymetric coefficient
        n=0,                 # bathymetric power: 0, 1, 2 or -2
        amp=0.3,          # tidal amplitude
        freq=2*np.pi/(12.4*3600),  # tidal angular frequency (rad/s)
        phase=0.0,           # tidal phase (rad)
        tau=0.005,            # linear bottom friction factor
        g=9.81,              # gravitational acceleration
    )
#    T_FINAL = 5 * 44712.0    # time of the final snapshot (s)
    T_FINAL = 5*86400; # seconds

    FILES = {
        "jump_llf":
        [
            "Cartesian/Mesh1/fort.63.nc",
            "Cartesian/Mesh2/fort.63.nc",
            "Cartesian/Mesh3/fort.63.nc",
            "Cartesian/Mesh4/fort.63.nc",
        ] ,
        "adc":
         [
            "Cartesian/Mesh1/fort.63.adc.nc",
            "Cartesian/Mesh2/fort.63.adc.nc",
            "Cartesian/Mesh3/fort.63.adc.nc",
            "Cartesian/Mesh4/fort.63.adc.nc",
         ],
        "nodal_avg":
         [
            "Cartesian/Mesh1/fort.63.dg2.nc",
            "Cartesian/Mesh2/fort.63.dg2.nc",
            "Cartesian/Mesh3/fort.63.dg2.nc",
            "Cartesian/Mesh4/fort.63.dg2.nc",
         ]
    }
    H = [15000.0, 7500.0, 3750.0, 1875.0]  # mesh spacing for each file
    #H = [15000.0, 7500.0, 3750.0]  # mesh spacing for each file

    for name, paths in FILES.items():
        run_convergence_study(paths, H, DATA, T_FINAL, output=f"convergence_{name}.png", show=False)
