import numpy as np


def refract4wardmodel(h, v, shotx, geox):
    """N-layer seismic refraction forward model

    Args:
        h: layer thicknesses, length n
        v: layer speeds, length n+1
        shotx: shot locations, length m
        geox: geophone locations, length m

    Returns:
        First arrival times at each geophone location
    """

    # Refraction travel times per-interface (
    tt = np.zeros((len(shotx), len(v)))

    # get travel times for direct wave
    tt[:, 0] = np.abs(geox - shotx) / v[0]

    # get travel times for other layers
    for i in range(len(v) - 1):
        iin = np.arcsin(v[i] / v[i + 1])  # incident/critical angle for interface i
        t0 = (2 * h[i] / np.cos(iin)) / v[i]  # travel time
        x0 = 2 * h[i] * np.tan(iin)  # critical distance
        for j in range(i - 1, -1, -1):  # Propagate critical angle up
            iin = np.arcsin(np.sin(iin) * (v[j] / v[j + 1]))
            t0 += (2 * h[j] / np.cos(iin)) / v[j]
            x0 += 2 * h[j] * np.tan(iin)

        mask = np.abs(geox - shotx) < x0
        tt[:, i + 1] = t0 + (np.abs(geox - shotx) - x0) / v[i + 1]
        tt[mask, i + 1] = np.inf  # Mask out points below critical distance

    return np.min(tt, axis=1)
