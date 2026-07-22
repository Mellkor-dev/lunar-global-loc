import numpy as np
from scipy.ndimage import grey_dilation

def pixelated_circle_footprint(n: int) -> np.ndarray:
    
    size = 2 * n + 1
    yy, xx = np.mgrid[-n:n+1, -n:n+1]
    return (xx**2 + yy**2 <= n**2).astype(np.uint8)

def detect_peaks(elevation: np.ndarray, n: int, flatness_eps: float = 1e-3):
    
    footprint = pixelated_circle_footprint(n)
    dilated = grey_dilation(elevation, footprint=footprint)
    unchanged = np.isclose(dilated, elevation, atol=flatness_eps)

    peak_ij = np.argwhere(unchanged)

    keep = []
    half = n
    H, W = elevation.shape
    for (i, j) in peak_ij:
        i0, i1 = max(0, i-half), min(H, i+half+1)
        j0, j1 = max(0, j-half), min(W, j+half+1)
        patch = elevation[i0:i1, j0:j1]
        if patch.max() - patch.min() > flatness_eps * 10:
            keep.append((i, j))

    return enforce_min_distance(np.array(keep), elevation, min_dist_px=n)

def enforce_min_distance(peak_ij, elevation, min_dist_px):
    if len(peak_ij) == 0:
        return peak_ij
    elevs = elevation[peak_ij[:, 0], peak_ij[:, 1]]
    order = np.argsort(-elevs)
    kept = []
    for idx in order:
        p = peak_ij[idx]
        if all(np.linalg.norm(p - k) >= min_dist_px for k in kept):
            kept.append(p)
    return np.array(kept)

def detect_craters(elevation: np.ndarray, n: int, flatness_eps: float = 1e-3):
    
    return detect_peaks(-elevation, n=n, flatness_eps=flatness_eps)
