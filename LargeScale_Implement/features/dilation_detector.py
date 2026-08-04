import numpy as np
from scipy.ndimage import grey_dilation

try:
    import cv2
except ImportError:  # SciPy remains the portable fallback.
    cv2 = None

def pixelated_circle_footprint(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1")

    size = 2 * n + 1
    yy, xx = np.mgrid[-n:n+1, -n:n+1]
    return (xx**2 + yy**2 <= n**2).astype(np.uint8)

def detect_peaks(
    elevation: np.ndarray,
    n: int,
    flatness_eps: float = 1e-3,
    min_valid_fraction: float = 0.0,
    exclude_border: bool = False,
):
    """Detect maxima in finite portions of a possibly sparse elevation grid."""
    elevation = np.asarray(elevation)
    if elevation.ndim != 2:
        raise ValueError(f"elevation must be 2-D, got {elevation.shape}")
    if flatness_eps < 0.0:
        raise ValueError("flatness_eps must be non-negative")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between zero and one")

    footprint = pixelated_circle_footprint(n)
    finite = np.isfinite(elevation)
    if not finite.any():
        return np.empty((0, 2), dtype=np.int64)

    # Invalid cells must neither become candidates nor influence dilation.
    dilation_input = np.where(finite, elevation, -np.inf)
    if cv2 is not None and finite.all():
        # OpenCV's morphology is dramatically faster for native 0.25 m
        # rasters with a large circular footprint.
        dilated = cv2.dilate(
            dilation_input,
            footprint,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=-np.inf,
        )
    else:
        dilated = grey_dilation(dilation_input, footprint=footprint)
    unchanged = finite & np.isclose(
        dilated,
        elevation,
        atol=flatness_eps,
        rtol=0.0,
    )
    if exclude_border:
        unchanged[:n, :] = False
        unchanged[-n:, :] = False
        unchanged[:, :n] = False
        unchanged[:, -n:] = False

    if cv2 is not None and finite.all():
        eroded = cv2.erode(
            elevation,
            footprint,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=np.inf,
        )
        unchanged &= (dilated - eroded) > flatness_eps * 10
        peak_ij = np.argwhere(unchanged)
        return enforce_min_distance(
            peak_ij,
            elevation,
            min_dist_px=n,
        )

    peak_ij = np.argwhere(unchanged)

    keep = []
    H, W = elevation.shape
    for (i, j) in peak_ij:
        i0, i1 = max(0, i-n), min(H, i+n+1)
        j0, j1 = max(0, j-n), min(W, j+n+1)
        patch = elevation[i0:i1, j0:j1]
        patch_finite = np.isfinite(patch)

        footprint_i0 = i0 - (i - n)
        footprint_j0 = j0 - (j - n)
        patch_footprint = footprint[
            footprint_i0:footprint_i0 + (i1 - i0),
            footprint_j0:footprint_j0 + (j1 - j0),
        ].astype(bool)
        supported = patch_finite & patch_footprint
        support_fraction = supported.sum() / patch_footprint.sum()

        if support_fraction < min_valid_fraction or not supported.any():
            continue

        values = patch[supported]
        if np.ptp(values) > flatness_eps * 10:
            keep.append((i, j))

    if not keep:
        return np.empty((0, 2), dtype=np.int64)
    return enforce_min_distance(
        np.asarray(keep, dtype=np.int64),
        elevation,
        min_dist_px=n,
    )

def enforce_min_distance(peak_ij, elevation, min_dist_px):
    if len(peak_ij) == 0:
        return peak_ij
    elevs = elevation[peak_ij[:, 0], peak_ij[:, 1]]
    order = np.argsort(-elevs)
    kept = []
    spatial_bins = {}
    distance_squared = min_dist_px * min_dist_px
    for idx in order:
        p = peak_ij[idx]
        bin_i = int(p[0] // min_dist_px)
        bin_j = int(p[1] // min_dist_px)
        accept = True
        for offset_i in (-1, 0, 1):
            for offset_j in (-1, 0, 1):
                for kept_point in spatial_bins.get(
                    (bin_i + offset_i, bin_j + offset_j), ()
                ):
                    delta = p - kept_point
                    if int(delta @ delta) < distance_squared:
                        accept = False
                        break
                if not accept:
                    break
            if not accept:
                break
        if accept:
            kept.append(p)
            spatial_bins.setdefault((bin_i, bin_j), []).append(p)
    if not kept:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(kept, dtype=np.int64)

def detect_craters(
    elevation: np.ndarray,
    n: int,
    flatness_eps: float = 1e-3,
    min_valid_fraction: float = 0.0,
    exclude_border: bool = False,
):
    return detect_peaks(
        -elevation,
        n=n,
        flatness_eps=flatness_eps,
        min_valid_fraction=min_valid_fraction,
        exclude_border=exclude_border,
    )
