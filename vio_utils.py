"""Shared primitives for the VIO pipeline.

Grouped by domain: geometric (projective / rigid-body algebra), probabilistic
(Bayesian filtering), and trajectory metrics (ATE / RTE evaluation). Anything
computed in more than one place belongs here rather than being re-derived.
"""

import numpy as np


# ============================================================================
# Geometric utils
# ============================================================================

def skew(t: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [t]x, so that [t]x @ v == cross(t, v).

    Args:
        t: 3-vector (any shape flattenable to 3).

    Returns:
        (3, 3) skew-symmetric matrix.
    """
    tx, ty, tz = np.asarray(t, dtype=float).flatten()
    return np.array([[0.0, -tz, ty],
                     [tz, 0.0, -tx],
                     [-ty, tx, 0.0]])


def pixels_to_bearings(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project pixels to camera-frame bearing vectors: p = K^-1 [u, v, 1]^T.

    Not unit-normalized — the third component stays 1, which is what the epipolar
    constraint and the pinhole re-projection both expect.

    Args:
        pixels: (N, 2) pixel coordinates.
        K: (3, 3) camera intrinsic matrix.

    Returns:
        (N, 3) bearing vectors in the camera frame.
    """
    pixels = np.asarray(pixels, dtype=float)
    homogeneous = np.hstack([pixels, np.ones((len(pixels), 1))])
    return (np.linalg.inv(K) @ homogeneous.T).T


def project_points(points_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project 3D points through a 3x4 projection matrix into pixels.

    Args:
        points_3d: (N, 3) points in the frame P is defined against.
        P: (3, 4) projection matrix (rectified P1 / P2 for a stereo pair).

    Returns:
        (N, 2) pixel coordinates.
    """
    points_3d = np.asarray(points_3d, dtype=float)
    homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    projected = P @ homogeneous.T
    return (projected[:2] / projected[2:3]).T


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Normalize each row to unit length, leaving degenerate rows at zero.

    Args:
        vectors: (N, D) array, or a single (D,) vector.

    Returns:
        New array of the same shape with unit-norm rows.
    """
    v = np.atleast_2d(np.asarray(vectors, dtype=float))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    out = np.divide(v, norms, out=np.zeros_like(v), where=norms > 1e-12)
    return out.reshape(np.shape(vectors))


def angle_between_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Angle between two vectors in degrees, 0.0 if either is degenerate.

    Args:
        vec_a: First 3-vector.
        vec_b: Second 3-vector.

    Returns:
        Angle in degrees, clipped to a valid arccos domain.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    cos_angle = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


# ============================================================================
# Probabilistic utils
# ============================================================================

def gaussian_pdf(x: float, mean: float, variance: float) -> float:
    """Univariate normal density N(x; mean, variance).

    Args:
        x: Sample point.
        mean: Distribution mean.
        variance: Distribution variance (must be positive).

    Returns:
        Probability density at x.
    """
    return float(
        (1.0 / np.sqrt(2.0 * np.pi * variance))
        * np.exp(-0.5 * (x - mean) ** 2 / variance)
    )


def gaussian_uniform_update(mu, sigma2, inlier_prob, z, tau2, p_uniform):
    """One Bayesian step of a Gaussian+uniform mixture filter (Vogiatzis / SVO).

    The state is p(z) = a*N(mu, sigma2) + (1-a)*U(range): the Gaussian carries the
    estimate, the uniform absorbs outliers. A measurement both Kalman-updates the
    Gaussian and re-weights a by how well it fits, so persistent disagreement drives
    a down instead of corrupting mu.

    Args:
        mu: Current Gaussian mean (inverse depth).
        sigma2: Current Gaussian variance.
        inlier_prob: Current mixture weight a on the Gaussian.
        z: Measurement (inverse depth).
        tau2: Measurement variance.
        p_uniform: Density of the uniform component over its range.

    Returns:
        (mu, sigma2, inlier_prob) after the update, unchanged if degenerate.
    """
    S = sigma2 + tau2
    residual = z - mu
    p_gauss = gaussian_pdf(z, mu, S)

    denom = inlier_prob * p_gauss + (1.0 - inlier_prob) * p_uniform
    if denom < 1e-30:
        return mu, sigma2, inlier_prob

    gain = sigma2 / S
    return (
        mu + gain * residual,
        (1.0 - gain) * sigma2,
        (inlier_prob * p_gauss) / denom,
    )


# ============================================================================
# Trajectory metrics utils
# ============================================================================

def umeyama_align(est: np.ndarray, gt: np.ndarray):
    """Rigid SE(3) alignment of an estimated trajectory onto ground truth.

    Scale is deliberately not estimated: a stereo+IMU system observes metric scale,
    so fitting it would mask real scale error.

    Args:
        est: (N, 3) estimated positions.
        gt: (N, 3) ground-truth positions.

    Returns:
        (R_align, t_align) mapping est into the ground-truth frame.
    """
    est_mean = est.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    H = (est - est_mean).T @ (gt - gt_mean)
    U, _, Vt = np.linalg.svd(H)
    # Reflection guard: force det(R) = +1 so the fit stays a rotation.
    sign_matrix = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)])
    R_align = Vt.T @ sign_matrix @ U.T
    return R_align, gt_mean - R_align @ est_mean


def compute_ate(est_positions, gt_positions):
    """Absolute Trajectory Error after rigid alignment — global accuracy.

    Args:
        est_positions: Sequence of estimated 3D positions.
        gt_positions: Sequence of ground-truth 3D positions.

    Returns:
        (ate_rmse, ate_errors, R_align, t_align); zeros if fewer than 3 poses.
    """
    est = np.asarray(est_positions, dtype=float)
    gt = np.asarray(gt_positions, dtype=float)
    if len(est) < 3:
        return 0.0, np.zeros(len(est)), np.eye(3), np.zeros(3)

    R_align, t_align = umeyama_align(est, gt)
    est_aligned = (R_align @ est.T).T + t_align
    ate_errors = np.linalg.norm(gt - est_aligned, axis=1)
    return float(np.sqrt(np.mean(ate_errors ** 2))), ate_errors, R_align, t_align


def compute_rte(est_positions, gt_positions, delta: int = 4):
    """Relative Trajectory Error over `delta`-pose segments — local drift.

    Complements ATE: because it compares relative displacements it needs no global
    alignment, so it stays sensitive to drift that alignment would absorb.

    Args:
        est_positions: Sequence of estimated 3D positions.
        gt_positions: Sequence of ground-truth 3D positions.
        delta: Segment length in poses.

    Returns:
        (rte_rmse, rte_errors) in meters; zeros if the trajectory is shorter than delta.
    """
    est = np.asarray(est_positions, dtype=float)
    gt = np.asarray(gt_positions, dtype=float)
    if len(est) <= delta:
        return 0.0, np.zeros(max(len(est) - delta, 0))

    rte_errors = np.linalg.norm(
        (est[delta:] - est[:-delta]) - (gt[delta:] - gt[:-delta]), axis=1
    )
    return float(np.sqrt(np.mean(rte_errors ** 2))), rte_errors
