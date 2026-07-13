"""
Gymnasium environment wrapping the tightly-coupled VIO pipeline.

The RL agent observes map statistics and keypoint information at each frame,
then selects tunable parameters for the next VIO step.  The reward is derived
from pose error in a sliding window (Umeyama-aligned) plus runtime penalties.

Runs the VIO pipeline **synchronously** (no threads) for deterministic training.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

import cv2

sys.path.append(os.path.dirname(__file__))

import gtsam
from data_manager import DataManager
from v_frontend import vFeature
from imu_pipeline import IMUPipeline, IMUCalibration, IMUSample
from vio_optimizer import GraphOptimizer, X, V, B, L


# ---------------------------------------------------------------------------
# Action catalogue — each tunable is a separate Categorical axis
# ---------------------------------------------------------------------------
# 1. Parallax threshold for landmark promotion (degrees)
PARALLAX_OPTIONS      = [1.0, 1.5, 2.5, 3.5, 5.0]
# 2. Minimum tracked features before triggering new detection
MIN_FEATURES_OPTIONS  = [200, 300, 400, 500, 700, 1000]
# 3. Minimum incremental parallax for re-observations (degrees)
MIN_OBS_PARALLAX_OPTIONS = [0.5, 1.0, 2.0, 3.0, 5.0]
# 4. Landmark regularization sigma (meters) — how tightly landmarks are anchored
LANDMARK_REG_OPTIONS  = [0.5, 1.0, 1.5, 2.5, 4.0]

ACTION_DIMS = [
    len(PARALLAX_OPTIONS),
    len(MIN_FEATURES_OPTIONS),
    len(MIN_OBS_PARALLAX_OPTIONS),
    len(LANDMARK_REG_OPTIONS),
]

# Observation vector size (fixed part, excluding variable keypoints)
MAP_STATS_DIM = 15
MAX_KEYPOINTS_FOR_OBS = 200  # Subsample tracked keypoints to this count
KEYPOINT_FEAT_DIM = 4        # (u_norm, v_norm, depth_inv, track_age_norm)
FLAT_KP_DIM = MAX_KEYPOINTS_FOR_OBS * KEYPOINT_FEAT_DIM

OBS_DIM = MAP_STATS_DIM + FLAT_KP_DIM


def _umeyama_align(est: np.ndarray, gt: np.ndarray):
    """SE(3) Umeyama alignment (rotation + translation, NO scale).

    Matches main_threaded.py's compute_ate — correct for stereo+IMU
    which has absolute scale. Scale estimation would hide real drift.
    """
    n = len(est)
    if n < 3:
        return est, np.zeros(n)
    mu_e = est.mean(0)
    mu_g = gt.mean(0)
    ce = est - mu_e
    cg = gt - mu_g
    H = ce.T @ cg
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = mu_g - R @ mu_e
    aligned = (R @ est.T).T + t
    errors = np.linalg.norm(gt - aligned, axis=1)
    return aligned, errors


class VIOEnv(gym.Env):
    """Gymnasium env that runs the VIO pipeline one frame per step()."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir: str = "f:/Code/exercise_10/data/MH_01_easy/mav0",
        frame_step: int = 10,
        start_frame: int = 400,
        max_episode_frames: int = 200,
        reward_window: int = 5,
        matcher_type: str = "xfeat",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.frame_step = frame_step
        self.start_frame = start_frame
        self.max_episode_frames = max_episode_frames
        self.reward_window = reward_window
        self.matcher_type = matcher_type

        # Action: MultiDiscrete
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)

        # Observation: flat vector (map_stats + flattened keypoints)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Will be created in reset()
        self._data_manager: Optional[DataManager] = None
        self._feature_pipeline: Optional[vFeature] = None
        self._imu_pipeline: Optional[IMUPipeline] = None
        self._optimizer: Optional[GraphOptimizer] = None
        self._frame_iter = None

        # Tracking for reward
        self._est_positions = []
        self._gt_positions = []
        self._frame_count = 0
        self._prev_cam_ts = None
        self._P1 = None
        self._initial_bias = None
        self._T_cam_body = None
        self._R_cam_body = None
        self._R_body_cam = None

        # Last observation cache (for info)
        self._last_n_tracked = 0
        self._last_n_landmarks = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # --- Data ---
        dm = DataManager(self.data_dir)
        dm.load_data()
        self._data_manager = dm

        K1, dist1, _ = dm.load_camera_calib("cam0")
        K2, dist2, _ = dm.load_camera_calib("cam1")
        baseline = dm.get_baseline(K1, K2)

        # --- Frontend ---
        fp = vFeature(
            matcher_type=self.matcher_type,
            baseline=baseline,
            intrinsics=K1,
            dist_coeffs=dist1,
            T_cam1_cam0=dm.T_cam1_cam0,
            device="cpu",
        )
        fp.initialize()
        self._feature_pipeline = fp

        # --- IMU + Optimizer ---
        imu_calib = IMUCalibration()
        self._imu_pipeline = IMUPipeline(imu_calib)

        self._optimizer = GraphOptimizer(
            use_isam=True,
            body_P_sensor=dm.T_imu_cam0,
            imu_calib=imu_calib,
        )

        # Gravity-aligned initial orientation (matching main_threaded.py)
        # Use static accelerometer data to find pitch/roll alignment to gravity
        from scipy.spatial.transform import Rotation as R_scipy
        static_samples = int(3.0 * 200)  # 3 seconds at 200Hz
        static_accel = dm.imu_df[["a_x", "a_y", "a_z"]].values[:static_samples]
        up_body = static_accel.mean(axis=0)
        up_body /= np.linalg.norm(up_body)
        initial_rot, _ = R_scipy.align_vectors([[0, 0, 1]], [up_body])
        initial_pose = gtsam.Pose3(
            gtsam.Rot3(initial_rot.as_matrix()), gtsam.Point3(0, 0, 0)
        )
        initial_vel = np.zeros(3)
        self._initial_bias = gtsam.imuBias.ConstantBias(
            imu_calib.accel_bias, imu_calib.gyro_bias
        )
        self._optimizer.add_initial_state(
            initial_pose, initial_vel, self._initial_bias, set_priors=True
        )
        self._optimizer.optimize()

        # Transforms
        self._T_cam_body = np.linalg.inv(dm.T_imu_cam0)
        self._R_cam_body = self._T_cam_body[:3, :3]
        self._R_body_cam = dm.T_imu_cam0[:3, :3]

        # Frame iterator
        self._frame_iter = dm.iter_stereo_frames(
            step=self.frame_step, start_frame=self.start_frame
        )

        # Reset tracking
        self._est_positions = []
        self._gt_positions = []
        self._frame_count = 0
        self._prev_cam_ts = None
        self._P1 = None
        self._last_n_tracked = 0
        self._last_n_landmarks = 0

        # Run frame 0 with default actions (no agent decision needed)
        obs = self._run_frame_0()
        return obs, {}

    def step(self, action: np.ndarray):
        # Decode 4-action space
        parallax = PARALLAX_OPTIONS[action[0]]
        min_feat = MIN_FEATURES_OPTIONS[action[1]]
        min_obs_parallax = MIN_OBS_PARALLAX_OPTIONS[action[2]]
        landmark_reg = LANDMARK_REG_OPTIONS[action[3]]

        # Apply actions to pipeline
        self._optimizer.parallax_threshold = parallax
        self._feature_pipeline.MIN_TRACKED_FEATURES = min_feat
        self._optimizer.min_obs_parallax_deg = min_obs_parallax
        self._optimizer.landmark_reg_sigma = landmark_reg

        # Run one VIO frame
        obs, reward, terminated, truncated, info = self._run_one_frame(
            is_keyframe=True, lc_min_gap=9
        )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal pipeline execution
    # ------------------------------------------------------------------

    def _get_gt_position(self, timestamp: float) -> np.ndarray:
        gt_df = self._data_manager.gt_df
        idx = gt_df["timestamp"].searchsorted(timestamp)
        idx = min(idx, len(gt_df) - 1)
        row = gt_df.iloc[idx]
        return np.array([row["p_x"], row["p_y"], row["p_z"]])

    def _run_frame_0(self) -> np.ndarray:
        """Process the very first frame (no IMU, no agent decision)."""
        try:
            row, left_img, right_img = next(self._frame_iter)
        except StopIteration:
            return np.zeros(OBS_DIM, dtype=np.float32)

        ts = row["timestamp"]

        # Set projection matrices
        self._feature_pipeline.P1 = self._data_manager.P1
        self._feature_pipeline.P2 = self._data_manager.P2
        self._P1 = self._data_manager.P1[:3, :3]

        # Process frame 0
        result = self._feature_pipeline.process_stereo_frame2(left_img, right_img)
        observations, new_landmarks_3d = result[0], result[1]

        # Add observations to optimizer
        for lm_id, frame_id, uv in observations:
            lm_3d = new_landmarks_3d.get(lm_id, None)
            self._optimizer.add_landmark_observation(
                landmark_id=lm_id, state_idx=0, uv=uv, K=self._P1, landmark_3d=lm_3d
            )

        # Track positions
        est_pos = np.zeros(3)
        self._est_positions.append(est_pos)
        self._gt_positions.append(self._get_gt_position(ts))

        self._prev_cam_ts = ts
        self._frame_count = 1
        self._last_n_tracked = len(observations)

        return self._build_observation()

    def _run_one_frame(self, is_keyframe: bool, lc_min_gap: int):
        """Run a single VIO frame. Returns (obs, reward, terminated, truncated, info)."""
        # Get next stereo pair
        try:
            row, left_img, right_img = next(self._frame_iter)
        except StopIteration:
            obs = self._build_observation()
            return obs, 0.0, True, False, {"reason": "sequence_end"}

        i = self._frame_count
        ts = row["timestamp"]

        # --- IMU preintegration ---
        imu_samples = []
        if self._prev_cam_ts is not None:
            for imu_row in self._data_manager.iter_imu_between(
                self._prev_cam_ts, ts
            ):
                imu_samples.append(
                    IMUSample(
                        t=imu_row["timestamp"],
                        accel=imu_row[["a_x", "a_y", "a_z"]].values.astype(float),
                        gyro=imu_row[["w_x", "w_y", "w_z"]].values.astype(float),
                    )
                )

        current_estimate = self._optimizer.get_current_estimate()
        if current_estimate.exists(B(i - 1)):
            last_bias = current_estimate.atConstantBias(B(i - 1))
        else:
            last_bias = self._initial_bias

        self._imu_pipeline.preint.reset(
            last_bias.accelerometer(), last_bias.gyroscope()
        )
        self._imu_pipeline.preint.integrate(imu_samples)

        # --- Frontend: extract & match ---
        # Compute R_prev_curr from IMU for optical flow guidance
        R_prev_curr = None
        if len(imu_samples) > 0:
            R_body = self._imu_pipeline.preint.preint.deltaRij().matrix()
            R_prev_curr = self._R_cam_body @ R_body @ self._R_body_cam

        result = self._feature_pipeline.process_stereo_frame2(
            left_img, right_img, R_prev_curr=R_prev_curr
        )
        observations, new_landmarks_3d = result[0], result[1]

        self._last_n_tracked = len(observations)
        self._last_n_landmarks = len(self._feature_pipeline.landmarks)

        # --- Backend: add state + factors ---
        last_pose = current_estimate.atPose3(X(i - 1))
        last_vel = current_estimate.atVector(V(i - 1))
        last_state = gtsam.NavState(last_pose, last_vel)
        predicted_state = self._imu_pipeline.preint.preint.predict(
            last_state, last_bias
        )

        self._optimizer.add_state_variable(i, predicted_state, last_bias)
        self._optimizer.add_imu_factor(
            self._imu_pipeline.preint.preint, i - 1, i
        )

        # Visual factors
        for lm_id, frame_id, uv in observations:
            lm_3d = new_landmarks_3d.get(lm_id, None)
            self._optimizer.add_landmark_observation(
                landmark_id=lm_id, state_idx=i, uv=uv,
                K=self._P1, landmark_3d=lm_3d,
            )

        # Optimize
        try:
            self._optimizer.optimize()
        except Exception as e:
            obs = self._build_observation()
            return obs, -1.0, True, False, {"reason": f"optimizer_crash: {e}"}

        # --- Track estimated & GT positions ---
        est = self._optimizer.get_current_estimate()
        if est is not None and est.exists(X(i)):
            ep = est.atPose3(X(i)).translation()
            self._est_positions.append(np.array([ep[0], ep[1], ep[2]]))
        else:
            self._est_positions.append(self._est_positions[-1].copy())

        self._gt_positions.append(self._get_gt_position(ts))

        # --- Reward (sliding window pose error) ---
        reward = self._compute_reward(is_keyframe)

        self._prev_cam_ts = ts
        self._frame_count += 1

        # Termination
        terminated = False
        truncated = self._frame_count >= self.max_episode_frames

        # Compute final ATE when episode ends (must be done here, before SB3
        # auto-resets and clears _est_positions)
        final_ate = None
        if terminated or truncated:
            if len(self._est_positions) >= 3:
                _, errors = _umeyama_align(
                    np.array(self._est_positions), np.array(self._gt_positions)
                )
                final_ate = float(np.sqrt(np.mean(errors ** 2)))

        info = {
            "frame": i,
            "n_observations": len(observations),
            "n_landmarks": self._last_n_landmarks,
            "ate_window": self._last_window_error,
        }
        if final_ate is not None:
            info["final_ate"] = final_ate

        obs = self._build_observation()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    _last_window_error = 0.0

    def _compute_reward(self, is_keyframe: bool) -> float:
        """Full-trajectory aligned position error reward.

        Uses Umeyama alignment over ALL positions so far (not a tiny window).
        This removes the constant offset from initializing at origin, while
        still capturing accumulated drift.

        Reward = λ · max(-1, ε_target - e_pos)
        Pure accuracy signal — no keyframe penalty since keyframe is now fixed.
        """
        n = len(self._est_positions)
        if n < 3:
            return 0.0

        # Align full trajectory so far, then take error at current frame
        est_all = np.array(self._est_positions)
        gt_all = np.array(self._gt_positions[:n])
        _, errors = _umeyama_align(est_all, gt_all)
        current_error = errors[-1] if len(errors) > 0 else 0.0
        self._last_window_error = float(current_error)

        # Pure accuracy reward — directly penalizes drift
        # Range: [-0.03, +0.009] per frame
        lambda1 = 0.03
        epsilon_target = 0.3  # meters — positive when aligned error < 30cm

        return lambda1 * max(-1.0, epsilon_target - current_error)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """Build fixed-size observation vector: [map_stats | flattened_keypoints]."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # --- Map statistics (15-dim) ---
        idx = 0
        # 0: normalized tracked feature count
        obs[idx] = self._last_n_tracked / 1000.0; idx += 1
        # 1: normalized landmark count
        obs[idx] = self._last_n_landmarks / 5000.0; idx += 1
        # 2: current frame index (normalized)
        obs[idx] = self._frame_count / 500.0; idx += 1

        # 3-5: last position error sliding window
        if len(self._est_positions) >= self.reward_window:
            est_w = np.array(self._est_positions[-self.reward_window:])
            gt_w = np.array(self._gt_positions[-self.reward_window:])
            _, errs = _umeyama_align(est_w, gt_w)
            obs[idx] = errs[-1] if len(errs) > 0 else 0.0; idx += 1
            obs[idx] = errs.mean() if len(errs) > 0 else 0.0; idx += 1
            obs[idx] = errs.max() if len(errs) > 0 else 0.0; idx += 1
        else:
            idx += 3

        # 6-8: optimizer diagnostics
        if self._optimizer is not None:
            metrics = self._optimizer.diagnostics()
            if metrics:
                obs[idx] = np.clip(metrics.get("avg_error_per_factor", 0) / 10.0, -5, 5); idx += 1
                obs[idx] = metrics.get("n_landmarks", 0) / 1000.0; idx += 1
                obs[idx] = metrics.get("n_factors", 0) / 5000.0; idx += 1
            else:
                idx += 3
        else:
            idx += 3

        # 9-11: position covariance eigenvalues
        if self._optimizer is not None and self._frame_count > 0:
            pos_cov = self._optimizer.get_position_covariance(self._frame_count - 1)
            if pos_cov is not None:
                eigs = np.sort(np.linalg.eigvalsh(pos_cov))
                obs[idx] = np.clip(np.sqrt(eigs[0]), 0, 5); idx += 1
                obs[idx] = np.clip(np.sqrt(eigs[1]), 0, 5); idx += 1
                obs[idx] = np.clip(np.sqrt(eigs[2]), 0, 5); idx += 1
            else:
                idx += 3
        else:
            idx += 3

        # 12: degeneracy flag
        if self._optimizer is not None and self._frame_count > 0:
            deg = self._optimizer.detect_degeneracy(self._frame_count - 1)
            obs[idx] = 1.0 if deg.get("degenerate", False) else 0.0; idx += 1
        else:
            idx += 1

        # 13-14: IMU preintegration delta (norm of delta position, delta velocity)
        if (self._imu_pipeline is not None
                and self._imu_pipeline.preint.preint is not None):
            try:
                dp = self._imu_pipeline.preint.preint.deltaPij()
                dv = self._imu_pipeline.preint.preint.deltaVij()
                obs[idx] = np.linalg.norm(dp); idx += 1
                obs[idx] = np.linalg.norm(dv); idx += 1
            except Exception:
                idx += 2
        else:
            idx += 2

        # --- Keypoints (flattened, zero-padded) ---
        kp_start = MAP_STATS_DIM
        if self._feature_pipeline is not None and self._feature_pipeline.prev_keypoints is not None:
            kpts = self._feature_pipeline.prev_keypoints
            n = min(len(kpts), MAX_KEYPOINTS_FOR_OBS)
            # Subsample if needed (evenly spaced)
            if len(kpts) > MAX_KEYPOINTS_FOR_OBS:
                indices = np.linspace(0, len(kpts) - 1, MAX_KEYPOINTS_FOR_OBS, dtype=int)
                kpts = kpts[indices]
            # Normalize pixel coords to [-1, 1]
            img_w, img_h = 752, 480
            for j in range(n):
                base = kp_start + j * KEYPOINT_FEAT_DIM
                obs[base + 0] = (kpts[j, 0] / img_w) * 2 - 1  # u normalized
                obs[base + 1] = (kpts[j, 1] / img_h) * 2 - 1  # v normalized
                obs[base + 2] = 0.0  # depth_inv placeholder
                obs[base + 3] = 0.0  # track_age placeholder

        return obs
