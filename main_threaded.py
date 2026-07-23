"""
Tightly-Coupled VIO — Frontend/Backend Threaded Architecture.

Frontend thread: stereo feature processing (XFeat, KLT, loop candidates) and
median displacement estimation for keyframe gating.
Backend thread (main): IMU preintegration, keyframe-gated factor graph optimization,
covariance-based init, and visualization.

The frontend runs one frame ahead of the backend. IMU-guided optical flow uses the
rotation from the previous backend optimization (one-frame latency).
--------------------------------------------------------------------------
Frontend Thread                          Backend Thread (main)
─────────────────                        ─────────────────────
Frame i+1:                               Frame i:
  • Read stereo images                     • IMU preintegration
  • XFeat / stereo matching                • Keyframe decision (disp threshold)
  • KLT optical flow                       • Add state + IMU factor
  • Triangulation                          • Add visual / loop factors
  • Median displacement                    • ISAM2 optimize
  • Loop closure candidates                • Covariance / degeneracy
         │                                        • Gating analysis logging
         └──── Queue(maxsize=2) ──────────────────┘
                                                  │
                                    r_prev_curr_holder[0] ← R from IMU
                                                  │
         ┌────────────────────────────────────────┘
         │ (one-frame latency)
  Frame i+2:
  • Uses R_prev_curr from frame i's optimization
"""

import os
import cv2
import numpy as np
import yaml
import pandas as pd
from scipy.spatial.transform import Rotation as R
from threading import Thread, Lock
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import time

import sys

# sys.path.append(r"f:\Code\SLAM")
from data_manager import DataManager
from v_frontend import vFeature
from imu_pipeline import IMUPipeline, IMUCalibration, IMUSample
from vio_optimizer import GraphOptimizer, X, V, B, L
from vio_visualizer import VIOVisualizer
import gtsam

# RL Agent imports
from stable_baselines3 import PPO
from rl_vio_env import (
    PARALLAX_OPTIONS, MIN_FEATURES_OPTIONS,
    MIN_OBS_PARALLAX_OPTIONS,
    MAP_STATS_DIM, MAX_KEYPOINTS_FOR_OBS, KEYPOINT_FEAT_DIM, OBS_DIM,
    _umeyama_align,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class FrontendResult:
    """Data produced by the frontend for one frame."""
    frame_idx: int
    timestamp: float
    left_img: np.ndarray
    right_img: np.ndarray
    observations: List
    new_landmarks_3d: Dict
    loop_candidates: List
    lc_matched_frames: List  # [(frame_idx, vote_count), ...] from query_similar_frames
    median_displacement: float = 0.0


# Sentinel to signal frontend completion
_FRONTEND_DONE = None


@dataclass
class VisFrame:
    """All data the visualization thread needs for one frame."""
    frame_idx: int
    left_img: np.ndarray
    right_img: np.ndarray
    observations: List
    new_landmarks_3d: Dict
    all_landmarks: Dict  # snapshot of feature_pipeline.landmarks
    gtsam_estimate: Any  # GTSAM Values (already a copy from get_current_estimate)
    num_states: int
    imu_samples_count: int
    metrics: Optional[Dict]
    loop_closure_ids: List
    gt_pose: Optional[np.ndarray]
    pose_covariance: Optional[np.ndarray]
    ate_rmse: Optional[float]



# ---------------------------------------------------------------------------
# RL Agent observation builder (mirrors rl_vio_env._build_observation)
# ---------------------------------------------------------------------------
def build_rl_observation(
    n_tracked: int,
    n_landmarks: int,
    frame_count: int,
    est_positions: list,
    gt_positions: list,
    optimizer: GraphOptimizer,
    kf_idx: int,
    imu_pipeline_obj,
    feature_pipeline,
    reward_window: int = 5,
) -> np.ndarray:
    """Build the same observation vector the RL agent expects."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    idx = 0

    obs[idx] = n_tracked / 1000.0; idx += 1
    obs[idx] = n_landmarks / 5000.0; idx += 1
    obs[idx] = frame_count / 500.0; idx += 1

    # Sliding window ATE
    if len(est_positions) >= reward_window:
        est_w = np.array(est_positions[-reward_window:])
        gt_w = np.array(gt_positions[-reward_window:])
        _, errs = _umeyama_align(est_w, gt_w)
        obs[idx] = errs[-1] if len(errs) > 0 else 0.0; idx += 1
        obs[idx] = errs.mean() if len(errs) > 0 else 0.0; idx += 1
        obs[idx] = errs.max() if len(errs) > 0 else 0.0; idx += 1
    else:
        idx += 3

    # Optimizer diagnostics
    metrics = optimizer.diagnostics()
    if metrics:
        obs[idx] = np.clip(metrics.get("avg_error_per_factor", 0) / 10.0, -5, 5); idx += 1
        obs[idx] = metrics.get("n_landmarks", 0) / 1000.0; idx += 1
        obs[idx] = metrics.get("n_factors", 0) / 5000.0; idx += 1
    else:
        idx += 3

    # Position covariance eigenvalues
    pos_cov = optimizer.get_position_covariance(kf_idx)
    if pos_cov is not None:
        eigs = np.sort(np.linalg.eigvalsh(pos_cov))
        obs[idx] = np.clip(np.sqrt(eigs[0]), 0, 5); idx += 1
        obs[idx] = np.clip(np.sqrt(eigs[1]), 0, 5); idx += 1
        obs[idx] = np.clip(np.sqrt(eigs[2]), 0, 5); idx += 1
    else:
        idx += 3

    # Degeneracy flag
    deg = optimizer.detect_degeneracy(kf_idx)
    obs[idx] = 1.0 if deg.get("degenerate", False) else 0.0; idx += 1

    # IMU preintegration deltas
    try:
        dp = imu_pipeline_obj.preint.preint.deltaPij()
        dv = imu_pipeline_obj.preint.preint.deltaVij()
        obs[idx] = np.linalg.norm(dp); idx += 1
        obs[idx] = np.linalg.norm(dv); idx += 1
    except Exception:
        idx += 2

    # Keypoints
    kp_start = MAP_STATS_DIM
    if feature_pipeline.prev_keypoints is not None:
        kpts = feature_pipeline.prev_keypoints
        n = min(len(kpts), MAX_KEYPOINTS_FOR_OBS)
        if len(kpts) > MAX_KEYPOINTS_FOR_OBS:
            indices = np.linspace(0, len(kpts) - 1, MAX_KEYPOINTS_FOR_OBS, dtype=int)
            kpts = kpts[indices]
        img_w, img_h = 752, 480
        for j in range(n):
            base = kp_start + j * KEYPOINT_FEAT_DIM
            obs[base + 0] = (kpts[j, 0] / img_w) * 2 - 1
            obs[base + 1] = (kpts[j, 1] / img_h) * 2 - 1
            obs[base + 2] = 0.0
            obs[base + 3] = 0.0

    return obs


def compute_ate(est_positions, gt_positions):
    """Compute Absolute Trajectory Error (ATE) with SE(3) Umeyama alignment."""
    est = np.array(est_positions)
    gt = np.array(gt_positions)
    n = len(est)
    if n < 3:
        return 0.0, np.zeros(n), np.eye(3), np.zeros(3)

    est_mean = est.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    est_centered = est - est_mean
    gt_centered = gt - gt_mean

    H = est_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])

    R_align = Vt.T @ sign_matrix @ U.T
    t_align = gt_mean - R_align @ est_mean
    est_aligned = (R_align @ est.T).T + t_align

    ate_errors = np.linalg.norm(gt - est_aligned, axis=1)
    ate_rmse = float(np.sqrt(np.mean(ate_errors ** 2)))
    return ate_rmse, ate_errors, R_align, t_align


def compute_rte(est_positions, gt_positions, delta=4):
    """Compute Relative Trajectory Error (RTE) — drift per segment.

    Measures the error in relative motion over segments of `delta` poses.
    Unlike ATE (which uses global alignment), RTE captures local drift
    and is independent of global reference frame alignment.

    Args:
        est_positions: List of estimated 3D positions.
        gt_positions: List of ground truth 3D positions.
        delta: Number of poses between segment start and end (default: 4).

    Returns:
        rte_rmse: RMSE of relative translation errors (meters)
        rte_errors: Per-segment relative translation errors (N-delta,)
    """
    est = np.array(est_positions)
    gt = np.array(gt_positions)
    n = len(est)
    if n <= delta:
        return 0.0, np.zeros(max(n - delta, 0))

    # Relative displacements over `delta` poses
    est_deltas = est[delta:] - est[:-delta]  # (N-delta, 3)
    gt_deltas = gt[delta:] - gt[:-delta]     # (N-delta, 3)

    # Per-segment relative error
    rte_errors = np.linalg.norm(est_deltas - gt_deltas, axis=1)
    rte_rmse = float(np.sqrt(np.mean(rte_errors ** 2)))
    return rte_rmse, rte_errors


def vis_worker(
    visualizer: VIOVisualizer,
    vis_queue: Queue,
):
    """Visualization thread: pulls VisFrame objects from queue and logs to Rerun.

    Always processes the latest frame, dropping stale ones if the backend
    outruns visualization.
    """

    while True:
        # Get the next frame (blocking)
        try:
            vis_data = vis_queue.get(timeout=2.0)
        except Empty:
            continue

        if vis_data is None:  # Sentinel: shutdown
            break

        # Drain queue to get the latest frame (drop stale ones)
        while True:
            try:
                newer = vis_queue.get_nowait()
            except Empty:
                break
            if newer is None:  # Sentinel found while draining
                vis_data = None
                break
            vis_data = newer

        if vis_data is None:
            break

        # --- Regular Rerun visualization ---
        try:
            visualizer.update(
                frame_idx=vis_data.frame_idx,
                left_img=vis_data.left_img,
                right_img=vis_data.right_img,
                observations=vis_data.observations,
                new_landmarks_3d=vis_data.new_landmarks_3d,
                all_landmarks=vis_data.all_landmarks,
                gtsam_estimate=vis_data.gtsam_estimate,
                num_states=vis_data.num_states,
                imu_samples_count=vis_data.imu_samples_count,
                metrics=vis_data.metrics,
                loop_closure_ids=vis_data.loop_closure_ids,
                gt_pose=vis_data.gt_pose,
                pose_covariance=vis_data.pose_covariance,
                ate_rmse=vis_data.ate_rmse,
            )
        except Exception as e:
            print(f"  [Vis] Error: {e}")




def frontend_worker(
    feature_pipeline: vFeature,
    data_manager: DataManager,
    result_queue: Queue,
    r_prev_curr_holder: List,  # Mutable container: [R_prev_curr or None]
    r_prev_curr_lock: Lock,    # Protects reads/writes to r_prev_curr_holder
    frame_step: int,
    start_frame: int,
):
    """Frontend thread: extracts features from stereo frames and pushes results to queue.

    Reads r_prev_curr_holder[0] under lock for IMU-guided optical flow (one-frame latency).
    """
    first_run = True

    for i, (row, left_img, right_img) in enumerate(
        data_manager.iter_stereo_frames(step=frame_step, start_frame=start_frame)
    ):
        timestamp = row["timestamp"]

        if first_run:
            feature_pipeline.P1 = data_manager.P1
            feature_pipeline.P2 = data_manager.P2
            first_run = False

        # Read IMU-predicted rotation (one-frame latency from backend)
        with r_prev_curr_lock:
            R_prev_curr = r_prev_curr_holder[0]

        # Heavy computation: SuperPoint + LightGlue + KLT + triangulation
        t0 = time.perf_counter()
        observations, new_landmarks_3d, loop_candidates, median_displacement = (
            feature_pipeline.process_stereo_frame2(
                left_img, right_img, R_prev_curr=R_prev_curr
            )
        )
        frontend_ms = (time.perf_counter() - t0) * 1000
        print(
            f"Frontend frame {i} processed in {frontend_ms:.1f} ms")
        result = FrontendResult(
            frame_idx=i,
            timestamp=timestamp,
            left_img=left_img,
            right_img=right_img,
            observations=observations,
            new_landmarks_3d=new_landmarks_3d,
            loop_candidates=loop_candidates,
            lc_matched_frames=list(feature_pipeline.lc_matched_frames),
            median_displacement=median_displacement,
        )

        # Block until backend is ready (queue maxsize=2 provides backpressure)
        result_queue.put(result)

    # Signal completion
    result_queue.put(_FRONTEND_DONE)


def main():
    data_dir = "/home/liorsl/Self/datasets/MH_01_easy/MH_01_easy/mav0"
    # 
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    # --- Data Loading and Calibration ---
    data_manager = DataManager(data_dir)
    data_manager.load_data()

    K1, dist_coeffs1, new_K1 = data_manager.load_camera_calib("cam0")
    K2, dist_coeffs2, new_K2 = data_manager.load_camera_calib("cam1")
    baseline = data_manager.get_baseline(K1, K2)
    print(f"Baseline between cam0 and cam1: {baseline} meters")

    feature_pipeline = vFeature(
        matcher_type="xfeat",
        baseline=baseline,
        intrinsics=K1,
        dist_coeffs=dist_coeffs1,
        T_cam1_cam0=data_manager.T_cam1_cam0,
        device="cpu",
    )

    # =================================================================
    # --- 1. VIO System Initialization ---
    # =================================================================
    print("Initializing VIO system...")
    
    # Estimate IMU biases and gravity-aligned orientation from static period
    accel_bias_init, gyro_bias_init, initial_orient_quat, _ = (
        data_manager.calculate_initial_biases_and_gravity(static_duration_sec=3.0)
    )
    print(f"  Initial gyro bias: {gyro_bias_init}")
    print(f"  Initial accel bias: {accel_bias_init}")
    
    imu_calib = IMUCalibration(
        accel_bias=accel_bias_init,
        gyro_bias=gyro_bias_init,
        accel_noise=2.0e-3,
        gyro_noise=1.6968e-4,
    )
    imu_pipeline = IMUPipeline(imu_calib)
    optimizer = GraphOptimizer(
        use_isam=True, body_P_sensor=data_manager.T_imu_cam0, imu_calib=imu_calib
    )

    # Gravity-aligned initial orientation (pitch/roll only, yaw=0).
    # Accelerometer static mean points "up" in body frame.
    # Find minimal rotation: body "up" → world "up" [0,0,1].
    static_accel = data_manager.imu_df[["a_x", "a_y", "a_z"]].values[:int(3.0 * 200)]
    up_body = static_accel.mean(axis=0)
    up_body /= np.linalg.norm(up_body)
    print(f"  Estimated gravity direction in body frame: {up_body}")
    initial_rot, _ = R.align_vectors([[0, 0, 1]], [up_body])
    initial_pose = gtsam.Pose3(
        gtsam.Rot3(initial_rot.as_matrix()), gtsam.Point3(0, 0, 0)
    )
    # initial_pose = gtsam.Pose3()
    initial_vel = np.zeros(3)
    initial_bias = gtsam.imuBias.ConstantBias(
        imu_calib.accel_bias, imu_calib.gyro_bias
    )

    optimizer.add_initial_state(initial_pose, initial_vel, initial_bias, set_priors=True)
    optimizer.optimize()

    feature_pipeline.initialize()
    print("VIO system initialized.")

    # --- Ground Truth ---
    gt_df = data_manager.gt_df

    def get_gt_position(timestamp):
        idx = gt_df["timestamp"].searchsorted(timestamp)
        idx = min(idx, len(gt_df) - 1)
        row = gt_df.iloc[idx]
        return np.array([row["p_x"], row["p_y"], row["p_z"]])

    # --- ATE Tracking ---
    ate_est_positions = []
    ate_gt_positions = []

    # --- RL Agent ---
    rl_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_vio_agent_mh01.zip")
    rl_agent = None
    if os.path.exists(rl_model_path):
        rl_agent = PPO.load(rl_model_path, device="cpu")
        print(f"  RL agent loaded from {rl_model_path}")
    else:
        print(f"  RL agent NOT found at {rl_model_path} — using fixed parameters.")

    # RL action log
    import csv
    rl_action_log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rl_action_log.csv"
    )
    rl_action_log_file = open(rl_action_log_path, "w", newline="")
    rl_action_writer = csv.writer(rl_action_log_file)
    rl_action_writer.writerow([
        "frame_idx", "kf_idx", "timestamp",
        "parallax_threshold", "min_features", "min_obs_parallax",
        "action_idx_0", "action_idx_1", "action_idx_2",
    ])

    # --- Visualization ---
    visualizer = VIOVisualizer(max_trail_length=500, update_interval=0.05)

    # =================================================================
    # --- 2. Launch Frontend Thread ---
    # =================================================================
    frame_step = 10
    start_frame = 40 * frame_step + 22*frame_step
    loop_closure_start_frame = 10
    use_imu_for_flow = False
    # NOTE: Hinders performance in some cases
    imu_flow_error_threshold = 0.2
    use_rl_agent = False  # Set to False to disable RL parameter tuning

    # Shared state: backend writes R_prev_curr here, frontend reads it.
    # Protected by a lock so the frontend always sees a complete, consistent
    # rotation matrix (or None) — never a partially-written value.
    r_prev_curr_holder = [None]
    r_prev_curr_lock = Lock()

    # Queue with maxsize=2: allows frontend to be 1 frame ahead, blocks if backend is slow
    result_queue = Queue(maxsize=2)

    # Visualization queue: maxsize=3 so backend never blocks; vis thread drains to latest
    vis_queue = Queue(maxsize=2)

    frontend_thread = Thread(
        target=frontend_worker,
        args=(
            feature_pipeline,
            data_manager,
            result_queue,
            r_prev_curr_holder,
            r_prev_curr_lock,
            frame_step,
            start_frame,
        ),
        daemon=True,
    )
    frontend_thread.start()
    print("Frontend thread started.")

    # --- Combined Visualization Thread ---
    vis_thread = Thread(
        target=vis_worker,
        args=(visualizer, vis_queue),
        daemon=True,
    )
    vis_thread.start()
    print("Visualization thread started.")

    # =================================================================
    # --- 3. Backend Loop (Main Thread) ---
    # =================================================================
    P1 = None
    prev_cam_timestamp = None
    T_cam_body = np.linalg.inv(data_manager.T_imu_cam0)
    R_cam_body = T_cam_body[:3, :3]
    R_body_cam = data_manager.T_imu_cam0[:3, :3]

    # --- Gating analysis log: per-state geometry (parallax) vs. conditioning ---
    gating_log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gating_analysis_log.csv"
    )
    gating_log_file = open(gating_log_path, "w", newline="")
    gating_writer = csv.writer(gating_log_file)
    gating_writer.writerow([
        "frame_idx", "timestamp", "n_observations", "n_imu_samples",
        "median_pixel_disp", "parallax_max_deg", "parallax_median_deg", "parallax_count",
        "cov_trace", "cov_eig0", "cov_eig1", "cov_eig2", "condition_number",
        "degenerate", "motion_type", "avg_error_per_factor", "total_error", "n_factors",
        "n_landmarks", "n_buffered", "n_new_landmarks",
        "df_attempted", "df_promoted", "df_outlier", "df_pending",
        "ate_rmse", "frame_error",
        "worst_dir_x", "worst_dir_y", "worst_dir_z",
        "backend_ms",
    ])

    # --- Keyframe gating + initialization state ---
    kf_idx = 0                  # graph index of the last committed keyframe (X(0) = initial state)
    last_bias = initial_bias    # bias of the most recent keyframe (used for predict + preint reset)
    accumulated_disp = 0.0      # median pixel parallax accumulated since the last keyframe
    accumulated_imu = 0         # IMU samples integrated since the last keyframe
    imu_count_this_kf = 0       # IMU count of the keyframe interval just committed (for viz)
    init_done = False           # becomes True once the graph covariance stabilizes
    cov_trace_window = []       # recent position-covariance traces (init stability window)
    INIT_STABLE_WINDOW = 5      # keyframes of stable covariance required to finish init
    INIT_STABLE_REL_STD = 0.00  # relative std (std/mean) threshold for "stable"
    KF_DISP_THRESHOLD = 5.5    # px of accumulated parallax to trigger a keyframe (tunable)

    # Preintegration accumulates from the initial bias until the first keyframe commits.
    imu_pipeline.preint.reset(initial_bias.accelerometer(), initial_bias.gyroscope())

    while True:
        # Pull next frontend result (blocks until available)
        result = result_queue.get()
        if result is _FRONTEND_DONE:
            break

        i = result.frame_idx
        current_cam_timestamp = result.timestamp
        left_img = result.left_img
        right_img = result.right_img
        observations = result.observations
        new_landmarks_3d = result.new_landmarks_3d
        loop_candidates = result.loop_candidates

        t_backend_start = time.perf_counter()
        print(f"\n--- Backend Frame {i} (Timestamp: {current_cam_timestamp}) ---")
        print(f"  {len(observations)} observations from frontend.")

        # Set P1 once
        if P1 is None:
            P1 = data_manager.P1[:3, :3]

        # --- A. IMU Preintegration (accumulate across skipped frames) ---
        # Integrate this frame's IMU window onto the running preintegration WITHOUT
        # resetting, so a single IMU factor can span the whole keyframe-to-keyframe
        # interval. The accumulator is reset only after a keyframe commits (or at kf 0).
        imu_samples = []
        if prev_cam_timestamp is not None:
            for imu_row in data_manager.iter_imu_between(
                prev_cam_timestamp, current_cam_timestamp
            ):
                imu_samples.append(
                    IMUSample(
                        t=imu_row["timestamp"],
                        accel=imu_row[["a_x", "a_y", "a_z"]].values.astype(float),
                        gyro=imu_row[["w_x", "w_y", "w_z"]].values.astype(float),
                    )
                )
            imu_pipeline.preint.integrate(imu_samples)
            accumulated_imu += len(imu_samples)
            print(
                f"  Integrated {len(imu_samples)} IMU samples "
                f"(accumulated {accumulated_imu} since last keyframe)."
            )
        prev_cam_timestamp = current_cam_timestamp

        # --- B. Frame 0: Buffer landmarks only ---
        if i == 0:
            for lm_id, frame_id, uv in observations:
                lm_3d = new_landmarks_3d.get(lm_id, None)
                optimizer.add_landmark_observation(
                    landmark_id=lm_id, state_idx=0, uv=uv, K=P1, landmark_3d=lm_3d
                )

            visualizer.update(
                frame_idx=i,
                left_img=left_img,
                right_img=right_img,
                observations=observations,
                new_landmarks_3d=new_landmarks_3d,
                all_landmarks=feature_pipeline.landmarks,
                gtsam_estimate=optimizer.get_current_estimate(),
                num_states=1,
                imu_samples_count=0,
                metrics=None,
                gt_pose=get_gt_position(current_cam_timestamp),
            )
            imu_pipeline.preint.reset(last_bias.accelerometer(), last_bias.gyroscope())
            accumulated_imu = 0
            accumulated_disp = 0.0
            continue

        # --- B2. Keyframe gating decision ---
        # Accumulate visual parallax (median pixel displacement) since the last keyframe.
        # During initialization every frame is a keyframe so the graph can gather enough
        # constraints; once the backend covariance has stabilized (init_done), a frame is
        # promoted to a keyframe only when the accumulated parallax crosses the threshold.
        # Non-keyframes keep accumulating IMU and drop their visual observations.
        accumulated_disp += float(result.median_displacement)
        if init_done:
            is_keyframe = accumulated_disp >= KF_DISP_THRESHOLD
        else:
            is_keyframe = True

        if not is_keyframe:
            print(
                f"  [non-keyframe] accumulated_disp={accumulated_disp:.1f}px "
                f"< {KF_DISP_THRESHOLD}px - IMU accumulated, visual dropped."
            )
            backend_ms = (time.perf_counter() - t_backend_start) * 1000
            print(f"  Backend time: {backend_ms:.1f} ms")
            continue

        # --- C. RL Agent: select parameters for this keyframe ---
        if use_rl_agent and rl_agent is not None and len(ate_est_positions) >= 3:
            t_rl_start = time.perf_counter()
            rl_obs = build_rl_observation(
                n_tracked=len(observations),
                n_landmarks=len(feature_pipeline.landmarks),
                frame_count=kf_idx + 1,
                est_positions=ate_est_positions,
                gt_positions=ate_gt_positions,
                optimizer=optimizer,
                kf_idx=kf_idx,
                imu_pipeline_obj=imu_pipeline,
                feature_pipeline=feature_pipeline,
            )
            rl_action, _ = rl_agent.predict(rl_obs, deterministic=True)
            rl_inference_ms = (time.perf_counter() - t_rl_start) * 1000

            # Apply actions
            parallax_val = PARALLAX_OPTIONS[rl_action[0]]
            min_feat_val = MIN_FEATURES_OPTIONS[rl_action[1]]
            min_obs_par_val = MIN_OBS_PARALLAX_OPTIONS[rl_action[2]]

            optimizer.parallax_threshold = parallax_val
            feature_pipeline.MIN_TRACKED_FEATURES = min_feat_val
            optimizer.min_obs_parallax_deg = min_obs_par_val

            # Log
            rl_action_writer.writerow([
                i, kf_idx + 1, current_cam_timestamp,
                f"{parallax_val:.1f}", min_feat_val,
                f"{min_obs_par_val:.1f}",
                int(rl_action[0]), int(rl_action[1]),
                int(rl_action[2]),
            ])
            rl_action_log_file.flush()

            print(
                f"  [RL] parallax={parallax_val:.1f}°, min_feat={min_feat_val}, "
                f"obs_par={min_obs_par_val:.1f}° "
                f"({rl_inference_ms:.2f} ms)"
            )

        # --- C2. Commit a new keyframe state and factors ---
        prev_kf = kf_idx
        kf_idx += 1
        current_estimate = optimizer.get_current_estimate()
        last_pose = current_estimate.atPose3(X(prev_kf))
        last_vel = current_estimate.atVector(V(prev_kf))
        last_state = gtsam.NavState(last_pose, last_vel)

        predicted_state = imu_pipeline.preint.preint.predict(last_state, last_bias)

        optimizer.add_state_variable(kf_idx, predicted_state, last_bias)
        optimizer.add_imu_factor(imu_pipeline.preint.preint, prev_kf, kf_idx)

        # Stationary detection: if accumulated pixel displacement is negligible,
        # add zero-velocity and zero-motion constraints to prevent drift.
        STATIONARY_DISP_THRESHOLD = 1.0  # px — below this, assume stationary
        if accumulated_disp < STATIONARY_DISP_THRESHOLD:
            
            optimizer.add_zero_velocity_prior(kf_idx, sigma=0.01)
            optimizer.add_zero_motion_constraint(prev_kf, kf_idx)
            print(f"  [STATIC] Zero-motion constraints added (disp={accumulated_disp:.2f}px)")
        else:
            INIT_STABLE_REL_STD = 0.00
            # Cold-start: on first motion frame, force-promote top landmarks
            # so the graph has visual constraints before IMU drift accumulates.
            # if not optimizer._static_promotion_done and len(optimizer.landmark_initialized) == 0:
            #     optimizer.force_promote_top_n(n=50)
        # Visual factors
        for lm_id, frame_id, uv in observations:
            lm_3d = new_landmarks_3d.get(lm_id, None)
            optimizer.add_landmark_observation(
                landmark_id=lm_id, state_idx=kf_idx, uv=uv, K=P1, landmark_3d=lm_3d
            )

        # Loop closure factors
        if loop_candidates and kf_idx >= loop_closure_start_frame:
            current_est = optimizer.get_current_estimate()
            current_pose = predicted_state.pose()
            T_world_cam = current_pose.compose(
                gtsam.Pose3(
                    gtsam.Rot3(data_manager.T_imu_cam0[:3, :3]),
                    gtsam.Point3(data_manager.T_imu_cam0[:3, 3]),
                )
            )
            T_cam_world = T_world_cam.inverse()
            obs_by_lm = {int(lm_id): uv for lm_id, _, uv in observations}
            added_loop_factors = 0
            for candidate in loop_candidates:
                if not isinstance(candidate, (tuple, list)) or len(candidate) < 1:
                    continue
                candidate_lm_id = int(candidate[0])
                lm_key = L(candidate_lm_id)
                if not current_est.exists(lm_key):
                    continue
                lm_world = current_est.atPoint3(lm_key)
                lm_cam = T_cam_world.transformFrom(gtsam.Point3(lm_world))
                if lm_cam[2] <= 0.2:
                    continue
                uv = obs_by_lm.get(candidate_lm_id)
                if uv is None:
                    continue
                optimizer.add_loop_closure_observation(
                    landmark_id=candidate_lm_id, state_idx=kf_idx, uv=uv
                )
                added_loop_factors += 1
            if added_loop_factors > 0:
                print(f"  + {added_loop_factors} LC reprojection factors added.")
        elif loop_candidates and kf_idx < loop_closure_start_frame:
            print(f"  Loop candidates deferred until keyframe {loop_closure_start_frame}.")

        # Pose-graph loop closure via matched frames (Ill posed, should work with an outside PnP measurement)
        # lc_matched_frames = result.lc_matched_frames
        # if lc_matched_frames and kf_idx >= loop_closure_start_frame:
        #     current_est_lc = optimizer.get_current_estimate()
        #     # Use predicted_state.pose() for current kf since it's not in ISAM2 yet
        #     current_body_pose = predicted_state.pose()
        #     for matched_frame_idx, vote_count in lc_matched_frames:
        #         matched_kf = matched_frame_idx
        #         if matched_kf >= kf_idx or matched_kf < 1:
        #             continue
        #         if current_est_lc is None or not current_est_lc.exists(X(matched_kf)):
        #             print(f"    [LC-skip] X({matched_kf}) not in estimate (votes={vote_count})")
        #             continue
        #         if vote_count < 80:
        #             continue
        #         if kf_idx - matched_kf < 15:
        #             continue
        #         matched_body_pose = current_est_lc.atPose3(X(matched_kf))
        #         relative_pose = matched_body_pose.between(current_body_pose)
        #         confidence = min(vote_count / 200.0, 1.0)
        #         rot_sigma = 0.08 / (0.5 + confidence)
        #         trans_sigma = 0.25 / (0.5 + confidence)
        #         noise_sigmas = np.array([rot_sigma]*3 + [trans_sigma]*3)
        #         optimizer.add_loop_closure_pose_constraint(
        #             current_idx=kf_idx,
        #             matched_idx=matched_kf,
        #             relative_pose=relative_pose,
        #             noise_sigmas=noise_sigmas,
        #         )
        #         print(
        #             f"  + LC pose constraint: kf[{kf_idx}]↔kf[{matched_kf}] "
        #             f"(votes={vote_count}, σ_r={rot_sigma:.3f}, σ_t={trans_sigma:.3f})"
        #         )

        # --- D. Optimize ---
        optimizer.optimize()
        print("  Graph optimized.")
        print(f"  {optimizer.summarize()}")
        metrics = optimizer.diagnostics()
        print(
            f"  Error: {metrics['total_error']:.2f} | Avg/factor: {metrics['avg_error_per_factor']:.4f} | Vars: {metrics['n_variables']} | Factors: {metrics['n_factors']}"
        )
        df_stats = optimizer.drain_depth_filter_stats()
        if df_stats["attempted"] > 0:
            print(
                f"  [DepthFilter] attempted={df_stats['attempted']} promoted={df_stats['promoted']} "
                f"outlier={df_stats['outlier']} pending={df_stats['pending']}"
            )

        # --- Filter high-uncertainty landmarks (every 10 frames) ---
        # Slow - prefer not to use
        # if i % 10 == 0 and i > 0:
        #     optimizer.filter_uncertain_landmarks(max_trace=3.0)

        # --- Prune stale buffer entries ---
        if optimizer.buffer_max_age_kf > 0:
            pruned = optimizer.prune_stale_buffer(kf_idx)
            if pruned > 0:
                print(f"  [Buffer] Pruned {pruned} stale landmarks (age > {optimizer.buffer_max_age_kf} kf)")

        # --- Covariance & Degeneracy ---
        pos_cov = optimizer.get_position_covariance(kf_idx)
        degeneracy = optimizer.detect_degeneracy(kf_idx)
        if degeneracy["degenerate"]:
            eigs = degeneracy["position_eigenvalues"]
            print(f"  ⚠ DEGENERATE: {degeneracy['motion_type']}")
            print(f"    Eigenvalues: [{eigs[0]:.6f}, {eigs[1]:.6f}, {eigs[2]:.6f}]")
            print(f"    Condition number: {degeneracy['condition_number']:.1f}")
        elif pos_cov is not None:
            eigs = degeneracy["position_eigenvalues"]
            print(
                f"  Pose uncertainty (3σ): [{np.sqrt(eigs[0])*3:.3f}, {np.sqrt(eigs[1])*3:.3f}, {np.sqrt(eigs[2])*3:.3f}] m"
            )

        # --- Initialization detection (graph covariance stability) ---
        # Track the position-covariance trace across recent keyframes. Once it is stable
        # (low relative spread) over INIT_STABLE_WINDOW keyframes, declare init complete
        # and switch on keyframe gating.
        if not init_done:
            cov_trace_init = (
                float(np.sum(degeneracy["position_eigenvalues"]))
                if degeneracy["position_eigenvalues"] is not None
                else None
            )
            if cov_trace_init is not None and np.isfinite(cov_trace_init):
                cov_trace_window.append(cov_trace_init)
                if len(cov_trace_window) > INIT_STABLE_WINDOW:
                    cov_trace_window.pop(0)
                if len(cov_trace_window) == INIT_STABLE_WINDOW:
                    w = np.asarray(cov_trace_window, dtype=float)
                    rel_std = float(w.std() / max(w.mean(), 1e-12))
                    if rel_std < INIT_STABLE_REL_STD:
                        init_done = True
                        print(
                            f"  ** INIT COMPLETE: covariance stable (rel_std={rel_std:.3f} "
                            f"over {INIT_STABLE_WINDOW} keyframes). Keyframe gating ENABLED "
                            f"(threshold {KF_DISP_THRESHOLD}px)."
                        )
                    else:
                        print(
                            f"  Init: cov_trace rel_std={rel_std:.3f} "
                            f"(need < {INIT_STABLE_REL_STD})."
                        )

        # --- Gating analysis logging — moved after ATE so all data is available ---

        # --- Adaptive IMU-guided flow + update shared rotation for frontend ---
        prev_use_imu = use_imu_for_flow
        use_imu_for_flow = metrics["avg_error_per_factor"] < imu_flow_error_threshold
        if use_imu_for_flow != prev_use_imu:
            status = "ENABLED" if use_imu_for_flow else "DISABLED"
            print(f"  ** IMU-guided optical flow {status}")

        # Compute R_prev_curr for the frontend from the accumulated keyframe-interval
        # rotation (one-frame latency). Done BEFORE resetting the preintegration below.
        if use_imu_for_flow and accumulated_imu > 0:
            R_body = imu_pipeline.preint.preint.deltaRij().matrix()
            R_new = R_cam_body @ R_body @ R_body_cam
            with r_prev_curr_lock:
                r_prev_curr_holder[0] = R_new
        else:
            with r_prev_curr_lock:
                r_prev_curr_holder[0] = None

        # --- Update bias and reset preintegration for the next keyframe interval ---
        est_after = optimizer.get_current_estimate()
        if est_after is not None and est_after.exists(B(kf_idx)):
            last_bias = est_after.atConstantBias(B(kf_idx))
        imu_pipeline.preint.reset(last_bias.accelerometer(), last_bias.gyroscope())
        imu_count_this_kf = accumulated_imu
        accumulated_imu = 0
        accumulated_disp = 0.0

        # --- ATE ---
        ate_rmse_current = None
        current_est_for_ate = optimizer.get_current_estimate()
        if current_est_for_ate is not None and current_est_for_ate.exists(X(kf_idx)):
            est_pos = current_est_for_ate.atPose3(X(kf_idx)).translation()
            ate_est_positions.append(np.array([est_pos[0], est_pos[1], est_pos[2]]))
            ate_gt_positions.append(get_gt_position(current_cam_timestamp))
            if len(ate_est_positions) >= 3:
                ate_rmse_current, ate_errors, _, _ = compute_ate(
                    ate_est_positions, ate_gt_positions
                )
                print(
                    f"  ATE (RMSE): {ate_rmse_current:.4f} m | Frame error: {ate_errors[-1]:.4f} m"
                )

        # --- Gating analysis logging (all metrics collected) ---
        para_stats = optimizer.drain_parallax_stats()
        cond_num = degeneracy.get("condition_number", float("nan"))
        eigs_log = degeneracy.get("position_eigenvalues")
        evecs_log = degeneracy.get("position_eigenvectors")
        if eigs_log is not None:
            cov_trace = float(np.sum(eigs_log))
            eig0, eig1, eig2 = float(eigs_log[0]), float(eigs_log[1]), float(eigs_log[2])
        else:
            cov_trace = eig0 = eig1 = eig2 = float("nan")
        if evecs_log is not None:
            worst_dir = evecs_log[:, -1]
        else:
            worst_dir = np.array([float("nan")] * 3)
        gating_writer.writerow([
            i, current_cam_timestamp, len(observations), accumulated_imu,
            f"{accumulated_disp:.4f}",
            f"{para_stats['parallax_max_deg']:.4f}",
            f"{para_stats['parallax_median_deg']:.4f}",
            para_stats["parallax_count"],
            f"{cov_trace:.8f}", f"{eig0:.8f}", f"{eig1:.8f}", f"{eig2:.8f}",
            f"{cond_num:.4f}",
            int(bool(degeneracy.get("degenerate", False))),
            degeneracy.get("motion_type", ""),
            f"{metrics['avg_error_per_factor']:.6f}",
            f"{metrics['total_error']:.6f}",
            metrics["n_factors"],
            len(optimizer.landmark_initialized),
            len(optimizer.landmark_buffer),
            len(new_landmarks_3d),
            df_stats["attempted"], df_stats["promoted"], df_stats["outlier"], df_stats["pending"],
            f"{ate_rmse_current:.6f}" if ate_rmse_current is not None else "",
            f"{ate_errors[-1]:.6f}" if ate_rmse_current is not None else "",
            f"{worst_dir[0]:.6f}", f"{worst_dir[1]:.6f}", f"{worst_dir[2]:.6f}",
            f"{(time.perf_counter() - t_backend_start) * 1000:.1f}",
        ])
        gating_log_file.flush()

        # --- E. Enqueue visualization (non-blocking, offloads to vis thread) ---
        lc_ids = []
        if loop_candidates and kf_idx >= loop_closure_start_frame:
            vis_est = optimizer.get_current_estimate()
            vis_pose = vis_est.atPose3(X(kf_idx))
            T_world_cam_vis = vis_pose.compose(
                gtsam.Pose3(
                    gtsam.Rot3(data_manager.T_imu_cam0[:3, :3]),
                    gtsam.Point3(data_manager.T_imu_cam0[:3, 3]),
                )
            )
            T_cam_world_vis = T_world_cam_vis.inverse()
            for candidate in loop_candidates:
                if isinstance(candidate, (tuple, list)) and len(candidate) >= 1:
                    cid = int(candidate[0])
                    lm_key = L(cid)
                    if not vis_est.exists(lm_key):
                        continue
                    lm_world = vis_est.atPoint3(lm_key)
                    lm_cam = T_cam_world_vis.transformFrom(gtsam.Point3(lm_world))
                    if lm_cam[2] > 0.3:
                        lc_ids.append(cid)

        vis_frame = VisFrame(
            frame_idx=kf_idx,
            left_img=left_img,
            right_img=right_img,
            observations=observations,
            new_landmarks_3d=new_landmarks_3d,
            all_landmarks=dict(feature_pipeline.landmarks),
            gtsam_estimate=optimizer.get_current_estimate(),
            num_states=kf_idx + 1,
            imu_samples_count=imu_count_this_kf,
            metrics=metrics,
            loop_closure_ids=lc_ids,
            gt_pose=get_gt_position(current_cam_timestamp),
            pose_covariance=pos_cov,
            ate_rmse=ate_rmse_current,
        )

        # Non-blocking put: if queue is full, drop oldest to keep backend fast
        try:
            vis_queue.put_nowait(vis_frame)
        except:
            try:
                vis_queue.get_nowait()  # drop oldest
            except Empty:
                pass
            vis_queue.put_nowait(vis_frame)

        backend_ms = (time.perf_counter() - t_backend_start) * 1000
        print(f"  Backend time: {backend_ms:.1f} ms")

    # --- Stop visualization thread ---
    vis_queue.put(None)  # Sentinel to stop vis_worker
    vis_thread.join(timeout=10.0)

    # --- Wait for frontend to finish ---
    frontend_thread.join()

    gating_log_file.close()
    print(f"Gating analysis log saved to {gating_log_path}")

    rl_action_log_file.close()
    if rl_agent is not None:
        print(f"RL action log saved to {rl_action_log_path}")

    # --- Final Trajectory Error Summary ---
    if len(ate_est_positions) >= 3:
        ate_rmse, ate_errors, R_align, t_align = compute_ate(
            ate_est_positions, ate_gt_positions
        )
        rte_rmse, rte_errors = compute_rte(ate_est_positions, ate_gt_positions)
        print("\n" + "=" * 60)
        print(f"  FINAL ATE (Absolute Trajectory Error)")
        print(f"  RMSE:    {ate_rmse:.4f} m")
        print(f"  Mean:    {np.mean(ate_errors):.4f} m")
        print(f"  Median:  {np.median(ate_errors):.4f} m")
        print(f"  Max:     {np.max(ate_errors):.4f} m")
        print(f"  Std:     {np.std(ate_errors):.4f} m")
        print(f"  Frames:  {len(ate_errors)}")
        print("-" * 60)
        print(f"  FINAL RTE (Relative Trajectory Error)")
        print(f"  RMSE:    {rte_rmse:.4f} m")
        print(f"  Mean:    {np.mean(rte_errors):.4f} m")
        print(f"  Median:  {np.median(rte_errors):.4f} m")
        print(f"  Max:     {np.max(rte_errors):.4f} m")
        print(f"  Std:     {np.std(rte_errors):.4f} m")
        print(f"  Segments: {len(rte_errors)}")
        print("=" * 60)

    # Cleanup
    visualizer.close()

    # Save promoted depth log for visualization
    if optimizer.promoted_depth_log:
        from visualize_depth_filter import save_depth_log
        save_depth_log(optimizer.promoted_depth_log)
        optimizer.visualize_factor_graph_3d("factor_graph_3d.html")

    print("\n=== VIO processing complete. ===")
    input("Press Enter to exit...")


if __name__ == "__main__":
    import cProfile
    import pstats

    # profiler = cProfile.Profile()
    # profiler.enable()
    # try:
    main()
    # except KeyboardInterrupt:
    #     print("\n\n*** Interrupted by user — printing profile results ***")
    # finally:
    #     profiler.disable()

    # # Print top 40 functions by cumulative time
    # stats = pstats.Stats(profiler)
    # stats.sort_stats("cumulative")
    # print("\n" + "=" * 80)
    # print("PROFILING RESULTS (sorted by cumulative time)")
    # print("=" * 80)
    # stats.print_stats(40)

    # # Also save to file for later analysis
    # stats.dump_stats("profile_results.prof")
    # print("Full profile saved to profile_results.prof")
    # print("View with: python -m snakeviz profile_results.prof")
