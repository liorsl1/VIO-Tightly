"""
Tightly-Coupled VIO — Frontend/Backend Threaded Architecture.

Frontend thread: stereo feature processing (SuperPoint, LightGlue, KLT, triangulation)
Backend thread (main): IMU preintegration, factor graph optimization, visualization

The frontend runs one frame ahead of the backend. IMU-guided optical flow uses
the rotation from the *previous* backend optimization (one-frame latency).
--------------------------------------------------------------------------
Frontend Thread                          Backend Thread (main)
─────────────────                        ─────────────────────
Frame i+1:                               Frame i:
  • Read stereo images                     • IMU preintegration
  • SuperPoint extraction                  • Factor graph update
  • LightGlue stereo matching              • ISAM2 optimize
  • KLT optical flow                       • Covariance / degeneracy
  • Triangulation                          • ATE computation
  • Loop closure candidates                • Rerun visualization
         │                                        │
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
from threading import Thread, Event
from queue import Queue
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import time

import sys

sys.path.append(r"f:\Code\SLAM")
from data_manager import DataManager
from v_frontend import vFeature
from imu_pipeline import IMUPipeline, IMUCalibration, IMUSample
from vio_optimizer import GraphOptimizer, X, V, B, L
from vio_visualizer import VIOVisualizer
import gtsam

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


# Sentinel to signal frontend completion
_FRONTEND_DONE = None


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


def frontend_worker(
    feature_pipeline: vFeature,
    data_manager: DataManager,
    result_queue: Queue,
    r_prev_curr_holder: List,  # Mutable container: [R_prev_curr or None]
    frame_step: int,
    start_frame: int,
):
    """Frontend thread: extracts features from stereo frames and pushes results to queue.

    Reads r_prev_curr_holder[0] for IMU-guided optical flow (one-frame latency).
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
        R_prev_curr = r_prev_curr_holder[0]

        # Heavy computation: SuperPoint + LightGlue + KLT + triangulation
        t0 = time.perf_counter()
        observations, new_landmarks_3d, loop_candidates = (
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
        )

        # Block until backend is ready (queue maxsize=2 provides backpressure)
        result_queue.put(result)

    # Signal completion
    result_queue.put(_FRONTEND_DONE)


def main():
    data_dir = "f:/Code/exercise_10/data/MH_01_easy/mav0"
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
    imu_calib = IMUCalibration()
    imu_pipeline = IMUPipeline(imu_calib)
    optimizer = GraphOptimizer(
        use_isam=True, body_P_sensor=data_manager.T_imu_cam0, imu_calib=imu_calib
    )

    initial_pose = gtsam.Pose3()
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

    # --- Visualization ---
    visualizer = VIOVisualizer(max_trail_length=500, update_interval=0.05)

    # =================================================================
    # --- 2. Launch Frontend Thread ---
    # =================================================================
    frame_step = 10
    start_frame = 40 * frame_step
    loop_closure_start_frame = 10
    use_imu_for_flow = False
    imu_flow_error_threshold = 2.0

    # Shared state: backend writes R_prev_curr here, frontend reads it
    # Using a mutable list as a simple lock-free holder (single writer, single reader)
    r_prev_curr_holder = [None]

    # Queue with maxsize=2: allows frontend to be 1 frame ahead, blocks if backend is slow
    result_queue = Queue(maxsize=2)

    frontend_thread = Thread(
        target=frontend_worker,
        args=(
            feature_pipeline,
            data_manager,
            result_queue,
            r_prev_curr_holder,
            frame_step,
            start_frame,
        ),
        daemon=True,
    )
    frontend_thread.start()
    print("Frontend thread started.")

    # =================================================================
    # --- 3. Backend Loop (Main Thread) ---
    # =================================================================
    P1 = None
    prev_cam_timestamp = None
    T_cam_body = np.linalg.inv(data_manager.T_imu_cam0)
    R_cam_body = T_cam_body[:3, :3]
    R_body_cam = data_manager.T_imu_cam0[:3, :3]

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

        # --- A. IMU Preintegration ---
        imu_samples = []
        if i > 0 and prev_cam_timestamp is not None:
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

            current_estimate = optimizer.get_current_estimate()
            if current_estimate.exists(B(i - 1)):
                last_bias = current_estimate.atConstantBias(B(i - 1))
            else:
                print(f"  No bias for frame {i-1}, using initial calibration")
                last_bias = initial_bias

            imu_pipeline.preint.reset(last_bias.accelerometer(), last_bias.gyroscope())
            imu_pipeline.preint.integrate(imu_samples)
            print(f"  Integrated {len(imu_samples)} IMU samples.")

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
            prev_cam_timestamp = current_cam_timestamp
            continue

        # --- C. Add New State and Factors ---
        current_estimate = optimizer.get_current_estimate()
        last_pose = current_estimate.atPose3(X(i - 1))
        last_vel = current_estimate.atVector(V(i - 1))
        last_state = gtsam.NavState(last_pose, last_vel)

        predicted_state = imu_pipeline.preint.preint.predict(last_state, last_bias)

        optimizer.add_state_variable(i, predicted_state, last_bias)
        optimizer.add_imu_factor(imu_pipeline.preint.preint, i - 1, i)

        # Visual factors
        for lm_id, frame_id, uv in observations:
            lm_3d = new_landmarks_3d.get(lm_id, None)
            optimizer.add_landmark_observation(
                landmark_id=lm_id, state_idx=i, uv=uv, K=P1, landmark_3d=lm_3d
            )

        # Loop closure factors
        if loop_candidates and i >= loop_closure_start_frame:
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
                if lm_cam[2] <= 0.5:
                    continue
                uv = obs_by_lm.get(candidate_lm_id)
                if uv is None:
                    continue
                optimizer.add_loop_closure_observation(
                    landmark_id=candidate_lm_id, state_idx=i, uv=uv
                )
                added_loop_factors += 1
            if added_loop_factors > 0:
                print(f"  + {added_loop_factors} LC reprojection factors added.")
        elif loop_candidates and i < loop_closure_start_frame:
            print(f"  Loop candidates deferred until frame {loop_closure_start_frame}.")

        # --- D. Optimize ---
        optimizer.optimize()
        print("  Graph optimized.")
        print(f"  {optimizer.summarize()}")
        metrics = optimizer.diagnostics()
        print(
            f"  Error: {metrics['total_error']:.2f} | Avg/factor: {metrics['avg_error_per_factor']:.4f} | Vars: {metrics['n_variables']} | Factors: {metrics['n_factors']}"
        )

        # --- Filter high-uncertainty landmarks (every 10 frames) ---
        # Slow - prefer not to use
        # if i % 10 == 0 and i > 0:
        #     optimizer.filter_uncertain_landmarks(max_trace=3.0)

        # --- Covariance & Degeneracy ---
        pos_cov = optimizer.get_position_covariance(i)
        degeneracy = optimizer.detect_degeneracy(i)
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

        # --- Adaptive IMU-guided flow + update shared rotation for frontend ---
        prev_use_imu = use_imu_for_flow
        use_imu_for_flow = metrics["avg_error_per_factor"] < imu_flow_error_threshold
        if use_imu_for_flow != prev_use_imu:
            status = "ENABLED" if use_imu_for_flow else "DISABLED"
            print(f"  ** IMU-guided optical flow {status}")

        # Compute R_prev_curr for the frontend (one-frame latency)
        # This rotation will be used by the frontend for frame i+1 (or i+2 depending on queue state)
        if use_imu_for_flow and len(imu_samples) > 0:
            R_body = imu_pipeline.preint.preint.deltaRij().matrix()
            r_prev_curr_holder[0] = R_cam_body @ R_body @ R_body_cam
        else:
            r_prev_curr_holder[0] = None

        # --- ATE ---
        ate_rmse_current = None
        current_est_for_ate = optimizer.get_current_estimate()
        if current_est_for_ate is not None and current_est_for_ate.exists(X(i)):
            est_pos = current_est_for_ate.atPose3(X(i)).translation()
            ate_est_positions.append(np.array([est_pos[0], est_pos[1], est_pos[2]]))
            ate_gt_positions.append(get_gt_position(current_cam_timestamp))
            if len(ate_est_positions) >= 3:
                ate_rmse_current, ate_errors, _, _ = compute_ate(
                    ate_est_positions, ate_gt_positions
                )
                print(
                    f"  ATE (RMSE): {ate_rmse_current:.4f} m | Frame error: {ate_errors[-1]:.4f} m"
                )

        # --- E. Visualize ---
        lc_ids = []
        if loop_candidates and i >= loop_closure_start_frame:
            vis_est = optimizer.get_current_estimate()
            vis_pose = vis_est.atPose3(X(i))
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
                    if lm_cam[2] > 0.5:
                        lc_ids.append(cid)

        visualizer.update(
            frame_idx=i,
            left_img=left_img,
            right_img=right_img,
            observations=observations,
            new_landmarks_3d=new_landmarks_3d,
            all_landmarks=feature_pipeline.landmarks,
            gtsam_estimate=optimizer.get_current_estimate(),
            num_states=i + 1,
            imu_samples_count=len(imu_samples),
            metrics=metrics,
            loop_closure_ids=lc_ids,
            gt_pose=get_gt_position(current_cam_timestamp),
            pose_covariance=pos_cov,
            ate_rmse=ate_rmse_current,
        )

        backend_ms = (time.perf_counter() - t_backend_start) * 1000
        print(f"  Backend time: {backend_ms:.1f} ms")

        prev_cam_timestamp = current_cam_timestamp

    # --- Wait for frontend to finish ---
    frontend_thread.join()

    # --- Final ATE Summary ---
    if len(ate_est_positions) >= 3:
        ate_rmse, ate_errors, R_align, t_align = compute_ate(
            ate_est_positions, ate_gt_positions
        )
        print("\n" + "=" * 60)
        print(f"  FINAL ATE (Absolute Trajectory Error)")
        print(f"  RMSE:    {ate_rmse:.4f} m")
        print(f"  Mean:    {np.mean(ate_errors):.4f} m")
        print(f"  Median:  {np.median(ate_errors):.4f} m")
        print(f"  Max:     {np.max(ate_errors):.4f} m")
        print(f"  Std:     {np.std(ate_errors):.4f} m")
        print(f"  Frames:  {len(ate_errors)}")
        print("=" * 60)

    # Cleanup
    visualizer.close()
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
