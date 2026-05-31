# from imu_utils import align_ground_truth_to_gravity_world, align_ground_truth_to_z_gravity_frame, create_gravity_aligned_coordinate_system, transform_imu_to_z_gravity_frame, transform_vicon_to_body_frame
# from rel_pose_vis import RelativePoseVisualizer
# from vslam import *
# from factor_graph_vio import VIFusionGraphISAM2
import os
import cv2
import numpy as np
import yaml
import pandas as pd
from scipy.spatial.transform import Rotation as R

import sys

sys.path.append(r"f:\Code\SLAM")
from data_manager import DataManager
from vfeature import vFeature
from imu_pipeline import IMUPipeline, IMUCalibration, IMUSample
from vio_optimizer import GraphOptimizer, X, V, B, L  # Add X, V, B, L imports
from vio_visualizer import VIOVisualizer
import gtsam

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    data_dir = "f:/Code/exercise_10/data/MH_01_easy/mav0"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    # --- Data Loading and Calibration ---
    data_manager = DataManager(data_dir)
    data_manager.load_data()

    # Camera calibration
    K1, dist_coeffs1, new_K1 = data_manager.load_camera_calib("cam0")
    K2, dist_coeffs2, new_K2 = data_manager.load_camera_calib("cam1")
    baseline = data_manager.get_baseline(K1, K2)
    print(f"Baseline between cam0 and cam1: {baseline} meters")

    feature_pipeline = vFeature(
        matcher_type="superpoint",
        baseline=baseline,
        intrinsics=K1,
        dist_coeffs=dist_coeffs1,
        T_cam1_cam0=data_manager.T_cam1_cam0,  # Camera 1 to Camera 0 transformation
        device="cpu",  # Use CPU for processing
    )

    # first_run = True
    # # get the first stereo pair from data manager
    # frame_step = 10
    # for row, left_img, right_img in data_manager.iter_stereo_frames(step=frame_step):
    #     if first_run:
    #         # Initialize the mapping system with the first stereo pair
    #         feature_pipeline.P1 = data_manager.P1  # Rectified projection matrix for cam0
    #         feature_pipeline.P2 = data_manager.P2  # Rectified projection matrix for cam1
    #         first_run = False

    #     # Process the first stereo frames
    #     observations, new_landmarks = feature_pipeline.process_stereo_frame2(left_img, right_img)
    #     current_cam_timestamp = row["timestamp"]

    #     if prev_cam_timestamp is not None:
    #         for imu_row in data_manager.iter_imu_between(prev_cam_timestamp, current_cam_timestamp):
    #             gyro = imu_row[["w_x", "w_y", "w_z"]].values.astype(float)
    #             acc = imu_row[["a_x", "a_y", "a_z"]].values.astype(float)
    #             timestamp = imu_row["timestamp"]
    #             # fg_vio.add_imu_measurement(acc, gyro, timestamp)
    #             # imu_ekf.predict(acc, gyro, timestamp)

    #     prev_cam_timestamp = current_cam_timestamp
    # Visualization
    # points_3d = visual_manager.global_map_points[-1] if visual_manager.global_map_points else None

    # Get the synced and transformed ground truth pose for this timestamp

    # =================================================================
    # --- 1. VIO System Initialization (Before the Loop) ---
    # =================================================================
    print("Initializing VIO system...")
    imu_calib = IMUCalibration()  # Using default noise values for now
    imu_pipeline = IMUPipeline(imu_calib)
    optimizer = GraphOptimizer(use_isam=True, body_P_sensor=data_manager.T_imu_cam0, imu_calib=imu_calib)

    # --- Initial State (at the time of the first frame) ---
    # We assume the system starts at the origin, at rest.
    initial_pose = gtsam.Pose3()  # Identity matrix
    initial_vel = np.zeros(3)

    # IMPROVED: Use the sensor's known bias as initial guess, not zeros
    initial_bias = gtsam.imuBias.ConstantBias(
        imu_calib.accel_bias, imu_calib.gyro_bias  # Use calibrated values
    )

    # Add the first state to the graph with a strong prior to anchor the world frame
    optimizer.add_initial_state(
        initial_pose, initial_vel, initial_bias, set_priors=True
    )
    optimizer.optimize()  # Flush initial state into ISAM2 so X(0), V(0), B(0) are available

    feature_pipeline.initialize()

    print("VIO system initialized.")

    # --- Visualization ---
    visualizer = VIOVisualizer(max_trail_length=500, update_interval=0.05)
    # =================================================================
    # --- 2. Main Processing Loop ---
    # =================================================================
    frame_step = 10
    loop_closure_start_frame = 10  # Delay LC insertion until trajectory is better constrained
    use_imu_for_flow = False  # Start without IMU; enable once GTSAM error is low
    imu_flow_error_threshold = 2.0  # avg_error_per_factor below this → trust IMU rotation
    first_run = True
    prev_cam_timestamp = None

    # Use an iterator that provides timestamps
    for i, (row, left_img, right_img) in enumerate(
        data_manager.iter_stereo_frames(step=frame_step, start_frame=40*frame_step)
    ):
        current_cam_timestamp = row["timestamp"]

        print(f"\n--- Processing Frame {i} (Timestamp: {current_cam_timestamp}) ---")

        if first_run:
            # Initialize the mapping system with the first stereo pair
            feature_pipeline.P1 = (
                data_manager.P1
            )  # Rectified projection matrix for cam0
            feature_pipeline.P2 = (
                data_manager.P2
            )  # Rectified projection matrix for cam1
            first_run = False
        P1 = data_manager.P1[:3, :3]  # P1 is the rectified projection matrix for cam0

        # --- A. IMU Preintegration ---
        if i > 0:  # We need a previous frame to have an interval
            # Get all IMU measurements between the last camera frame and this one
            imu_samples = []
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

            # Get the latest optimized bias from the previous state
            current_estimate = optimizer.get_current_estimate()
            if current_estimate.exists(B(i - 1)):
                last_bias = current_estimate.atConstantBias(B(i - 1))
            else:
                # Fallback to initial calibrated bias if optimizer hasn't run yet
                print(
                    f"No previous bias estimate found for frame {i-1}, using initial calibration"
                )
                last_bias = initial_bias

            # Integrate the measurements
            imu_pipeline.preint.reset(last_bias.accelerometer(), last_bias.gyroscope())
            imu_pipeline.preint.integrate(imu_samples)
            print(f"Integrated {len(imu_samples)} IMU samples.")

        # --- B. Visual Feature Processing ---
        # Compute IMU-predicted rotation for optical flow initialization
        # Only use when GTSAM error is low (IMU bias is well-calibrated)
        R_prev_curr = None
        if i > 0 and use_imu_for_flow:
            # IMU preintegration gives rotation in body frame: R_body_prev_to_curr
            R_body = imu_pipeline.preint.preint.deltaRij().matrix()
            # Transform to camera frame: R_cam = T_cam_body * R_body * T_body_cam
            T_cam_body = np.linalg.inv(data_manager.T_imu_cam0)
            R_cam_body = T_cam_body[:3, :3]
            R_body_cam = data_manager.T_imu_cam0[:3, :3]
            R_prev_curr = R_cam_body @ R_body @ R_body_cam

        observations, new_landmarks_3d, loop_candidates = feature_pipeline.process_stereo_frame2(
            left_img, right_img, R_prev_curr=R_prev_curr
        )
        print(f"Processed visual features: {len(observations)} observations.")

        # Visualize first frame (before optimizer is active)
        if i == 0:
            # Add landmark observations for the first frame — they will be buffered
            # until re-observed from a second pose (multi-view initialization).
            for lm_id, frame_id, uv in observations:
                lm_3d = new_landmarks_3d.get(lm_id, None)
                optimizer.add_landmark_observation(
                    landmark_id=lm_id,
                    state_idx=0,
                    uv=uv,
                    K=P1,
                    landmark_3d=lm_3d,
                )
            # No optimize() needed here — landmarks are buffered, not yet in ISAM2

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
            )

        # --- C. Add New State and Factors to the Graph ---
        if i > 0:
            # 1. Predict next state using IMU for a good initial guess
            current_estimate = optimizer.get_current_estimate()
            last_pose = current_estimate.atPose3(X(i - 1))
            last_vel = current_estimate.atVector(V(i - 1))
            last_state = gtsam.NavState(last_pose, last_vel)

            predicted_state = imu_pipeline.preint.preint.predict(last_state, last_bias)

            # 2. Add the new state variables to the graph with the predicted values
            optimizer.add_state_variable(i, predicted_state, last_bias)

            # 3. Add the IMU factor as a constraint between previous (i-1) and current (i) state
            optimizer.add_imu_factor(imu_pipeline.preint.preint, i - 1, i)

            # 4. Add visual factors for all observations in the current frame
            for lm_id, frame_id, uv in observations:
                # Get the 3D position if this is a new landmark
                lm_3d = new_landmarks_3d.get(lm_id, None)
                optimizer.add_landmark_observation(
                    landmark_id=lm_id,
                    state_idx=i,
                    uv=uv,
                    K=P1,
                    landmark_3d=lm_3d,  # Only needed for first observation
                )

            # 5. Add loop-closure reprojection factors after a burn-in period.
            # Computing relative pose from the current GTSAM estimate is tautological —
            # it encodes drift and cannot correct it. For now, we use reprojection factors
            # which at least provide genuine pixel-level measurements.
            if loop_candidates and i >= loop_closure_start_frame:
                current_est = optimizer.get_current_estimate()
                current_pose = predicted_state.pose()

                # Reprojection factors for geometrically visible LC landmarks
                T_world_cam = current_pose.compose(
                    gtsam.Pose3(gtsam.Rot3(data_manager.T_imu_cam0[:3, :3]),
                                gtsam.Point3(data_manager.T_imu_cam0[:3, 3]))
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
                        landmark_id=candidate_lm_id,
                        state_idx=i,
                        uv=uv,
                    )
                    added_loop_factors += 1
                if added_loop_factors > 0:
                    print(f"  + {added_loop_factors} LC reprojection factors added.")
            elif loop_candidates and i < loop_closure_start_frame:
                print(
                    f"Loop candidates found ({len(loop_candidates)}) but deferred until frame {loop_closure_start_frame}."
                )

            # --- D. Optimize the Graph ---
            optimizer.optimize()
            print("Graph optimized.")
            print(optimizer.summarize())
            metrics = optimizer.diagnostics()
            print(
                f"Error: {metrics['total_error']:.2f} | Avg/factor: {metrics['avg_error_per_factor']:.4f} | Vars: {metrics['n_variables']} | Factors: {metrics['n_factors']}"
            )

            # --- Adaptive IMU-guided optical flow ---
            prev_use_imu = use_imu_for_flow
            use_imu_for_flow = metrics['avg_error_per_factor'] < imu_flow_error_threshold
            if use_imu_for_flow != prev_use_imu:
                status = "ENABLED" if use_imu_for_flow else "DISABLED"
                print(f"** IMU-guided optical flow {status} (avg_err={metrics['avg_error_per_factor']:.2f}, threshold={imu_flow_error_threshold})")

            # --- E. Visualize ---
            # Collect loop closure landmark IDs (only geometrically visible ones)
            lc_ids = []
            if loop_candidates and i >= loop_closure_start_frame:
                vis_est = optimizer.get_current_estimate()
                vis_pose = vis_est.atPose3(X(i))
                T_world_cam_vis = vis_pose.compose(
                    gtsam.Pose3(gtsam.Rot3(data_manager.T_imu_cam0[:3, :3]),
                                gtsam.Point3(data_manager.T_imu_cam0[:3, 3]))
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
                        if lm_cam[2] > 0.5:  # Only show if in front of camera
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
            )

        # Update timestamp for the next iteration
        prev_cam_timestamp = current_cam_timestamp

    # Cleanup
    visualizer.close()
    print("\n=== VIO processing complete. Rerun viewer remains open. ===")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
