import cv2
import numpy as np
import pandas as pd
#from imu_utils import from_two_vectors2,from_two_vectors, align_ground_truth_to_gravity_world, from_two_vectors3
from scipy.spatial.transform import Rotation as R

import os

class DataManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cam_df = None
        self.imu_df = None
        self.gt_df = None
        self.T_VICONB = np.array([
            [ 0.33638, -0.01749,  0.94156,  0.06901],
         [-0.02078, -0.99972, -0.01114, -0.02781],
          [0.94150, -0.01582, -0.33665, -0.12395],
            [  0.0,      0.0,      0.0,      1.0],
        ]) # Vicon to Body

        self.T_imu_cam0 = np.array(
            [[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
         [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
        [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
         [0.0, 0.0, 0.0, 1.0]]
        )

        self.T_imu_cam1 = np.array(
             [[0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
         [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
        [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
         [0.0, 0.0, 0.0, 1.0]]
        )
        self.T_cam0_imu = np.linalg.inv(self.T_imu_cam0)  # Camera to IMU
        self.T_cam1_imu = np.linalg.inv(self.T_imu_cam1)
        self.T_cam1_cam0 = self.T_cam1_imu @ self.T_imu_cam0  # Camera 0 to Camera 1
        self.R_cam1_cam0 = self.T_cam1_cam0[:3, :3]  # Rotation from Camera 0 to Camera 1
        self.t_cam1_cam0 = self.T_cam1_cam0[:3, 3]
        self.R_align_IMU_to_gravity = np.eye(3) # Will be computed after loading data


    def load_data(self):
        cam_path = os.path.join(self.data_dir, "cam0/data.csv")
        imu_path = os.path.join(self.data_dir, "imu0/data.csv")
        gt_path = os.path.join(self.data_dir, "state_groundtruth_estimate0/data.csv")
        sensor_data0 = os.path.join(self.data_dir, "cam0/sensor.yaml")
        sensor_data1 = os.path.join(self.data_dir, "cam1/sensor.yaml")
        # read T_BS from sensor data
        
        
        self.cam_df = pd.read_csv(cam_path)
        self.imu_df = pd.read_csv(imu_path)
        self.gt_df = pd.read_csv(gt_path)
        self.cam_df.columns = ["timestamp", "filename"]
        self.imu_df.columns = ["timestamp","w_x", "w_y", "w_z","a_x", "a_y", "a_z"]
        self.gt_df.columns = ["timestamp",  "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z",
            "v_x",
            "v_y",
            "v_z",
            "w_x",
            "w_y",
            "w_z",
            "a_x",
            "a_y",
            "a_z",
        ]
        # Convert timestamps to seconds
        self.cam_df["timestamp"] = self.cam_df["timestamp"].astype(float) * 1e-9
        self.imu_df["timestamp"] = self.imu_df["timestamp"].astype(float) * 1e-9
        self.gt_df["timestamp"] = self.gt_df["timestamp"].astype(float) * 1e-9
        print("Data loaded successfully.")
        return True

    def get_initial_state_from_gt(self):
        # Extract initial state from the first row of ground truth
        initial_gt = self.gt_df.iloc[0]

        pos = initial_gt[["p_x", "p_y", "p_z"]].values
        vel = initial_gt[["v_x", "v_y", "v_z"]].values
        quat = initial_gt[["q_x", "q_y", "q_z", "q_w"]].values # Scipy format [x,y,z,w]
        
        # # Transform from Vicon frame to Body frame
        # pos_body, vel_body, quat_body = self.transform_vicon_to_body_frame(pos, vel, quat)
        
        # Return pose array [x,y,z, qx,qy,qz,qw] and velocity vector
        return pos, vel, quat

    def find_static_window(
        self,
        duration_sec=3.0,
        anchor_time=None,
        search_margin_sec=10.0,
        stride_sec=0.05,
        gyro_std_thresh=0.01,
        accel_std_thresh=1.0,
        gravity_dev_thresh=0.3,
        verbose=True,
    ):
        """Find the most static `duration_sec` window of IMU data.

        Gyro bias and gravity direction are only observable at rest, and the
        attitude read off gravity is only valid at the instant it was measured —
        so the window must be located by measurement, not assumed at t=0.

        Candidates are scored by accel std + gyro std + |accel| deviation from
        9.81, each normalized by its threshold so the terms are comparable; the
        minimum wins. Thresholds only set the `is_static` warning flag.
        `anchor_time` confines the search near where processing begins, so the
        recovered attitude matches the first pose in the graph.

        Returns a dict describing the winning window (indices, timestamps,
        accel/gyro means and stds, `up_body` gravity direction, is_static).
        """
        t = self.imu_df["timestamp"].values.astype(float)
        accel = self.imu_df[["a_x", "a_y", "a_z"]].values.astype(float)
        gyro = self.imu_df[["w_x", "w_y", "w_z"]].values.astype(float)
        n = len(t)

        # Infer sample rate from the data rather than hardcoding 200 Hz.
        dt_median = float(np.median(np.diff(t))) if n > 1 else 0.005
        hz = 1.0 / max(dt_median, 1e-9)
        win = max(int(round(duration_sec * hz)), 2)
        step = max(int(round(stride_sec * hz)), 1)

        if win >= n:
            raise ValueError(
                f"IMU record too short ({n} samples) for a {duration_sec}s window."
            )

        lo, hi = 0, n - win
        if anchor_time is not None:
            lo = max(int(np.searchsorted(
                t, anchor_time - duration_sec - search_margin_sec)), 0)
            hi = min(int(np.searchsorted(
                t, anchor_time + search_margin_sec)) - win, n - win)
            if hi < lo:  # anchor near a record boundary — search globally
                lo, hi = 0, n - win

        # Sliding mean/std via cumulative sums: O(n) instead of O(n * win).
        def _windowed_mean_std(x):
            c1 = np.concatenate([np.zeros((1, 3)), np.cumsum(x, axis=0)])
            c2 = np.concatenate([np.zeros((1, 3)), np.cumsum(x ** 2, axis=0)])
            starts = np.arange(lo, hi + 1, step)
            mean = (c1[starts + win] - c1[starts]) / win
            msq = (c2[starts + win] - c2[starts]) / win
            return starts, mean, np.sqrt(np.maximum(msq - mean ** 2, 0.0))

        starts, accel_mean, accel_std = _windowed_mean_std(accel)
        _, gyro_mean, gyro_std = _windowed_mean_std(gyro)

        accel_std_n = np.linalg.norm(accel_std, axis=1)
        gyro_std_n = np.linalg.norm(gyro_std, axis=1)
        gravity_norm = np.linalg.norm(accel_mean, axis=1)
        gravity_dev = np.abs(gravity_norm - 9.81)

        score = (
            accel_std_n / accel_std_thresh
            + gyro_std_n / gyro_std_thresh
            + gravity_dev / gravity_dev_thresh
        )
        k = int(np.argmin(score))
        s = int(starts[k])
        e = s + win

        a_mean = accel_mean[k]
        up_body = a_mean / np.linalg.norm(a_mean)
        is_static = bool(
            accel_std_n[k] < accel_std_thresh
            and gyro_std_n[k] < gyro_std_thresh
            and gravity_dev[k] < gravity_dev_thresh
        )

        window = {
            "start_idx": s,
            "end_idx": e,
            "t_start": float(t[s]),
            "t_end": float(t[e - 1]),
            "n_samples": win,
            "accel_mean": a_mean,
            "up_body": up_body,
            "gyro_mean": gyro_mean[k],
            "accel_std": accel_std[k],
            "gyro_std": gyro_std[k],
            "gravity_norm": float(gravity_norm[k]),
            "score": float(score[k]),
            "is_static": is_static,
        }
        self.static_window = window

        if verbose:
            t0 = float(t[0])
            print(
                f"  [StaticDetect] best {duration_sec:.1f}s window: "
                f"t = {window['t_start'] - t0:.2f}..{window['t_end'] - t0:.2f}s "
                f"(samples {s}..{e}, {hz:.0f} Hz)"
            )
            print(
                f"    |accel| = {window['gravity_norm']:.4f} m/s^2  "
                f"accel_std = {np.round(window['accel_std'], 4)}  "
                f"gyro_std = {np.round(window['gyro_std'], 5)}"
            )
            print(f"    gravity direction in body frame: {np.round(up_body, 5)}")
            if anchor_time is not None:
                print(
                    f"    anchored at t = {anchor_time - t0:.2f}s "
                    f"(window ends {window['t_end'] - anchor_time:+.2f}s from it)"
                )
            if not is_static:
                print(
                    "    [WARN] best window still looks dynamic — gravity direction "
                    "and gyro bias may be corrupted by motion."
                )
        return window

    def calculate_initial_biases_and_gravity(
        self, static_duration_sec=2, anchor_time=None, auto_detect=True
    ):
        """
        Calculates IMU biases and performs gravity alignment.

        Returns:
            true_bias_accel (np.array): Accelerometer bias (assumed zero).
            bias_gyro (np.array): Gyroscope bias.
            initial_orientation_quat (np.array): Initial orientation [x,y,z,w] to align IMU with a gravity-defined world frame.
            world_gravity_vec (np.array): The ideal gravity vector in the world frame [0, 0, -g].
        """

        if auto_detect:
            w = self.find_static_window(
                duration_sec=static_duration_sec, anchor_time=anchor_time
            )
            lo, hi = w["start_idx"], w["end_idx"]
        else:
            imu_hz = 200  # Assuming 200Hz from your project
            lo, hi = 0, int(static_duration_sec * imu_hz)

        accel_data = self.imu_df[["a_x", "a_y", "a_z"]].values[lo:hi]
        gyro_data = self.imu_df[["w_x", "w_y", "w_z"]].values[lo:hi]

        # 1. Calculate biases
        bias_gyro = gyro_data.mean(axis=0)
        # Accel bias stays zero: from a single static attitude it is not
        # separable from gravity magnitude/direction. ISAM2 refines it online.
        true_bias_accel = np.zeros(3)

        # 2. Measure gravity vector in the IMU's local frame
        gravity_imu_frame = accel_data.mean(axis=0)
        gravity_norm = np.linalg.norm(gravity_imu_frame)
        gravity_imu_frame /= gravity_norm  # Normalize to unit vector
        print(f"Measured gravity vector in IMU frame: {gravity_imu_frame} (Magnitude: {gravity_norm:.2f})")
        print(f"Estimated gyroscope bias: {bias_gyro}")

        # 3. Define the ideal world frame gravity vector
        world_gravity_vec = np.array([0.0, 0.0, gravity_norm])

        # 4. Calculate the initial orientation that aligns the IMU frame with the world frame
        # We want to find the rotation that maps the measured gravity to the ideal gravity.
        initial_orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Default to identity quaternion
        #initial_orientation_quat = from_two_vectors3(gravity_imu_frame, world_gravity_vec)
        # Verify
 
        #world_gravity_vec[2] = -world_gravity_vec[2]  # Ensure gravity is the opposite
        return true_bias_accel, bias_gyro, initial_orientation_quat, world_gravity_vec

    def compute_alignment_rotation(self):
        """
        Computes the rotation matrix to align the Vicon/GT world frame
        with the gravity-aligned world frame.
        """
        from scipy.spatial.transform import Rotation as R

        # Get the initial orientation from GT (Vicon -> Body)
        _, _, initial_gt_quat = self.get_initial_state_from_gt()

        # Get the initial orientation from gravity alignment (Gravity World -> Body)
        _, _, initial_gravity_quat, _ = self.calculate_initial_biases_and_gravity(static_duration_sec=2) # Use short duration

        # Convert to rotation objects
        #r_vicon_to_body = R.from_quat(initial_gt_quat)
        r_body_to_gravity = R.from_quat(initial_gravity_quat)

        # We want R_gravity_to_vicon.
        # R_gravity_to_vicon = R_gravity_to_body * inv(R_vicon_to_body)
        self.R_align_IMU_to_gravity = (r_body_to_gravity).as_matrix()
        #self.R_align_IMU_to_gravity = self.R_align_IMU_to_gravity.T  # Transpose to get the correct alignment
        print("Computed IMU-to-Gravity alignment rotation.")

    def get_synced_gt_poses(self, timestamp):
        """
        Finds the closest GT pose and aligns it to the gravity world frame.
        """
        if self.gt_df is None or self.gt_df.empty:
            return None, None

        idx_gt = np.argmin(np.abs(self.gt_df["timestamp"].values - timestamp))
        gt_pose_row = self.gt_df.iloc[idx_gt]

        pos_world = gt_pose_row[["p_x", "p_y", "p_z"]].values
        quat_world = gt_pose_row[["q_x", "q_y", "q_z", "q_w"]].values

        # Align the Vicon GT data to our gravity-defined world
        # pos_aligned, quat_aligned = align_ground_truth_to_gravity_world(
        #     pos_vicon, quat_vicon, self.R_align_vicon_to_gravity
        # )
        return pos_world, quat_world

    def load_camera_calib(self, cam_id):
        if cam_id not in ["cam0", "cam1"]:
            raise ValueError("Invalid camera ID. Use 'cam0' or 'cam1'.")
        
        w, h = 752, 480  # Assuming fixed resolution for both cameras
        if cam_id == "cam0":
            fu, fv, cu, cv = [458.654, 457.296, 367.215, 248.375]
            K = np.array([[fu, 0, cu],
                          [0, fv, cv],
                          [0, 0, 1]])
            dist_coeffs = np.array( [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
            self.K_cam0 = np.copy(K)
            self.dist_coeffs_cam0 = np.copy(dist_coeffs)
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1)

        elif cam_id == "cam1":
            fu, fv, cu, cv = [457.587, 456.134, 379.999, 255.238]
            K = np.array([[fu, 0, cu],
                          [0, fv, cv],
                          [0, 0, 1]])
            dist_coeffs = np.array([-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05])
            self.K_cam1 = np.copy(K)
            self.dist_coeffs_cam1 = np.copy(dist_coeffs)
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1)


        return K, dist_coeffs, new_K

    def init_rectification(self):
        """Build the rectification maps eagerly, before any frame is read.

        R1/R2/P1/P2 are otherwise created lazily inside the first
        rectify_stereo_images() call, which happens on the frontend thread — too
        late for the backend, which needs R1 at construction time to express the
        camera extrinsic in the rectified frame. Reads one image only to learn
        the image size.

        Returns (P1, P2).
        """
        if not hasattr(self, "map1_left"):
            row = self.cam_df.iloc[0]
            name = row.get("left_img_path") or row.get("filename_left") or row.get("filename")
            img = cv2.imread(
                os.path.join(self.data_dir, "cam0", "data", name), cv2.IMREAD_GRAYSCALE
            )
            if img is None:
                raise RuntimeError(f"Could not read first cam0 image: {name}")
            self.image_size = (img.shape[1], img.shape[0])
            self.rectify_stereo_images(img, img)
        return self.P1, self.P2

    def T_imu_rect_cam0(self):
        """Extrinsic of the RECTIFIED left camera in the body/IMU frame.

        Rectification rotates the camera frame about its optical center by R1
        (p_rect = R1 @ p_cam0), so the calibrated T_imu_cam0 does not describe
        the frame that rectified pixels and stereo triangulations live in.
        Composing with R1^T undoes that rotation; the translation is unchanged.
        """
        self.init_rectification()
        T_rect = np.eye(4)
        T_rect[:3, :3] = self.R1.T
        return self.T_imu_cam0 @ T_rect

    def iter_stereo_frames(self, step=1, start_frame=0):
        """
        Generator yielding (row, left_img, right_img) for each stereo frame.
        Assumes self.cam_df has columns: 'timestamp', 'left_img_path', 'right_img_path'
        """
        for idx, row in self.cam_df.iloc[start_frame::step].iterrows():
            left_img_path = row.get("left_img_path") or row.get("filename_left") or row.get("filename")
            right_img_path = row.get("right_img_path") or row.get("filename_right")
            # Fallback: try to infer right image path from left
            if right_img_path is None and left_img_path is not None:
                right_img_path = left_img_path.replace("cam0", "cam1")
            if left_img_path is None or right_img_path is None:
                continue

            left_img = cv2.imread(os.path.join(self.data_dir, "cam0","data", left_img_path), cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(os.path.join(self.data_dir, "cam1","data", right_img_path), cv2.IMREAD_GRAYSCALE)
            if left_img is None or right_img is None:
                continue
            self.image_size = (left_img.shape[1], left_img.shape[0])  # (width, height)
            rectified_left, rectified_right = self.rectify_stereo_images(left_img, right_img)
            if rectified_left is None or rectified_right is None:
                continue
            yield row, rectified_left, rectified_right

    def rectify_stereo_images(self, curr_cam0, curr_cam1):
        """
        Rectify stereo images once at the beginning of processing.
        This creates rectified image pairs where epipolar lines are horizontal.
        """
        # Compute rectification maps once (can be done in __init__)
        if not hasattr(self, 'map1_left'):
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                self.K_cam0,  # Camera 0 intrinsics
                self.dist_coeffs_cam0,  # Camera 0 distortion
                self.K_cam1,  # Camera 1 intrinsics  
                self.dist_coeffs_cam1,  # Camera 1 distortion
                self.image_size,  # Image size
                self.R_cam1_cam0,  # Rotation between cameras
                self.t_cam1_cam0,  # Translation between cameras
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0  # Free scaling parameter
            )
            # print("K_cam0:\n", self.K_cam0)
            # print("dist_cam0:\n", self.dist_coeffs_cam0)
            # print("K_cam1:\n", self.K_cam1) 
            # print("dist_cam1:\n", self.dist_coeffs_cam1)
            # print("R_cam0_cam1:\n", self.R_cam0_cam1)
            # print("t_cam0_cam1:\n", self.t_cam0_cam1)
            # print("Image size:", self.image_size)
            # print("R1:\n", R1)
            # print("R2:\n", R2)
            # print("P1:\n", P1)
            # print("P2:\n", P2)
            self.P1 = P1  # Rectified projection matrix for cam0
            self.P2 = P2  # Rectified projection matrix for cam1
            self.R1 = R1  # Rectification rotation for cam0
            self.R2 = R2  # Rectification rotation for cam1
            # Create rectification maps
            self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.K_cam0, self.dist_coeffs_cam0, R1, P1,
            self.image_size, cv2.CV_16SC2
        )
            
            self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
                self.K_cam1, self.dist_coeffs_cam1, R2, P2,
                self.image_size, cv2.CV_16SC2
    )
        
        # Rectify current images
        cam0_rect = cv2.remap(curr_cam0, self.map1_left, self.map2_left, 
                            cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        cam1_rect = cv2.remap(curr_cam1, self.map1_right, self.map2_right,
                            cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        return cam0_rect, cam1_rect


    def iter_imu_between(self, start_timestamp, end_timestamp):
        """
        Generator yielding IMU rows between two timestamps.
        """
        mask = (self.imu_df["timestamp"] > start_timestamp) & (self.imu_df["timestamp"] <= end_timestamp)
        for _, imu_row in self.imu_df[mask].iterrows():
            yield imu_row
    
    def get_baseline(self, K1, K2):
        baseline = np.linalg.norm(self.T_cam1_cam0[:3, 3])  # distance in meters

        return baseline
    
    def get_alignment_matrix(self):
        """Return the computed alignment matrix for use by other components."""
        return self.R_align_IMU_to_gravity
    
    def get_projection_matrices(self):
        """Return the rectified projection matrices for both cameras."""
        return self.P1, self.P2
