"""
VIO Visualizer - Real-time visualization for tightly-coupled VIO/SLAM using Rerun.

Displays:
  - Left/Right camera frames with detected landmarks overlaid
  - 3D landmark point cloud (accumulated map)
  - GTSAM-optimized trajectory with pose axes
  - Time-series: optimization error, observation count, IMU sample count
  - System status metrics
"""

import numpy as np
import cv2
import time

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    raise ImportError("rerun-sdk is required. Install with: pip install rerun-sdk")


class VIOVisualizer:
    """Real-time multi-panel visualizer for VIO/SLAM pipeline using Rerun."""

    def __init__(self, max_trail_length=500, update_interval=0.05):
        """
        Args:
            max_trail_length: Max number of poses to keep in the trajectory trail.
            update_interval: Minimum seconds between updates (throttle).
        """
        self.max_trail_length = max_trail_length
        self.update_interval = update_interval

        # Accumulated data
        self.trajectory_poses = []       # List of (x, y, z)
        self.trajectory_orientations = []  # List of 3x3 rotation matrices
        self.all_landmarks_3d = {}       # {landmark_id: np.array([x,y,z])}
        self.gt_trajectory_poses = []    # Ground truth positions
        self.frame_idx = 0

        # Timing
        self._last_update_time = 0

        # Initialize Rerun
        self._init_rerun()

    def _init_rerun(self):
        """Initialize Rerun viewer with a layout blueprint."""
        rr.init("VIO_SLAM", spawn=True)

        # Define the viewer layout
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="3D Map",
                    origin="/world",
                ),
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(name="Left Camera", origin="/camera/left"),
                        rrb.Spatial2DView(name="Right Camera", origin="/camera/right"),
                    ),
                    rrb.Horizontal(
                        rrb.TimeSeriesView(name="Optimization Error", origin="/metrics/error"),
                        rrb.TimeSeriesView(name="Observations", origin="/metrics/observations"),
                    ),
                    rrb.TextDocumentView(name="Pose Status", origin="/status/pose"),
                ),
                column_shares=[2, 1],
            )
        )
        rr.send_blueprint(blueprint)

        # Set up the world coordinate system (Z-up for EuRoC)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def update(
        self,
        frame_idx,
        left_img=None,
        right_img=None,
        observations=None,
        new_landmarks_3d=None,
        all_landmarks=None,
        gtsam_estimate=None,
        num_states=0,
        imu_samples_count=0,
        metrics=None,
        loop_closure_ids=None,
        gt_pose=None,
        pose_covariance=None,
        ate_rmse=None,
    ):
        """
        Main update call - feed data from the VIO loop each frame.

        Args:
            frame_idx: Current frame index.
            left_img: Left camera image (grayscale or BGR).
            right_img: Right camera image (grayscale or BGR).
            observations: List of (landmark_id, frame_id, uv) from feature pipeline.
            new_landmarks_3d: Dict {landmark_id: np.array([x,y,z])} new landmarks this frame.
            all_landmarks: Dict of all landmarks {id: pos} from feature pipeline.
            gtsam_estimate: GTSAM Values object (current estimate).
            num_states: Number of state nodes in the graph.
            imu_samples_count: Number of IMU samples integrated this frame.
            metrics: Dict from optimizer.diagnostics().
            gt_pose: Ground truth position as np.array([x, y, z]) or None.
            pose_covariance: 3x3 position covariance matrix or None.
            ate_rmse: Current ATE RMSE value (meters) or None.
        """
        self.frame_idx = frame_idx

        # Set the timeline
        rr.set_time("frame", sequence=frame_idx)

        # --- Update accumulated data ---
        # Use GTSAM's optimized landmarks (world frame) instead of raw camera-frame ones
        if gtsam_estimate is not None:
            self._accumulate_gtsam_landmarks(gtsam_estimate)

        # Extract trajectory from GTSAM estimate
        if gtsam_estimate is not None:
            self._extract_trajectory(gtsam_estimate, num_states)

        # --- Accumulate ground truth trajectory ---
        if gt_pose is not None:
            self.gt_trajectory_poses.append(gt_pose)

        # --- Log camera images with feature overlays ---
        if left_img is not None:
            self._log_camera_images(left_img, right_img, observations)

        # --- Log 3D map (landmarks + trajectory) ---
        self._log_3d_map(loop_closure_ids)

        # --- Log trajectory poses ---
        self._log_trajectory()

        # --- Log pose uncertainty ellipsoid ---
        if pose_covariance is not None and len(self.trajectory_poses) > 0:
            self._log_covariance_ellipsoid(pose_covariance)

        # --- Log time-series metrics ---
        self._log_metrics(metrics, imu_samples_count, observations, ate_rmse)

    def _accumulate_gtsam_landmarks(self, estimate):
        """Extract optimized landmarks from GTSAM and accumulate into the point cloud."""
        import gtsam

        for key in estimate.keys():
            sym = gtsam.Symbol(key)
            if chr(sym.chr()) == 'l':
                lm_id = sym.index()
                pt = estimate.atPoint3(key)
                self.all_landmarks_3d[lm_id] = np.array([pt[0], pt[1], pt[2]])

    def _extract_trajectory(self, estimate, num_states):
        """Extract pose trajectory from GTSAM Values."""
        import gtsam
        from gtsam.symbol_shorthand import X

        self.trajectory_poses = []
        self.trajectory_orientations = []
        for i in range(num_states):
            try:
                if estimate.exists(X(i)):
                    pose = estimate.atPose3(X(i))
                    t = pose.translation()
                    self.trajectory_poses.append(np.array([t[0], t[1], t[2]]))
                    self.trajectory_orientations.append(pose.rotation().matrix())
            except Exception:
                continue

    def _log_camera_images(self, left_img, right_img, observations):
        """Log camera images with feature point overlays."""
        # Prepare left image
        left_vis = self._to_rgb(left_img)
        rr.log("camera/left/image", rr.Image(left_vis))

        # Log feature points on left image
        if observations:
            points = np.array([[uv[0], uv[1]] for _, _, uv in observations], dtype=np.float32)
            # Color by landmark ID for consistency
            colors = np.array([self._id_to_color(lm_id) for lm_id, _, _ in observations], dtype=np.uint8)
            rr.log(
                "camera/left/features",
                rr.Points2D(points, colors=colors, radii=3.0),
            )

        # Right camera
        if right_img is not None:
            right_vis = self._to_rgb(right_img)
            rr.log("camera/right/image", rr.Image(right_vis))

            if observations:
                rr.log(
                    "camera/right/features",
                    rr.Points2D(points, colors=colors, radii=3.0),
                )

    def _log_3d_map(self, loop_closure_ids=None):
        """Log the 3D landmark point cloud, highlighting loop closure landmarks."""
        if not self.all_landmarks_3d:
            return

        lm_ids = list(self.all_landmarks_3d.keys())
        pts = np.array(list(self.all_landmarks_3d.values()), dtype=np.float32)
        if len(pts) == 0:
            return

        # Color by depth (distance from camera/origin)
        depths = np.linalg.norm(pts, axis=1)
        d_min, d_max = depths.min(), depths.max()
        if d_max - d_min > 0.01:
            normalized = (depths - d_min) / (d_max - d_min)
        else:
            normalized = np.zeros_like(depths)

        # Use OpenCV's Viridis colormap for correct coloring
        gray = (normalized * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(gray.reshape(-1, 1), cv2.COLORMAP_VIRIDIS)
        colors = colormap.reshape(-1, 3)[:, ::-1]  # BGR -> RGB

        rr.log(
            "world/landmarks",
            rr.Points3D(pts, colors=colors, radii=0.02),
        )

        # Highlight loop closure landmarks in magenta with ID labels
        if loop_closure_ids:
            lc_set = set(loop_closure_ids)
            lc_pts = []
            lc_labels = []
            for lm_id in lc_set:
                if lm_id in self.all_landmarks_3d:
                    lc_pts.append(self.all_landmarks_3d[lm_id])
                    lc_labels.append(f"LC:{lm_id}")
            if lc_pts:
                lc_pts_arr = np.array(lc_pts, dtype=np.float32)
                rr.log(
                    "world/loop_closure_landmarks",
                    rr.Points3D(
                        lc_pts_arr,
                        colors=[[0, 255, 255]] * len(lc_pts_arr),
                        radii=0.06,
                        labels=lc_labels,
                    ),
                )

    def _log_trajectory(self):
        """Log the camera trajectory as a line strip with pose frames."""
        if len(self.trajectory_poses) < 2:
            return

        traj = np.array(self.trajectory_poses, dtype=np.float32)

        # Log trajectory as a line strip
        rr.log(
            "world/trajectory",
            rr.LineStrips3D([traj], colors=[[255, 50, 50]]),
        )

        # Log ground truth trajectory in green (shifted to start at same origin as estimate)
        if len(self.gt_trajectory_poses) >= 2:
            gt_traj = np.array(self.gt_trajectory_poses, dtype=np.float32)
            # Shift GT so its first point aligns with the estimated trajectory's first point
            gt_offset = self.trajectory_poses[0] - gt_traj[0]
            gt_traj_aligned = gt_traj + gt_offset
            rr.log(
                "world/ground_truth",
                rr.LineStrips3D([gt_traj_aligned], colors=[[50, 255, 50, 140]]),
            )

        # Log current pose as a coordinate frame
        if self.trajectory_orientations:
            pos = self.trajectory_poses[-1]
            rot = self.trajectory_orientations[-1]
            rr.log(
                "world/current_pose",
                rr.Transform3D(
                    translation=pos,
                    mat3x3=rot,
                ),
            )
            rr.log(
                "world/current_pose/axes",
                rr.Arrows3D(
                    origins=[[0, 0, 0]] * 3,
                    vectors=[[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                ),
            )

        # Log every Nth pose axis for trajectory orientation context
        step = max(1, len(self.trajectory_poses) // 15)
        origins = []
        vectors = []
        colors = []
        for i in range(0, len(self.trajectory_poses), step):
            pos = self.trajectory_poses[i]
            rot = self.trajectory_orientations[i]
            scale = 0.15
            for axis_idx, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                origins.append(pos)
                vectors.append((rot[:, axis_idx] * scale).tolist())
                colors.append(color)

        if origins:
            rr.log(
                "world/pose_axes",
                rr.Arrows3D(
                    origins=origins,
                    vectors=vectors,
                    colors=colors,
                ),
            )

        # Log a black point at each pose center
        pose_centers = np.array(self.trajectory_poses, dtype=np.float32)
        rr.log(
            "world/pose_points",
            rr.Points3D(pose_centers, colors=[[0, 0, 0]] * len(pose_centers), radii=0.03),
        )

    def _log_metrics(self, metrics, imu_samples_count, observations, ate_rmse=None):
        """Log time-series metrics."""
        # Optimization error
        if metrics:
            rr.log(
                "metrics/error/avg_per_factor",
                rr.Scalars(metrics.get("avg_error_per_factor", 0)),
            )
            rr.log(
                "metrics/error/total",
                rr.Scalars(metrics.get("total_error", 0)),
            )

        # ATE
        if ate_rmse is not None:
            rr.log("metrics/error/ate_rmse_m", rr.Scalars(ate_rmse))

        # Observation count
        obs_count = len(observations) if observations else 0
        rr.log("metrics/observations/count", rr.Scalars(obs_count))

        # IMU samples
        rr.log("metrics/observations/imu_samples", rr.Scalars(imu_samples_count))

        # Graph size
        if metrics:
            rr.log("metrics/observations/n_landmarks", rr.Scalars(metrics.get("n_landmarks", 0)))
            rr.log("metrics/observations/n_factors", rr.Scalars(metrics.get("n_factors", 0)))

        # Travel distance (time-series + text)
        if len(self.trajectory_poses) > 1:
            traj = np.array(self.trajectory_poses)
            diffs = np.diff(traj, axis=0)
            total_dist = float(np.sum(np.linalg.norm(diffs, axis=1)))
            rr.log("metrics/error/travel_distance_m", rr.Scalars(total_dist))

        # Current pose: rotation and translation as readable text
        self._log_pose_text()

    # ==================== Utility Methods ====================

    def _log_pose_text(self):
        """Log current frame's rotation and translation as readable text."""
        if not self.trajectory_poses or not self.trajectory_orientations:
            return

        pos = self.trajectory_poses[-1]
        rot = self.trajectory_orientations[-1]

        # Compute travel distance
        total_dist = 0.0
        if len(self.trajectory_poses) > 1:
            traj = np.array(self.trajectory_poses)
            diffs = np.diff(traj, axis=0)
            total_dist = float(np.sum(np.linalg.norm(diffs, axis=1)))

        # Compute inter-frame delta (relative to previous pose)
        if len(self.trajectory_poses) >= 2:
            prev_pos = self.trajectory_poses[-2]
            prev_rot = self.trajectory_orientations[-2]
            dt = pos - prev_pos
            # Relative rotation: R_rel = R_prev^T * R_curr
            r_rel = prev_rot.T @ rot
            # Convert to axis-angle magnitude (rotation angle in degrees)
            cos_angle = (np.trace(r_rel) - 1.0) / 2.0
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
            delta_text = (
                f"Delta Translation: [{dt[0]:+.4f}, {dt[1]:+.4f}, {dt[2]:+.4f}] m\n"
                f"Delta Rotation:    {angle_deg:.2f} deg"
            )
        else:
            delta_text = "Delta: N/A (first frame)"

        text = (
            f"## Frame {self.frame_idx} Pose\n\n"
            f"**Translation (world):**\n"
            f"  x = {pos[0]:+.4f} m\n"
            f"  y = {pos[1]:+.4f} m\n"
            f"  z = {pos[2]:+.4f} m\n\n"
            f"**Rotation (world):**\n"
            f"  [{rot[0,0]:+.4f} {rot[0,1]:+.4f} {rot[0,2]:+.4f}]\n"
            f"  [{rot[1,0]:+.4f} {rot[1,1]:+.4f} {rot[1,2]:+.4f}]\n"
            f"  [{rot[2,0]:+.4f} {rot[2,1]:+.4f} {rot[2,2]:+.4f}]\n\n"
            f"**{delta_text}**\n\n"
            f"---\n"
            f"**Total Distance Traveled: {total_dist:.3f} m**"
        )

        rr.log("status/pose", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN))

    def _log_covariance_ellipsoid(self, pos_cov, scale=3.0):
        """Log a 3-sigma uncertainty ellipsoid at the current pose.

        Args:
            pos_cov: 3x3 position covariance matrix.
            scale: Number of sigma for the ellipsoid (3.0 = 99.7% confidence).
        """
        # Eigen-decomposition: covariance = V * diag(eigenvalues) * V^T
        eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)

        # Clamp negative eigenvalues (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        # Half-sizes = scale * sqrt(eigenvalue) = scale * sigma along each axis
        half_sizes = scale * np.sqrt(eigenvalues)

        # Current pose position
        pos = self.trajectory_poses[-1]

        # Eigenvectors form the rotation matrix for the ellipsoid orientation
        rot_matrix = eigenvectors  # 3x3 rotation aligning ellipsoid axes to world

        # Log transform to position/orient the ellipsoid
        rr.log(
            "world/covariance_ellipsoid",
            rr.Transform3D(
                translation=pos,
                mat3x3=rot_matrix,
            ),
        )
        rr.log(
            "world/covariance_ellipsoid/ellipsoid",
            rr.Ellipsoids3D(
                half_sizes=[half_sizes.tolist()],
                colors=[[255, 165, 0, 60]],  # Orange, semi-transparent
            ),
        )

    def _to_rgb(self, img):
        """Convert image to RGB for Rerun (expects RGB)."""
        if img is None:
            return np.zeros((480, 752, 3), dtype=np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _id_to_color(self, landmark_id):
        """Generate a consistent RGB color from a landmark ID."""
        np.random.seed(landmark_id % 10000)
        return np.random.randint(50, 255, 3).tolist()

    def close(self):
        """Clean up (Rerun handles its own window lifecycle)."""
        pass
