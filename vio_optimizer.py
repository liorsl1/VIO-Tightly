import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import gtsam
    from gtsam.symbol_shorthand import X, V, B, L
except ImportError:  # Allow file existence without immediate dependency
    gtsam = None  # type: ignore


class GraphOptimizer:
    """
    Tightly coupled VIO/SLAM backend using GTSAM with explicit landmarks.

    Uses GenericProjectionFactorPose3Point3 for visual factors, allowing:
      - Incremental observation addition to existing landmarks in ISAM2
      - Direct access to optimized 3D landmark positions
      - Natural loop closure integration via re-observation factors
      - Robust Huber loss on reprojection errors
    """

    def __init__(
        self, use_isam: bool = True, body_P_sensor: np.ndarray = None, imu_calib=None
    ):
        if gtsam is None:
            self.isam = None
            self.graph = None
            self.initial = None
            self._depth_stats = {"attempted": 0, "promoted": 0, "outlier": 0, "pending": 0}
            self.promoted_depth_log = []
            self._static_promotion_done = False
            self.landmark_buffer = {}
            self.landmark_initialized = set()
            self.landmark_obs_count = {}
            return

        self.imu_calib = imu_calib  # IMUCalibration for bias noise computation

        # body_P_sensor: 4x4 transform T_body_cam (pose of camera in body/IMU frame)
        if body_P_sensor is not None:
            R = gtsam.Rot3(body_P_sensor[:3, :3])
            t = gtsam.Point3(body_P_sensor[:3, 3])
            self.body_P_sensor = gtsam.Pose3(R, t)
        else:
            self.body_P_sensor = None

        self.use_isam = use_isam
        self.isam_params = gtsam.ISAM2Params()
        # If I want to use Dogleg insead of standard Gauss-Newton, I can set these parameters:
        # dogleg_params = gtsam.ISAM2DoglegParams()
        # dogleg_params.setInitialDelta(0.5)
        # self.isam_params.setOptimizationParams(dogleg_params)
        self.isam_params.setRelinearizeThreshold(0.03)
        self.isam_params.relinearizeSkip = 2
        self.isam_params.cacheLinearizedFactors = True
        # self.isam_params.evaluateNonlinearError = True
        self.isam = gtsam.ISAM2(self.isam_params) if use_isam else None
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.state_index = 0

        # Batch-mode (LM) accumulated graph and values
        # In ISAM2 mode these are unused — ISAM2 keeps its own internal copy.
        # In batch mode, graph/initial are pending additions that get merged into
        # these accumulators before each LM solve.
        self._batch_graph = gtsam.NonlinearFactorGraph() if not use_isam else None
        self._batch_values = gtsam.Values() if not use_isam else None

        # --- Landmark management ---
        self.landmark_initialized = set()  # L(id) already in ISAM2 Values
        self.landmark_obs_count = {}  # {landmark_id: number of observations added}
        self.landmark_frozen = set()  # Landmarks frozen due to high uncertainty
        # Buffer: landmarks wait here until they have 2+ observations from different poses
        # {landmark_id: {"point_cam": np.array, "first_state": int, "observations": [(state_idx, uv), ...]}}
        self.landmark_buffer = {}
        # Camera center (world) of the last view that contributed a factor for each
        # initialized landmark. Used to gate new observations by incremental parallax.
        self.landmark_last_factor_cam = {}
        self.cal = None  # gtsam.Cal3_S2, set on first observation

        # Cached estimate (invalidated after each optimize call)
        self._cached_estimate = None

        # --- Noise models ---
        # Prior: tight rotation (0.03 rad ≈ 1.7°), moderate translation (0.2m)
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
        )
        self.prior_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.02)
        # Bias prior: anchors near calibrated values but loose enough for online refinement.
        # Zero-motion constraints during static periods provide the geometric anchoring,
        # so the bias prior can be relaxed to allow faster convergence.
        self.prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.03, 0.03, 0.03, 0.01, 0.01, 0.01])
        )
        # Landmark regularization prior (sigma in meters).
        # Tight enough to prevent ISAM2 from pushing landmarks to degenerate positions
        # during relinearization, but loose enough not to bias converged estimates.
        # Tunable at runtime via self.landmark_reg_sigma — the noise model is rebuilt
        # on each landmark promotion to reflect the current setting.
        self.landmark_reg_sigma = 1  # meters (≈3× max depth filter uncertainty)

        # Robust pixel noise (Huber) for projection factors
        pixel_sigma = 0.5  # pixels
        pixel_noise_base = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
        self.pixel_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), pixel_noise_base
        )

        # Tighter robust pixel noise for loop-closure reprojection factors.
        # A single loop re-observation must compete against the whole accumulated
        # IMU+visual chain, so it carries little weight at the standard 1.5px sigma.
        # Using a smaller sigma (more information per factor) lets a handful of loop
        # re-observations actually pull the trajectory back toward the closed loop.
        loop_pixel_sigma = 0.2  # pixels
        loop_pixel_noise_base = gtsam.noiseModel.Isotropic.Sigma(2, loop_pixel_sigma)
        self.loop_pixel_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.3), loop_pixel_noise_base
        )

        # Spatial distribution: grid bucketing (cell_size in pixels)
        self.obs_cell_size = 13  # pixels — one observation per 13x13 cell per frame
        self._frame_occupied_cells = {}  # {state_idx: set of (row, col) tuples}

        # Tunable thresholds (can be changed at runtime by RL agent)
        self.parallax_threshold = 2.5  # degrees — minimum parallax for landmark promotion

        # Minimum incremental parallax (deg) required to add a new observation factor to
        # an already-initialized landmark. A re-observation whose camera has barely moved
        # (relative to the last contributing view) adds little geometric information but,
        # at a tight pixel sigma, can over-constrain the pose; gating it improves accuracy.
        self.min_obs_parallax_deg = 2

        # Parallax instrumentation: max-parallax angles (deg) computed during landmark
        # promotion attempts since the last drain. Used to correlate the geometric
        # triangulation baseline with graph conditioning (covariance / condition number).
        self.parallax_log = []

        # --- Depth filter parameters (Vogiatzis Gaussian-Uniform mixture) ---
        # Only landmarks whose depth converges below this relative uncertainty get promoted.
        self.depth_filter_enabled = True
        self.depth_tau = 0.02        # Inverse-depth measurement noise std (m^-1)
        self.depth_convergence_rel = 0.1  # Max relative sigma for convergence
        self.depth_max_sigma_m = 0.25     # Max absolute depth sigma (meters)
        self.depth_min_inlier = 0.5   # Min inlier probability for convergence
        self.depth_outlier_thresh = 0.3  # Below this → landmark is outlier, remove
        self.buffer_max_age_kf = 0      # Prune buffer entries older than this many keyframes
        self._depth_stats = {"attempted": 0, "promoted": 0, "outlier": 0, "pending": 0}
        self.promoted_depth_log = []  # [(landmark_id, pt3_world, depth_mu, depth_sigma2, df_a, df_n)]
        self._static_promotion_done = False  # One-time cold-start bypass

    def _make_projection_factor(
        self, measurement, state_idx: int, landmark_id: int, noise=None
    ):
        """Create a projection factor with throwCheirality=False.

        Setting throwCheirality=False makes the factor return zero error (and
        zero Jacobians) when the landmark projects behind the camera, preventing
        IndeterminateLinearSystemException during ISAM2 relinearization.

        Args:
            noise: Optional noise model override (e.g. tighter loop-closure noise).
                   Defaults to the standard 1.5px robust pixel noise.
        """
        if noise is None:
            noise = self.pixel_noise
        if self.body_P_sensor is not None:
            # Signature: (measured, noise, poseKey, pointKey, K, throwCheirality, verboseCheirality, body_P_sensor)
            return gtsam.GenericProjectionFactorCal3_S2(
                measurement,
                noise,
                X(state_idx),
                L(landmark_id),
                self.cal,
                False,
                False,
                self.body_P_sensor,
            )
        else:
            # Signature: (measured, noise, poseKey, pointKey, K, throwCheirality, verboseCheirality)
            return gtsam.GenericProjectionFactorCal3_S2(
                measurement,
                noise,
                X(state_idx),
                L(landmark_id),
                self.cal,
                False,
                False,
            )

    def _is_cell_available(self, state_idx: int, uv: np.ndarray) -> bool:
        """Check if this pixel's grid cell is unoccupied for this frame.

        Ensures spatial distribution of observations — at most one
        projection factor per grid cell per pose, maximizing the
        geometric information contributed by visual factors.
        """
        col = int(uv[0]) // self.obs_cell_size
        row = int(uv[1]) // self.obs_cell_size
        cell = (row, col)

        if state_idx not in self._frame_occupied_cells:
            self._frame_occupied_cells[state_idx] = set()

        if cell in self._frame_occupied_cells[state_idx]:
            return False

        self._frame_occupied_cells[state_idx].add(cell)
        return True

    # ==================== Initialization ====================

    def add_initial_state(self, pose_wb, vel_w: np.ndarray, bias, set_priors=True):
        if gtsam is None:
            return
        if isinstance(pose_wb, gtsam.Pose3):
            pose3 = pose_wb
        elif isinstance(pose_wb, np.ndarray) and pose_wb.shape == (4, 4):
            pose3 = gtsam.Pose3(
                gtsam.Rot3(pose_wb[:3, :3]), gtsam.Point3(*pose_wb[:3, 3])
            )
        else:
            pose3 = gtsam.Pose3()

        self.initial.insert(X(self.state_index), pose3)
        self.initial.insert(V(self.state_index), gtsam.Point3(*vel_w))
        self.initial.insert(B(self.state_index), bias)
        if set_priors and self.state_index == 0:
            self.graph.add(gtsam.PriorFactorPose3(X(0), pose3, self.prior_pose_noise))
            self.graph.add(
                gtsam.PriorFactorVector(
                    V(0), gtsam.Point3(*vel_w), self.prior_vel_noise
                )
            )
            self.graph.add(
                gtsam.PriorFactorConstantBias(B(0), bias, self.prior_bias_noise)
            )

    # ==================== IMU Factors ====================

    def add_zero_velocity_prior(self, state_idx: int, sigma: float = 0.01):
        """Add a tight zero-velocity prior for stationary periods.

        Args:
            state_idx: State index to constrain.
            sigma: Velocity sigma in m/s (default 0.01 = 1cm/s).
        """
        if gtsam is None:
            return
        noise = gtsam.noiseModel.Isotropic.Sigma(3, sigma)
        self.graph.add(
            gtsam.PriorFactorVector(V(state_idx), np.zeros(3), noise)
        )

    def add_zero_motion_constraint(self, prev_idx: int, curr_idx: int,
                                   rot_sigma: float = 0.001, trans_sigma: float = 0.005):
        """Add a tight identity BetweenFactor<Pose3> for stationary periods.

        Constrains X(curr_idx) to be (nearly) identical to X(prev_idx).

        Args:
            rot_sigma: Rotation noise sigma in rad (default 0.001 ≈ 0.06°).
            trans_sigma: Translation noise sigma in m (default 0.005 = 5mm).
        """
        if gtsam is None:
            return
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([rot_sigma] * 3 + [trans_sigma] * 3)
        )
        self.graph.add(
            gtsam.BetweenFactorPose3(X(prev_idx), X(curr_idx), gtsam.Pose3(), noise)
        )

    def add_state_variable(self, idx, nav_state, bias):
        """Add a new state (pose, velocity, bias) to the initial values."""
        if gtsam is None:
            return
        self.initial.insert(X(idx), nav_state.pose())
        self.initial.insert(V(idx), nav_state.velocity())
        self.initial.insert(B(idx), bias)
        self.state_index = max(self.state_index, idx)

    def add_imu_factor(self, preint, prev_idx: int, curr_idx: int):
        if gtsam is None or preint is None:
            return
        fac = gtsam.ImuFactor(
            X(prev_idx), V(prev_idx), X(curr_idx), V(curr_idx), B(prev_idx), preint
        )
        self.graph.add(fac)
        # Bias random walk between consecutive states
        # Discrete noise = continuous RW density * sqrt(dt)
        # Uses preintegration interval for proper scaling
        dt = preint.deltaTij()
        bias_sigmas = self.imu_calib.bias_between_sigmas(dt)
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(bias_sigmas)
        self.graph.add(
            gtsam.BetweenFactorConstantBias(
                B(prev_idx), B(curr_idx), gtsam.imuBias.ConstantBias(), bias_noise
            )
        )

    # ==================== Visual Factors (Explicit Landmarks) ====================

    def add_landmark_observation(
        self,
        landmark_id: int,
        state_idx: int,
        uv: np.ndarray,
        K: np.ndarray,
        landmark_3d: np.ndarray = None,
    ):
        """
        Add a projection factor between pose X(state_idx) and landmark L(landmark_id).

        Landmarks are buffered until they have observations from 2+ different poses.
        Once promoted, all buffered observations are flushed as projection factors
        without needing an artificial prior.

        Args:
            landmark_id: Unique landmark identifier.
            state_idx: Pose state index this observation comes from.
            uv: 2D pixel measurement [u, v] in rectified image.
            K: 3x3 rectified camera intrinsics.
            landmark_3d: Initial 3D position in CAMERA frame (required for first observation).
        """
        if gtsam is None:
            return

        # Set calibration once
        if self.cal is None:
            self.cal = gtsam.Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])

        # --- Case 1: Landmark already in ISAM2 — just add a new projection factor ---
        if landmark_id in self.landmark_initialized:
            # Skip frozen (high-uncertainty) landmarks
            if landmark_id in self.landmark_frozen:
                return
            # Depth check: skip if landmark is behind the camera from this pose
            if not self._is_landmark_in_front(landmark_id, state_idx):
                return
            # Parallax check: skip near-zero-parallax re-observations. If the camera has
            # barely moved relative to the last view that contributed a factor for this
            # landmark, the new ray is nearly parallel to the old one and adds little
            # geometric information — yet at a tight pixel sigma it still pulls the pose.
            # Gating by incremental parallax keeps only views that widen the baseline.
            parallax = self._obs_parallax_deg(landmark_id, state_idx)
            if parallax is not None and parallax < self.min_obs_parallax_deg:
                return
            # Cell check: skip if this pixel's grid cell is already occupied for this pose
            if not self._is_cell_available(state_idx, uv):
                return
            measurement = gtsam.Point2(float(uv[0]), float(uv[1]))
            self.graph.add(
                self._make_projection_factor(measurement, state_idx, landmark_id)
            )
            self.landmark_obs_count[landmark_id] = (
                self.landmark_obs_count.get(landmark_id, 0) + 1
            )
            # Advance the reference view for the next incremental-parallax check
            cam_pos = self._camera_position(state_idx)
            if cam_pos is not None:
                self.landmark_last_factor_cam[landmark_id] = cam_pos
            return

        # --- Case 2: Landmark in buffer — add observation and maybe promote ---
        if landmark_id in self.landmark_buffer:
            # Cell check: ensure spatial distribution even for buffered landmarks
            if not self._is_cell_available(state_idx, uv):
                return
            buf = self.landmark_buffer[landmark_id]
            buf["observations"].append((state_idx, np.array(uv, dtype=float)))
            # Check if we have observations from 3+ distinct poses
            distinct_poses = set(s for s, _ in buf["observations"])
            if len(distinct_poses) >= 3:
                # Only attempt promotion every 2 new distinct poses after the initial 3,
                # to avoid repeated expensive parallax checks on every observation
                if len(distinct_poses) == 3 or len(distinct_poses) % 2 == 0:
                    self._promote_landmark(landmark_id)
            return

        # --- Case 3: First time seeing this landmark — add to buffer ---
        if landmark_3d is None:
            return  # Cannot initialize without 3D position
        # Cell check: claim this cell for the first observation
        if not self._is_cell_available(state_idx, uv):
            return
        self.landmark_buffer[landmark_id] = {
            "point_cam": np.array(landmark_3d, dtype=float),
            "first_state": state_idx,
            "observations": [(state_idx, np.array(uv, dtype=float))],
            # Depth filter state (Vogiatzis Gaussian-Uniform mixture in inverse depth)
            "df_mu": 1.0 / max(landmark_3d[2], 0.1),      # inverse depth mean
            "df_sigma2": (0.2 / max(landmark_3d[2], 0.1)) ** 2,  # initial ~20% rel uncertainty
            "df_a": 0.5,   # inlier probability (50/50 prior)
            "df_n": 1,     # measurement count
        }

    def force_promote_top_n(self, n: int = 100) -> int:
        """One-time cold-start promotion: bypass depth filter and promote the top N
        buffered landmarks ranked by quality (inlier probability * observation count).

        Only fires once (sets _static_promotion_done). Used when the system exits
        the static period so the graph has visual constraints before IMU drift accumulates.

        Returns number of landmarks actually promoted.
        """
        if self._static_promotion_done:
            return 0
        self._static_promotion_done = True

        if not self.landmark_buffer:
            return 0

        # Rank candidates: must have >= 2 distinct poses and valid depth
        candidates = []
        for lm_id, buf in self.landmark_buffer.items():
            distinct_poses = len(set(s for s, _ in buf["observations"]))
            if distinct_poses < 2:
                continue
            depth = buf["point_cam"][2]
            if depth < 0.2 or depth > 15.0:
                continue
            # Score: prioritize high inlier probability and many observations
            score = buf["df_a"] * buf["df_n"]
            candidates.append((score, lm_id))

        # Sort descending by score, take top N
        candidates.sort(reverse=True)
        promoted = 0
        for _, lm_id in candidates[:n]:
            buf = self.landmark_buffer[lm_id]
            pt3_world = self._landmark_to_world(buf["point_cam"], buf["first_state"])

            # Validate: landmark must be in front of all observing cameras
            pt_gtsam = gtsam.Point3(*pt3_world)
            valid = True
            for s_idx, _uv in buf["observations"]:
                estimate = self.get_current_estimate()
                if estimate is not None and estimate.exists(X(s_idx)):
                    world_T_body = estimate.atPose3(X(s_idx))
                elif self.initial.exists(X(s_idx)):
                    world_T_body = self.initial.atPose3(X(s_idx))
                else:
                    continue
                if self.body_P_sensor is not None:
                    world_T_cam = world_T_body.compose(self.body_P_sensor)
                else:
                    world_T_cam = world_T_body
                pt_cam = world_T_cam.transformTo(pt_gtsam)
                if pt_cam[2] < 0.2:
                    valid = False
                    break
            if not valid:
                continue

            # Promote (bypass depth filter and parallax checks)
            self.landmark_buffer.pop(lm_id)
            self._depth_stats["promoted"] += 1
            self.promoted_depth_log.append((
                lm_id, pt3_world.copy(),
                buf["df_mu"], buf["df_sigma2"], buf["df_a"], buf["df_n"],
            ))

            self.initial.insert(L(lm_id), gtsam.Point3(*pt3_world))
            self.landmark_initialized.add(lm_id)
            self.landmark_obs_count[lm_id] = 0

            reg_noise = gtsam.noiseModel.Isotropic.Sigma(3, self.landmark_reg_sigma)
            self.graph.add(
                gtsam.PriorFactorPoint3(L(lm_id), gtsam.Point3(*pt3_world), reg_noise)
            )

            for s_idx, uv in buf["observations"]:
                measurement = gtsam.Point2(float(uv[0]), float(uv[1]))
                self.graph.add(
                    self._make_projection_factor(measurement, s_idx, lm_id)
                )
                self.landmark_obs_count[lm_id] += 1

            last_s_idx = buf["observations"][-1][0]
            cam_pos = self._camera_position(last_s_idx)
            if cam_pos is not None:
                self.landmark_last_factor_cam[lm_id] = cam_pos
            promoted += 1

        if promoted > 0:
            print(f"  [COLD-START] Force-promoted {promoted}/{len(candidates)} landmarks from buffer")
        return promoted

    def _promote_landmark(self, landmark_id: int):
        """Move a landmark from the buffer into the factor graph (ISAM2).

        Called when a buffered landmark has 3+ observations from different poses.
        Adds the landmark variable + all buffered projection factors at once.
        Validates that the landmark is geometrically visible (positive depth) from
        all observing poses before committing — prevents indeterminate systems from
        zero-Jacobian projection factors.

        Returns without popping the buffer if parallax/depth checks fail,
        allowing the landmark to accumulate more observations and retry later.
        """
        buf = self.landmark_buffer[landmark_id]
        # Validate: reject landmarks with degenerate camera-frame depth
        depth = buf["point_cam"][2]
        if depth < 0.2 or depth > 15.0:
            # Bad triangulation — remove permanently
            self.landmark_buffer.pop(landmark_id)
            return

        # Transform landmark from camera frame to world frame using the first observing pose
        pt3_world = self._landmark_to_world(buf["point_cam"], buf["first_state"])

        # Validate: minimum parallax angle between observing poses
        max_parallax = self._max_parallax_deg(pt3_world, buf["observations"])

        # Validate: landmark must be in front of ALL observing cameras + collect depths
        pt_gtsam = gtsam.Point3(*pt3_world)
        observed_depths = []
        for s_idx, _uv in buf["observations"]:
            estimate = self.get_current_estimate()
            if estimate is not None and estimate.exists(X(s_idx)):
                world_T_body = estimate.atPose3(X(s_idx))
            elif self.initial.exists(X(s_idx)):
                world_T_body = self.initial.atPose3(X(s_idx))
            else:
                continue
            if self.body_P_sensor is not None:
                world_T_cam = world_T_body.compose(self.body_P_sensor)
            else:
                world_T_cam = world_T_body
            pt_cam = world_T_cam.transformTo(pt_gtsam)
            if pt_cam[2] < 0.2:  # Behind camera or too close
                # Keep in buffer — pose estimates may improve later
                return
            observed_depths.append(float(pt_cam[2]))

        # --- Depth filter: update regardless of parallax, check convergence ---
        if self.depth_filter_enabled and len(observed_depths) >= 2:
            self._depth_stats["attempted"] += 1
            self._update_depth_filter(buf, observed_depths)
            if buf["df_a"] < self.depth_outlier_thresh and buf["df_n"] >= 3:
                # Landmark is likely an outlier — remove permanently
                self._depth_stats["outlier"] += 1
                self.landmark_buffer.pop(landmark_id)
                return
            rel_sigma = np.sqrt(buf["df_sigma2"]) / max(abs(buf["df_mu"]), 1e-10)
            # Absolute depth sigma: σ_d = σ_ρ * d²
            depth_est = 1.0 / max(abs(buf["df_mu"]), 1e-10)
            abs_sigma_m = np.sqrt(buf["df_sigma2"]) * depth_est * depth_est
            if (rel_sigma > self.depth_convergence_rel
                    or abs_sigma_m > self.depth_max_sigma_m
                    or buf["df_a"] < self.depth_min_inlier):
                # Not converged yet — keep in buffer for more observations
                self._depth_stats["pending"] += 1
                return

        # Parallax gate (checked after depth filter so filter can accumulate)
        if max_parallax < self.parallax_threshold:
            return

        # All checks passed — pop from buffer and commit to graph
        self.landmark_buffer.pop(landmark_id)
        if self.depth_filter_enabled:
            self._depth_stats["promoted"] += 1
            # Log depth filter state at promotion for visualization
            self.promoted_depth_log.append((
                landmark_id, pt3_world.copy(),
                buf["df_mu"], buf["df_sigma2"], buf["df_a"], buf["df_n"],
            ))

        self.initial.insert(L(landmark_id), gtsam.Point3(*pt3_world))
        self.landmark_initialized.add(landmark_id)
        self.landmark_obs_count[landmark_id] = 0

        # Regularization prior to prevent indeterminate linear system
        # Uses current landmark_reg_sigma
        reg_noise = gtsam.noiseModel.Isotropic.Sigma(3, self.landmark_reg_sigma)
        self.graph.add(
            gtsam.PriorFactorPoint3(
                L(landmark_id),
                gtsam.Point3(*pt3_world),
                reg_noise,
            )
        )

        # Add all buffered projection factors
        for s_idx, uv in buf["observations"]:
            measurement = gtsam.Point2(float(uv[0]), float(uv[1]))
            self.graph.add(
                self._make_projection_factor(measurement, s_idx, landmark_id)
            )
            self.landmark_obs_count[landmark_id] += 1

        # Seed the incremental-parallax reference with the most recent observing view
        last_s_idx = buf["observations"][-1][0]
        cam_pos = self._camera_position(last_s_idx)
        if cam_pos is not None:
            self.landmark_last_factor_cam[landmark_id] = cam_pos

    def add_loop_closure_observation(
        self, landmark_id: int, state_idx: int, uv: np.ndarray
    ):
        """Add a reprojection factor for a loop closure re-observation.

        This is for when you re-detect an existing landmark from a distant frame.
        The landmark must already be initialized in the graph.
        """
        if gtsam is None or self.cal is None:
            return
        if landmark_id not in self.landmark_initialized:
            return  # Can't add factor to non-existent landmark

        measurement = gtsam.Point2(float(uv[0]), float(uv[1]))
        self.graph.add(
            self._make_projection_factor(
                measurement, state_idx, landmark_id, noise=self.loop_pixel_noise
            )
        )
        self.landmark_obs_count[landmark_id] = (
            self.landmark_obs_count.get(landmark_id, 0) + 1
        )

    def add_loop_closure_pose_constraint(
        self,
        current_idx: int,
        matched_idx: int,
        relative_pose,
        noise_sigmas: np.ndarray = None,
    ):
        """Add a BetweenFactor<Pose3> for loop closure between two frames.

        This provides a strong 6-DOF constraint that can correct accumulated drift,
        unlike reprojection-only loop closure which is too weak for trajectory correction.

        Args:
            current_idx: Current state index.
            matched_idx: Matched (earlier) state index.
            relative_pose: Measured relative pose T_matched_current (transform from
                           current frame to matched frame in body coordinates).
            noise_sigmas: 6-vector [rx, ry, rz, tx, ty, tz] noise sigmas.
                          Default: [0.05, 0.05, 0.05, 0.15, 0.15, 0.15] (rad, m).
        """
        if gtsam is None:
            return
        if noise_sigmas is None:
            noise_sigmas = np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2])
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
        self.graph.add(
            gtsam.BetweenFactorPose3(
                X(matched_idx), X(current_idx), relative_pose, noise_model
            )
        )

    def _update_depth_filter(self, buf: dict, observed_depths: List[float]):
        """Update Vogiatzis Gaussian-Uniform mixture depth filter.

        Models inverse depth as: p(d) = a * N(mu, sigma2) + (1-a) * U(1/d_max, 1/d_min)
        Each depth measurement updates the mixture via Bayesian inference.
        Inlier measurements tighten the Gaussian; outliers are absorbed by the uniform.

        Args:
            buf: Landmark buffer dict containing filter state (df_mu, df_sigma2, df_a, df_n).
            observed_depths: Depth measurements from different cameras (meters).
        """
        d_min, d_max = 0.1, 20.0
        uniform_range = 1.0 / d_min - 1.0 / d_max  # range in inverse depth
        p_uniform = 1.0 / uniform_range
        tau2 = self.depth_tau ** 2

        for depth in observed_depths:
            if depth <= d_min or depth > d_max:
                continue

            z = 1.0 / depth  # measurement in inverse depth
            mu = buf["df_mu"]
            sigma2 = buf["df_sigma2"]
            a = buf["df_a"]

            # Gaussian likelihood of this measurement
            S = sigma2 + tau2
            residual = z - mu
            p_gauss = (1.0 / np.sqrt(2.0 * np.pi * S)) * np.exp(-0.5 * residual ** 2 / S)

            # Posterior inlier probability
            denom = a * p_gauss + (1.0 - a) * p_uniform
            if denom < 1e-30:
                continue
            a_new = (a * p_gauss) / denom

            # Kalman update of Gaussian component
            K = sigma2 / S
            mu_new = mu + K * residual
            sigma2_new = (1.0 - K) * sigma2

            buf["df_mu"] = mu_new
            buf["df_sigma2"] = sigma2_new
            buf["df_a"] = a_new
            buf["df_n"] += 1

    def _max_parallax_deg(self, pt3_world: np.ndarray, observations: list) -> float:
        """ Important note because this function is critical:
        Compute max parallax angle (degrees) between any two observing cameras.
        The parallax angle is the angle between the rays from two camera centers to the landmark.
        It is directly proportional to the landmark's triangulation / depth uncertainty (sigma^2 ~ depth/sin(parallax)^2).
        Also: (arctan (baseline / depth)) ≈ baseline/depth for small angles (1° parallax at 5m depth means ~8.7cm baseline).
        This geometrical validation highly correlate with the quality of the results! Landmarks quality improve drastically,
        And so does the iSAM2 convergence and stability, especially in the presence of outliers and noisy initial estimates.
        Rejecting low-parallax landmarks prevents the optimizer from being corrupted by degenerate factors with near-zero Jacobians,
        which can cause indeterminate linear systems and convergence failures.
        """
        estimate = self.get_current_estimate()
        cam_positions = []
        for s_idx, _ in observations:
            if estimate is not None and estimate.exists(X(s_idx)):
                pose = estimate.atPose3(X(s_idx))
            elif self.initial.exists(X(s_idx)):
                pose = self.initial.atPose3(X(s_idx))
            else:
                continue
            if self.body_P_sensor is not None:
                cam_pos = pose.compose(self.body_P_sensor).translation()
            else:
                cam_pos = pose.translation()
            cam_positions.append(np.array(cam_pos))

        if len(cam_positions) < 2:
            return 0.0

        max_angle = 0.0
        for i in range(len(cam_positions)):
            ray_i = pt3_world - cam_positions[i]
            norm_i = np.linalg.norm(ray_i)
            if norm_i < 1e-10:
                continue
            for j in range(i + 1, len(cam_positions)):
                ray_j = pt3_world - cam_positions[j]
                norm_j = np.linalg.norm(ray_j)
                if norm_j < 1e-10:
                    continue
                cos_angle = np.dot(ray_i, ray_j) / (norm_i * norm_j)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                if angle > max_angle:
                    max_angle = angle
        return max_angle

    def _camera_position(self, state_idx: int) -> Optional[np.ndarray]:
        """World position of the camera center for pose X(state_idx), or None."""
        estimate = self.get_current_estimate()
        if estimate is not None and estimate.exists(X(state_idx)):
            pose = estimate.atPose3(X(state_idx))
        elif self.initial.exists(X(state_idx)):
            pose = self.initial.atPose3(X(state_idx))
        else:
            return None
        if self.body_P_sensor is not None:
            return np.array(pose.compose(self.body_P_sensor).translation())
        return np.array(pose.translation())

    def _obs_parallax_deg(self, landmark_id: int, state_idx: int) -> Optional[float]:
        """Parallax angle (deg) at the landmark between the current camera and the
        last camera that contributed a factor for this landmark.

        Returns None when there is no reference view yet or positions are unavailable,
        signalling the caller not to gate on parallax.
        """
        ref_pos = self.landmark_last_factor_cam.get(landmark_id)
        if ref_pos is None:
            return None
        estimate = self.get_current_estimate()
        if estimate is None or not estimate.exists(L(landmark_id)):
            return None
        pt_world = np.array(estimate.atPoint3(L(landmark_id)))
        cur_pos = self._camera_position(state_idx)
        if cur_pos is None:
            return None
        ray_ref = pt_world - ref_pos
        ray_cur = pt_world - cur_pos
        n_ref = np.linalg.norm(ray_ref)
        n_cur = np.linalg.norm(ray_cur)
        if n_ref < 1e-9 or n_cur < 1e-9:
            return 0.0
        cos_angle = np.dot(ray_ref, ray_cur) / (n_ref * n_cur)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    def _is_landmark_in_front(self, landmark_id: int, state_idx: int) -> bool:
        """Check if a landmark has positive depth from the given pose's camera.

        Returns False if the landmark would project behind the camera, which
        would cause singular Jacobians in the projection factor.
        """
        estimate = self.get_current_estimate()
        # Get landmark position
        if estimate is not None and estimate.exists(L(landmark_id)):
            pt_world = estimate.atPoint3(L(landmark_id))
        else:
            return True  # Can't check, assume OK

        # Get camera pose
        if estimate is not None and estimate.exists(X(state_idx)):
            world_T_body = estimate.atPose3(X(state_idx))
        elif self.initial.exists(X(state_idx)):
            world_T_body = self.initial.atPose3(X(state_idx))
        else:
            return True

        if self.body_P_sensor is not None:
            world_T_cam = world_T_body.compose(self.body_P_sensor)
        else:
            world_T_cam = world_T_body

        # Transform to camera frame and check depth (Z > 0)
        pt_cam = world_T_cam.transformTo(gtsam.Point3(*pt_world))
        return pt_cam[2] > 0.1  # Minimum 10cm in front

    def _landmark_to_world(self, point_cam: np.ndarray, state_idx: int) -> np.ndarray:
        """Transform a 3D point from camera frame to world frame.

        Uses the current estimate of pose X(state_idx) and body_P_sensor.
        """
        # Get current body pose in world
        estimate = self.get_current_estimate()
        if estimate is not None and estimate.exists(X(state_idx)):
            world_T_body = estimate.atPose3(X(state_idx))
        else:
            # Fallback: check initial values
            if self.initial.exists(X(state_idx)):
                world_T_body = self.initial.atPose3(X(state_idx))
            else:
                world_T_body = gtsam.Pose3()

        # world_T_cam = world_T_body * body_P_sensor
        if self.body_P_sensor is not None:
            world_T_cam = world_T_body.compose(self.body_P_sensor)
        else:
            world_T_cam = world_T_body

        # Transform point from camera frame to world
        pt_world = world_T_cam.transformFrom(gtsam.Point3(*point_cam))
        return np.array([pt_world[0], pt_world[1], pt_world[2]])

    # ==================== Optimization ====================

    def optimize(self):
        if gtsam is None:
            return None
        self._cached_estimate = None  # Invalidate cache
        if self.isam is not None:
            try:
                update_result = self.isam.update(self.graph, self.initial)
                # error_diff = (
                #     update_result.getErrorAfter() - update_result.getErrorBefore()
                # )
                # print(f"ISAM2 error change after update: {error_diff:.3f}, ")
                self.graph.resize(0)
                self.initial.clear()
                # One extra iteration helps convergence

                if (
                    update_result.getVariablesRelinearized() > 50
                    # or abs(error_diff) > 1.0
                ):
                    self.isam.update()
            except RuntimeError as e:
                if "Indeterminant" in str(e) or "indeterminant" in str(e):
                    # Graceful recovery: discard this batch and continue
                    print(f"[WARN] ISAM2 indeterminate system, skipping batch: {e}")
                    self.graph.resize(0)
                    self.initial.clear()
                else:
                    raise
            self._cached_estimate = self.isam.calculateEstimate()
            return self._cached_estimate
        else:
            # Batch LM mode: merge pending factors/values into accumulators
            for i in range(self.graph.size()):
                self._batch_graph.add(self.graph.at(i))
            self.graph.resize(0)

            # Merge new initial values (skip keys already present — use latest estimate)
            keys = self.initial.keys()
            for key in keys:
                if not self._batch_values.exists(key):
                    self._batch_values.insert(key, self.initial.atGeneric(key))
            self.initial.clear()

            # Use previous estimate as linearization point for known variables
            if self._cached_estimate is not None:
                for key in self._cached_estimate.keys():
                    if self._batch_values.exists(key):
                        self._batch_values.update(key, self._cached_estimate.atGeneric(key))

            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(20)
            lm = gtsam.LevenbergMarquardtOptimizer(
                self._batch_graph, self._batch_values, params
            )
            result = lm.optimize()
            self._cached_estimate = result
            # Update batch values to latest estimate for next iteration
            self._batch_values = result
            return result

    # ==================== Helpers ====================

    def get_current_estimate(self):
        if gtsam is None:
            return None
        if self._cached_estimate is not None:
            return self._cached_estimate
        if self.isam is not None:
            self._cached_estimate = self.isam.calculateEstimate()
            return self._cached_estimate
        return None

    def get_optimized_landmarks(self) -> Dict[int, np.ndarray]:
        """Get all optimized landmark 3D positions from the current estimate."""
        result = {}
        estimate = self.get_current_estimate()
        if estimate is None:
            return result
        for lm_id in self.landmark_initialized:
            try:
                if estimate.exists(L(lm_id)):
                    pt = estimate.atPoint3(L(lm_id))
                    result[lm_id] = np.array([pt[0], pt[1], pt[2]])
            except Exception:
                continue
        return result

    def summarize(self):
        if gtsam is None:
            return "GTSAM not available"
        n_landmarks = len(self.landmark_initialized)
        n_buffered = len(self.landmark_buffer)
        n_total_obs = sum(self.landmark_obs_count.values())
        return (
            f"States: {self.state_index+1}, Landmarks: {n_landmarks} "
            f"(+{n_buffered} buffered), "
            f"Observations: {n_total_obs}, Factors(pending): {self.graph.size()}"
        )

    def diagnostics(self):
        """Return optimization quality metrics."""
        if gtsam is None:
            return {}
        result = self.get_current_estimate()
        if result is None:
            return {}
        if self.isam is not None:
            factors = self.isam.getFactorsUnsafe()
        elif self._batch_graph is not None:
            factors = self._batch_graph
        else:
            return {}
        total_error = factors.error(result)
        n_vars = result.size()
        n_factors = factors.size()
        avg_error = total_error / max(n_factors, 1)
        metrics = {
            "total_error": total_error,
            "n_variables": n_vars,
            "n_factors": n_factors,
            "avg_error_per_factor": avg_error,
            "n_landmarks": len(self.landmark_initialized),
        }
        return metrics

    def drain_parallax_stats(self):
        """Return (and clear) parallax-angle stats accumulated since the last call."""
        if not self.parallax_log:
            return {"parallax_max_deg": 0.0, "parallax_median_deg": 0.0, "parallax_count": 0}
        arr = np.asarray(self.parallax_log, dtype=float)
        stats = {
            "parallax_max_deg": float(arr.max()),
            "parallax_median_deg": float(np.median(arr)),
            "parallax_count": int(arr.size),
        }
        self.parallax_log = []
        return stats

    def drain_depth_filter_stats(self) -> Dict:
        """Return (and clear) depth filter stats since last call."""
        stats = dict(self._depth_stats)
        self._depth_stats = {"attempted": 0, "promoted": 0, "outlier": 0, "pending": 0}
        return stats

    def prune_stale_buffer(self, current_kf_idx: int) -> int:
        """Remove buffer entries whose first observation is too old.

        Landmarks that haven't converged within buffer_max_age_kf keyframes
        will never converge — the camera has moved on and won't re-see them.
        Pruning them frees memory and reduces per-frame overhead.

        Returns number of pruned entries.
        """
        if self.buffer_max_age_kf <= 0:
            return 0
        cutoff = current_kf_idx - self.buffer_max_age_kf
        stale = [lm_id for lm_id, buf in self.landmark_buffer.items()
                 if buf["first_state"] < cutoff]
        for lm_id in stale:
            del self.landmark_buffer[lm_id]
        return len(stale)

    # ==================== Landmark Uncertainty Filtering ====================

    def filter_uncertain_landmarks(self, max_trace: float = 2.0) -> int:
        """Freeze landmarks whose position uncertainty exceeds a threshold.

        Computes the marginal covariance for each active (non-frozen) landmark
        and freezes those where trace(cov) > max_trace. Frozen landmarks will
        no longer receive new projection factors.

        Args:
            max_trace: Maximum allowed trace of 3x3 position covariance (m^2).
                       Default 2.0 means avg std > ~0.8m per axis.

        Returns:
            Number of newly frozen landmarks.
        """
        if gtsam is None or self.isam is None:
            return 0

        newly_frozen = 0
        for lm_id in list(self.landmark_initialized):
            if lm_id in self.landmark_frozen:
                continue
            try:
                cov = self.isam.marginalCovariance(L(lm_id))
                if np.trace(cov) > max_trace:
                    self.landmark_frozen.add(lm_id)
                    newly_frozen += 1
            except Exception:
                # Covariance computation can fail for poorly connected variables
                self.landmark_frozen.add(lm_id)
                newly_frozen += 1

        if newly_frozen > 0:
            print(
                f"  Froze {newly_frozen} high-uncertainty landmarks "
                f"(total frozen: {len(self.landmark_frozen)}/{len(self.landmark_initialized)})"
            )
        return newly_frozen

    # ==================== Covariance & Degeneracy ====================

    def get_pose_covariance(self, state_idx: int) -> Optional[np.ndarray]:
        """Get the 6x6 marginal covariance for pose X(state_idx).

        Returns None if computation fails. Ordered: [rot_x, rot_y, rot_z, tx, ty, tz].
        """
        if gtsam is None or self.isam is None:
            return None
        try:
            return self.isam.marginalCovariance(X(state_idx))
        except Exception:
            return None

    def get_position_covariance(self, state_idx: int) -> Optional[np.ndarray]:
        """Get the 3x3 position-only marginal covariance (translation block [3:6, 3:6])."""
        cov6 = self.get_pose_covariance(state_idx)
        if cov6 is None:
            return None
        return cov6[3:6, 3:6]

    def detect_degeneracy(self, state_idx: int, pos_cov: Optional[np.ndarray] = None) -> Dict:
        """Analyze pose covariance to detect degenerate motion conditions.

        Args:
            state_idx: Pose state index to analyze.
            pos_cov: Pre-computed 3x3 position covariance. If None, computed internally.

        Returns:
            'position_eigenvalues': sorted eigenvalues of position cov (ascending)
            'position_eigenvectors': corresponding eigenvectors (columns)
            'condition_number': max/min eigenvalue ratio (high = degenerate)
            'degenerate': bool
            'degenerate_direction': unit vector of worst-constrained direction
            'motion_type': string description
        """
        result = {
            "position_eigenvalues": None,
            "position_eigenvectors": None,
            "condition_number": 1.0,
            "degenerate": False,
            "degenerate_direction": None,
            "motion_type": "normal",
        }

        if pos_cov is None:
            pos_cov = self.get_position_covariance(state_idx)
        if pos_cov is None:
            return result

        eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)
        result["position_eigenvalues"] = eigenvalues
        result["position_eigenvectors"] = eigenvectors

        min_eig = max(eigenvalues[0], 1e-12)
        max_eig = eigenvalues[-1]
        condition_number = max_eig / min_eig
        result["condition_number"] = condition_number
        result["degenerate_direction"] = eigenvectors[:, -1]

        CONDITION_THRESHOLD = 100.0
        LARGE_UNCERTAINTY_M2 = 1.0

        if condition_number > CONDITION_THRESHOLD:
            result["degenerate"] = True
            worst_dir = eigenvectors[:, -1]
            if abs(worst_dir[2]) > 0.8:
                result["motion_type"] = (
                    "vertical_degeneracy (scale/gravity unobservable)"
                )
            else:
                result["motion_type"] = "lateral_degeneracy (insufficient parallax)"
        elif max_eig > LARGE_UNCERTAINTY_M2:
            result["degenerate"] = True
            result["motion_type"] = "high_overall_uncertainty"

        if all(e > 0.5 for e in eigenvalues):
            result["degenerate"] = True
            result["motion_type"] = "pure_rotation (no translational motion observed)"

        return result

    # ==================== Factor Graph Visualization ====================

    def visualize_factor_graph_3d(self, output_path: str = "factor_graph_3d.html"):
        """Render the GTSAM factor graph as an interactive 3D Plotly visualization.

        Shows:
          - Pose nodes (X) as red cubes along the trajectory, connected by a red line.
          - Landmark nodes (L) as small green dots at their optimized 3D positions.
          - Projection factor edges as thin blue lines (pose → landmark).
          - IMU factor edges as thick orange lines (pose → pose).
          - Prior factors as yellow markers on the constrained node.

        This replicates the factor graph diagrams seen in VIO papers (e.g. Fig. 2
        in Forster et al. 2017, Kimera paper Fig. 3) but in 3D with real coordinates.

        Args:
            output_path: Path to save the interactive HTML file.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("[visualize_factor_graph_3d] plotly not installed, skipping.")
            return

        estimate = self.get_current_estimate()
        if estimate is None:
            print("[visualize_factor_graph_3d] No estimate available.")
            return

        # --- Collect node positions ---
        pose_positions = {}  # {state_idx: np.array(3,)}
        for idx in range(self.state_index + 1):
            if estimate.exists(X(idx)):
                p = estimate.atPose3(X(idx)).translation()
                pose_positions[idx] = np.array([p[0], p[1], p[2]])

        # Build depth uncertainty lookup from promoted_depth_log
        # depth_sigma_m = sqrt(df_sigma2) * d^2, where d = 1/df_mu
        lm_depth_sigma = {}  # {landmark_id: depth_sigma_meters}
        for entry in self.promoted_depth_log:
            lm_id, _, df_mu, df_sigma2, df_a, df_n = entry
            d = 1.0 / max(abs(df_mu), 1e-10)
            sigma_m = np.sqrt(df_sigma2) * d * d
            lm_depth_sigma[lm_id] = sigma_m

        landmark_positions = {}  # {lm_id: np.array(3,)}
        for lm_id in self.landmark_initialized:
            if estimate.exists(L(lm_id)):
                p = estimate.atPoint3(L(lm_id))
                landmark_positions[lm_id] = np.array([p[0], p[1], p[2]])

        # --- Get the factor graph (ISAM2 stores it internally) ---
        if self.isam is not None:
            factors = self.isam.getFactorsUnsafe()
        elif self._batch_graph is not None:
            factors = self._batch_graph
        else:
            print("[visualize_factor_graph_3d] No factor graph available.")
            return

        # --- Classify edges by factor type ---
        projection_edges = []  # [(pose_pos, lm_pos), ...]
        imu_edges = []         # [(pose_pos_i, pose_pos_j), ...]
        between_edges = []     # [(pose_pos_i, pose_pos_j), ...]

        for i in range(factors.size()):
            factor = factors.at(i)
            factor_keys = factor.keys()
            # keys() may return a list or a KeyVector depending on GTSAM binding
            if hasattr(factor_keys, 'size'):
                keys = [factor_keys.at(j) for j in range(factor_keys.size())]
            else:
                keys = list(factor_keys)

            if len(keys) == 2:
                k0, k1 = keys
                s0 = gtsam.Symbol(k0)
                s1 = gtsam.Symbol(k1)
                c0, c1 = chr(s0.chr()), chr(s1.chr())
                i0, i1 = s0.index(), s1.index()

                # Projection factor: X(i) ↔ L(j)
                if (c0 == 'x' and c1 == 'l'):
                    if i0 in pose_positions and i1 in landmark_positions:
                        projection_edges.append((pose_positions[i0], landmark_positions[i1]))
                elif (c0 == 'l' and c1 == 'x'):
                    if i1 in pose_positions and i0 in landmark_positions:
                        projection_edges.append((pose_positions[i1], landmark_positions[i0]))

                # IMU / Between factor: X(i) ↔ X(j) or V/B pairs
                elif c0 == 'x' and c1 == 'x':
                    if i0 in pose_positions and i1 in pose_positions:
                        between_edges.append((pose_positions[i0], pose_positions[i1]))

            elif len(keys) == 5:
                # ImuFactor has 5 keys: X(i), V(i), X(j), V(j), B(i)
                s0 = gtsam.Symbol(keys[0])
                s2 = gtsam.Symbol(keys[2])
                if chr(s0.chr()) == 'x' and chr(s2.chr()) == 'x':
                    i0, i2 = s0.index(), s2.index()
                    if i0 in pose_positions and i2 in pose_positions:
                        imu_edges.append((pose_positions[i0], pose_positions[i2]))

        # --- Build Plotly traces ---
        traces = []

        # Trajectory line (poses connected in order)
        sorted_poses = sorted(pose_positions.items())
        if sorted_poses:
            traj = np.array([p for _, p in sorted_poses])
            traces.append(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode='lines+markers',
                marker=dict(size=4, color='red', symbol='square'),
                line=dict(color='red', width=3),
                name=f'Poses ({len(sorted_poses)})',
                text=[f'X({idx})' for idx, _ in sorted_poses],
                hoverinfo='text',
            ))

        # Landmark points — colored by depth filter uncertainty (heatmap)
        if landmark_positions:
            lm_ids_sorted = sorted(landmark_positions.keys())
            lm_pts = np.array([landmark_positions[lid] for lid in lm_ids_sorted])
            # Get depth sigma for each landmark; default to 0 if not in log
            lm_sigmas = np.array([lm_depth_sigma.get(lid, 0.0) for lid in lm_ids_sorted])
            # Clip for colorscale readability (cap at 0.5m)
            lm_sigmas_clipped = np.clip(lm_sigmas, 0, 0.5)
            traces.append(go.Scatter3d(
                x=lm_pts[:, 0], y=lm_pts[:, 1], z=lm_pts[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=lm_sigmas_clipped,
                    colorscale='RdYlGn_r',  # green=low uncertainty, red=high
                    cmin=0, cmax=0.3,
                    colorbar=dict(
                        title='Depth σ (m)',
                        thickness=15,
                        len=0.5,
                        x=1.02,
                    ),
                    opacity=0.7,
                ),
                name=f'Landmarks ({len(lm_pts)})',
                text=[f'L({lid})<br>σ={lm_depth_sigma.get(lid, 0):.4f}m' for lid in lm_ids_sorted],
                hoverinfo='text',
            ))

        # Projection edges (pose → landmark)
        # Subsample if too many for performance
        max_proj_edges = 2000
        proj_sample = projection_edges
        if len(projection_edges) > max_proj_edges:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(projection_edges), max_proj_edges, replace=False)
            proj_sample = [projection_edges[i] for i in indices]

        if proj_sample:
            px, py, pz = [], [], []
            for p0, p1 in proj_sample:
                px.extend([p0[0], p1[0], None])
                py.extend([p0[1], p1[1], None])
                pz.extend([p0[2], p1[2], None])
            traces.append(go.Scatter3d(
                x=px, y=py, z=pz,
                mode='lines',
                line=dict(color='rgba(155, 155, 175,0.8)', width=1),
                name=f'Projection factors ({len(projection_edges)})',
                hoverinfo='skip',
            ))

        # IMU edges
        if imu_edges:
            ix, iy, iz = [], [], []
            for p0, p1 in imu_edges:
                ix.extend([p0[0], p1[0], None])
                iy.extend([p0[1], p1[1], None])
                iz.extend([p0[2], p1[2], None])
            traces.append(go.Scatter3d(
                x=ix, y=iy, z=iz,
                mode='lines',
                line=dict(color='orange', width=4),
                name=f'IMU factors ({len(imu_edges)})',
                hoverinfo='skip',
            ))

        # Between edges (loop closure, zero-motion)
        if between_edges:
            bx, by, bz = [], [], []
            for p0, p1 in between_edges:
                bx.extend([p0[0], p1[0], None])
                by.extend([p0[1], p1[1], None])
                bz.extend([p0[2], p1[2], None])
            traces.append(go.Scatter3d(
                x=bx, y=by, z=bz,
                mode='lines',
                line=dict(color='purple', width=2),
                name=f'Between factors ({len(between_edges)})',
                hoverinfo='skip',
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f'Factor Graph — {len(pose_positions)} poses, '
                  f'{len(landmark_positions)} landmarks, '
                  f'{factors.size()} factors',
            scene=dict(
                xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                aspectmode='data',
            ),
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.write_html(output_path)
        print(f"  Factor graph visualization saved to {output_path}")
        return fig


