import numpy as np
from typing import Dict, List, Optional

from vio_utils import angle_between_deg, gaussian_uniform_update

try:
    import gtsam
    from gtsam.symbol_shorthand import X, V, B, L
except ImportError:  # Allow file existence without immediate dependency
    gtsam = None  # type: ignore


class GraphOptimizer:
    """Tightly coupled VIO/SLAM backend: GTSAM ISAM2 over explicit landmark variables.

    Explicit landmarks (rather than smart factors) are what allow observations to be
    added to an existing landmark incrementally, its optimized 3D position to be read
    back, and a loop closure to be expressed as one more re-observation.
    """

    def __init__(
        self, use_isam: bool = True, body_P_sensor: np.ndarray = None, imu_calib=None,
        enable_landmarks: bool = True,
    ):
        # Ablation switch. False makes the backend discard every visual observation, so
        # no landmark variables and no projection factors ever enter the graph and what
        # remains is the inertial chain plus the priors and zero-motion constraints.
        # The resulting ATE is the floor vision is measured against.
        self.enable_landmarks = enable_landmarks

        if gtsam is None:
            self.isam = None
            self.graph = None
            self.initial = None
            self._depth_stats = {"attempted": 0, "promoted": 0, "outlier": 0, "pending": 0}
            self.promoted_depth_log = []
            self.promotion_events = []
            self.promotion_trace_enabled = False
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
        self.isam_params.setRelinearizeThreshold(0.03)
        self.isam_params.relinearizeSkip = 2
        self.isam_params.cacheLinearizedFactors = True
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
        # {landmark_id: (keyframe_index, covariance_trace)} at the moment of freezing —
        # tells the promotion trace which promoted landmarks later went bad.
        self.landmark_frozen_kf = {}
        # Buffer: landmarks wait here until they have 2+ observations from different poses
        # {landmark_id: {"point_cam": np.array, "first_state": int, "observations": [(state_idx, uv), ...]}}
        self.landmark_buffer = {}
        # Camera center (world) of the last view that contributed a factor for each
        # initialized landmark. Used to gate new observations by incremental parallax.
        self.landmark_last_factor_cam = {}
        self.cal = None  # gtsam.Cal3_S2, set on first observation

        # Cached estimate (invalidated after each optimize call)
        self._cached_estimate = None

        # Diagnostics throttling: full-graph error evaluation is O(n_factors), so it
        # runs every N keyframes and the value is held in between. See diagnostics().
        self.diagnostics_interval = 10
        self._diag_total_error = 0.0
        self._diag_last_idx = None

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
        self.landmark_reg_sigma = 1  # meters — sigma PERPENDICULAR to the viewing ray
        # Along-ray sigma = clip(k * sigma_z, landmark_reg_sigma, max). Widening the
        # prior in the ill-conditioned (depth) direction keeps it from asserting more
        # depth confidence than the triangulation geometry supports. Clipped from
        # below by landmark_reg_sigma so the prior is never TIGHTER than isotropic.
        self.landmark_reg_along_k = 2     # how many sigma_z to allow along the ray
        self.landmark_reg_along_max = 6  # meters — numerical sanity ceiling

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
        self.parallax_threshold = 5  # degrees — minimum parallax for landmark promotion

        # Minimum incremental parallax (deg) required to add a new observation factor to
        # an already-initialized landmark. A re-observation whose camera has barely moved
        # (relative to the last contributing view) adds little geometric information but,
        # at a tight pixel sigma, can over-constrain the pose; gating it improves accuracy.
        self.min_obs_parallax_deg = 4.5

        # Parallax instrumentation: max-parallax angles (deg) computed during landmark
        # promotion attempts since the last drain. Used to correlate the geometric
        # triangulation baseline with graph conditioning (covariance / condition number).
        self.parallax_log = []

        # --- Gate instrumentation ------------------------------------------------
        # Every exit path in add_landmark_observation / _promote_landmark increments a
        # counter, so we can see WHERE observations are lost instead of guessing.
        # Pure bookkeeping: no effect on the graph.
        self.gate_stats = self._zero_gate_stats()

        # --- Depth filter parameters (Vogiatzis Gaussian-Uniform mixture) ---
        # Only landmarks whose depth converges below this relative uncertainty get promoted.
        self.depth_filter_enabled = True
        self.depth_d_min = 0.2           # Min valid depth (meters)
        self.depth_d_max = 15.0          # Max valid depth (meters)
        self.depth_tau = 0.02        # Inverse-depth measurement noise std (m^-1)
        self.depth_convergence_rel = 0.1  # Max relative sigma for convergence
        self.depth_max_sigma_m = 0.25     # Max absolute depth sigma (meters)
        self.depth_min_inlier = 0.5   # Min inlier probability for convergence
        self.depth_outlier_thresh = 0.3  # Below this → landmark is outlier, remove
        self.buffer_max_age_kf = 70      # Drop buffered landmarks unseen for this many keyframes
        self._depth_stats = {"attempted": 0, "promoted": 0, "outlier": 0, "pending": 0}
        self.promoted_depth_log = []  # [(landmark_id, pt3_world, depth_mu, depth_sigma2, df_a, df_n)]
        self._static_promotion_done = False  # One-time cold-start bypass

        # --- Landmark lifecycle trace -------------------------------------------
        # gate_stats counts WHERE landmarks are lost; this records WHICH ones and with
        # what geometry, so the promotion decision can be second-guessed offline
        # (see visualize_landmark_promotion.py). Bookkeeping only, bounded in size.
        self.promotion_trace_enabled = True
        self.promotion_events = []
        self.promotion_events_max = 300000

    GATE_KEYS = (
        # Case 1: landmark already in the graph
        "g_frozen", "g_behind", "g_parallax", "g_cell", "g_accepted",
        # Case 2: landmark waiting in the buffer
        "b_cell", "b_appended",
        # Case 3: first sighting
        "n_no3d", "n_cell", "n_buffered",
        # _promote_landmark outcomes
        "p_attempts", "p_depth_range", "p_cheirality", "p_outlier", "p_pending",
        "p_parallax", "p_promoted", "p_flushed_factors",
    )

    def _zero_gate_stats(self):
        """Fresh all-zero gate counter dict.

        Returns:
            Dict mapping every GATE_KEYS entry to 0.
        """
        return {k: 0 for k in self.GATE_KEYS}

    def drain_gate_stats(self):
        """Return and clear the per-keyframe gate counters.

        Returns:
            Dict of counts keyed by GATE_KEYS.
        """
        st = dict(self.gate_stats)
        self.gate_stats = self._zero_gate_stats()
        return st

    def _make_projection_factor(
        self, measurement, state_idx: int, landmark_id: int, noise=None
    ):
        """Build a projection factor that degrades instead of throwing on cheirality.

        throwCheirality=False makes the factor yield zero error and zero Jacobians when
        the landmark falls behind the camera, which keeps ISAM2 relinearization from
        raising IndeterminateLinearSystemException mid-run.

        Args:
            measurement: gtsam.Point2 pixel measurement.
            state_idx: Pose state index.
            landmark_id: Landmark index.
            noise: Optional noise override; defaults to the robust pixel noise.

        Returns:
            A GenericProjectionFactorCal3_S2.
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

    def _body_pose(self, state_idx: int, default=None):
        """World pose of body frame X(state_idx), preferring the optimized estimate.

        Falls back to the pending initial values because a landmark can be evaluated
        in the same keyframe its pose was created, before ISAM2 has seen it.

        Args:
            state_idx: Pose state index.
            default: Value to return when the pose is unknown.

        Returns:
            gtsam.Pose3 for the body frame, or `default`.
        """
        estimate = self.get_current_estimate()
        if estimate is not None and estimate.exists(X(state_idx)):
            return estimate.atPose3(X(state_idx))
        if self.initial.exists(X(state_idx)):
            return self.initial.atPose3(X(state_idx))
        return default

    def _camera_pose(self, state_idx: int, default=None):
        """World pose of the camera frame for X(state_idx), via body_P_sensor.

        Args:
            state_idx: Pose state index.
            default: Value to return when the pose is unknown.

        Returns:
            gtsam.Pose3 for the camera frame, or `default`.
        """
        world_T_body = self._body_pose(state_idx)
        if world_T_body is None:
            return default
        if self.body_P_sensor is not None:
            return world_T_body.compose(self.body_P_sensor)
        return world_T_body

    def _is_cell_available(self, state_idx: int, uv: np.ndarray) -> bool:
        """Claim this pixel's grid cell for this frame, or reject if already taken.

        Capping observations at one per cell per pose spreads visual factors over the
        image; clustered measurements are near-redundant constraints on the pose.

        Args:
            state_idx: Pose state index.
            uv: Pixel measurement [u, v].

        Returns:
            True if the cell was free (and is now claimed), False otherwise.
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
        """Insert state 0 and, optionally, the priors that anchor the whole graph.

        Args:
            pose_wb: Initial body pose, as gtsam.Pose3 or a 4x4 matrix.
            vel_w: (3,) initial world-frame velocity.
            bias: Initial gtsam.imuBias.ConstantBias.
            set_priors: Whether to add pose/velocity/bias priors on X(0).

        Returns:
            None.
        """
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
        """Pin velocity to zero during a detected stationary period.

        Args:
            state_idx: State index to constrain.
            sigma: Velocity sigma in m/s.

        Returns:
            None.
        """
        if gtsam is None:
            return
        noise = gtsam.noiseModel.Isotropic.Sigma(3, sigma)
        self.graph.add(
            gtsam.PriorFactorVector(V(state_idx), np.zeros(3), noise)
        )

    def add_zero_motion_constraint(self, prev_idx: int, curr_idx: int,
                                   rot_sigma: float = 0.001, trans_sigma: float = 0.005):
        """Tie two consecutive poses together during a stationary period.

        A tight identity BetweenFactor stops IMU bias from integrating into phantom
        motion while the platform is not moving.

        Args:
            prev_idx: Earlier state index.
            curr_idx: Later state index.
            rot_sigma: Rotation noise sigma in rad.
            trans_sigma: Translation noise sigma in m.

        Returns:
            None.
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
        """Add pose/velocity/bias variables for a new keyframe.

        Args:
            idx: State index to create.
            nav_state: gtsam.NavState holding the predicted pose and velocity.
            bias: Initial bias estimate for this state.

        Returns:
            None.
        """
        if gtsam is None:
            return
        self.initial.insert(X(idx), nav_state.pose())
        self.initial.insert(V(idx), nav_state.velocity())
        self.initial.insert(B(idx), bias)
        self.state_index = max(self.state_index, idx)

    def add_imu_factor(self, preint, prev_idx: int, curr_idx: int):
        """Add the IMU factor for an interval, plus its bias random-walk factor.

        The bias BetweenFactor is scaled by the actual preintegration interval, so a
        long keyframe gap correctly allows the bias more room to drift.

        Args:
            preint: Preintegrated measurements spanning the interval.
            prev_idx: State index at the start of the interval.
            curr_idx: State index at the end.

        Returns:
            None.
        """
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
        """Route one observation: straight to the graph, into the buffer, or dropped.

        A landmark cannot enter the graph on first sight — a single view fixes bearing
        but not depth. It waits in the buffer until enough distinct poses have seen it,
        then all its observations are flushed together (see _promote_landmark).

        Args:
            landmark_id: Unique landmark identifier.
            state_idx: Pose state index this observation comes from.
            uv: 2D pixel measurement [u, v] in the rectified image.
            K: 3x3 rectified camera intrinsics.
            landmark_3d: 3D position in the CAMERA frame; required on first sighting.

        Returns:
            None.
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
                self.gate_stats["g_frozen"] += 1
                return
            # Depth check: skip if landmark is behind the camera from this pose
            if not self._is_landmark_in_front(landmark_id, state_idx):
                self.gate_stats["g_behind"] += 1
                return
            # Parallax check: skip near-zero-parallax re-observations. If the camera has
            # barely moved relative to the last view that contributed a factor for this
            # landmark, the new ray is nearly parallel to the old one and adds little
            # geometric information — yet at a tight pixel sigma it still pulls the pose.
            # Gating by incremental parallax keeps only views that widen the baseline.
            parallax = self._obs_parallax_deg(landmark_id, state_idx)
            if parallax is not None and parallax < self.min_obs_parallax_deg:
                self.gate_stats["g_parallax"] += 1
                return
            # Cell check: skip if this pixel's grid cell is already occupied for this pose
            if not self._is_cell_available(state_idx, uv):
                self.gate_stats["g_cell"] += 1
                return
            self.gate_stats["g_accepted"] += 1
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
                self.gate_stats["b_cell"] += 1
                return
            self.gate_stats["b_appended"] += 1
            buf = self.landmark_buffer[landmark_id]
            buf["observations"].append((state_idx, np.array(uv, dtype=float)))
            # The distinct-pose set is maintained incrementally rather than rebuilt from
            # the observation list on every append, which was O(k) per observation and
            # therefore O(k^2) over the life of a buffered track.
            buf["poses"].add(state_idx)
            n_distinct = len(buf["poses"])
            if n_distinct >= 3:
                # Only attempt promotion every 2 new distinct poses after the initial 3,
                # to avoid repeated expensive parallax checks on every observation
                if n_distinct == 3 or n_distinct % 2 == 0:
                    self._promote_landmark(landmark_id)
            return

        # --- Case 3: First time seeing this landmark — add to buffer ---
        if landmark_3d is None:
            self.gate_stats["n_no3d"] += 1
            return  # Cannot initialize without 3D position
        # Cell check: claim this cell for the first observation
        if not self._is_cell_available(state_idx, uv):
            self.gate_stats["n_cell"] += 1
            return
        self.gate_stats["n_buffered"] += 1
        self.landmark_buffer[landmark_id] = {
            "point_cam": np.array(landmark_3d, dtype=float),
            "first_state": state_idx,
            "observations": [(state_idx, np.array(uv, dtype=float))],
            "poses": {state_idx},  # distinct pose indices, kept in sync with observations
            # Depth filter state (Vogiatzis Gaussian-Uniform mixture in inverse depth)
            "df_mu": 1.0 / max(landmark_3d[2], 0.1),      # inverse depth mean
            "df_sigma2": (0.2 / max(landmark_3d[2], 0.1)) ** 2,  # initial ~20% rel uncertainty
            "df_a": 0.5,   # inlier probability (50/50 prior)
            "df_n": 1,     # measurement count
        }

    def force_promote_top_n(self, n: int = 100) -> int:
        """Cold-start bypass: promote the best buffered landmarks without the filters.

        Fires once, when the system leaves its static period, so the graph has visual
        constraints before IMU drift accumulates — at that point no landmark has the
        parallax history the normal gates require. Ranked by inlier probability times
        observation count; only cheirality is still enforced.

        Args:
            n: Maximum number of landmarks to promote.

        Returns:
            Number of landmarks actually promoted.
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
            if depth < self.depth_d_min or depth > self.depth_d_max:
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
                world_T_cam = self._camera_pose(s_idx)
                if world_T_cam is None:
                    continue
                if world_T_cam.transformTo(pt_gtsam)[2] < 0.2:
                    valid = False
                    break
            if not valid:
                continue

            # Promote (bypass depth filter and parallax checks)
            self._log_promotion_event(
                lm_id, buf, "promoted_coldstart", pt3_world,
                self._max_parallax_deg(pt3_world, buf["observations"]),
            )
            self.landmark_buffer.pop(lm_id)
            self._depth_stats["promoted"] += 1
            self.promoted_depth_log.append((
                lm_id, pt3_world.copy(),
                buf["df_mu"], buf["df_sigma2"], buf["df_a"], buf["df_n"],
            ))
            self._commit_landmark(lm_id, buf, pt3_world)
            promoted += 1

        if promoted > 0:
            print(f"  [COLD-START] Force-promoted {promoted}/{len(candidates)} landmarks from buffer")
        return promoted

    def _log_promotion_event(self, landmark_id: int, buf: dict, outcome: str,
                             pt3_world: np.ndarray = None, max_parallax: float = None):
        """Record the full state of one landmark at a lifecycle decision point.

        Every quantity the gates actually test is stored alongside the outcome, so an
        offline script can ask whether the rejected landmarks really were worse than the
        promoted ones instead of only seeing the survivors.

        Args:
            landmark_id: Landmark the decision concerns.
            buf: Its buffer entry (depth-filter state and observations).
            outcome: Which branch fired — see the outcome strings in _promote_landmark.
            pt3_world: World position, when it has been computed at that point.
            max_parallax: Max pairwise parallax (deg), when it has been computed.

        Returns:
            None.
        """
        if not self.promotion_trace_enabled:
            return
        if len(self.promotion_events) >= self.promotion_events_max:
            return

        mu = float(buf["df_mu"])
        sigma2 = max(float(buf["df_sigma2"]), 0.0)
        sigma_rho = float(np.sqrt(sigma2))
        depth_est = 1.0 / max(abs(mu), 1e-10)
        observations = buf["observations"]
        poses = buf.get("poses") or {s for s, _ in observations}

        # Gates that fire before pt3_world is computed still need a position, otherwise
        # the biggest loss populations cannot be placed on a map — and where a landmark
        # died is exactly what tells us which parts of the scene the frontend struggles
        # in. Fall back to the first-view stereo triangulation, which is the only
        # position those landmarks ever had.
        raw_position = pt3_world is None
        if raw_position:
            try:
                pt3_world = self._landmark_to_world(buf["point_cam"], buf["first_state"])
            except Exception:
                pt3_world = None

        self.promotion_events.append({
            "kf": int(self.state_index),
            "lm_id": int(landmark_id),
            "outcome": outcome,
            "pt3_world": None if pt3_world is None
                         else np.asarray(pt3_world, dtype=float).copy(),
            # True when the position above is raw triangulation, not the gated estimate
            "pos_is_raw": bool(raw_position),
            # Depth as stereo first triangulated it, vs. what the filter converged to
            "tri_depth": float(buf["point_cam"][2]),
            "depth_est": depth_est,
            "df_mu": mu,
            "df_sigma2": sigma2,
            "df_a": float(buf["df_a"]),
            "df_n": int(buf["df_n"]),
            "sigma_rho": sigma_rho,
            "rel_sigma": sigma_rho / max(abs(mu), 1e-10),
            "sigma_depth_m": sigma_rho * depth_est * depth_est,
            "max_parallax_deg": None if max_parallax is None else float(max_parallax),
            "n_obs": len(observations),
            "n_poses": len(poses),
            "first_kf": int(buf["first_state"]),
            "age_kf": int(self.state_index - buf["first_state"]),
        })

    def _promote_landmark(self, landmark_id: int):
        """Move a buffered landmark into the graph once its geometry is trustworthy.

        Gates in order: depth range, cheirality from every observing pose, depth-filter
        convergence, then parallax. A failed gate returns without popping the buffer so
        the landmark can accumulate more views and retry — except depth range and
        depth-filter outliers, which are permanent rejections.

        Args:
            landmark_id: Buffered landmark to attempt promoting.

        Returns:
            None.
        """
        buf = self.landmark_buffer[landmark_id]
        self.gate_stats["p_attempts"] += 1
        # Validate: reject landmarks with degenerate camera-frame depth
        depth = buf["point_cam"][2]
        if depth < self.depth_d_min or depth > self.depth_d_max:
            # Bad triangulation — remove permanently
            self.gate_stats["p_depth_range"] += 1
            self._log_promotion_event(landmark_id, buf, "reject_depth_range")
            self.landmark_buffer.pop(landmark_id)
            return

        # Transform landmark from camera frame to world frame using the first observing pose
        pt3_world = self._landmark_to_world(buf["point_cam"], buf["first_state"])

        # Validate: minimum parallax angle between observing poses
        max_parallax = self._max_parallax_deg(pt3_world, buf["observations"])
        # Record every promotion attempt, including the ones that get rejected below —
        # drain_parallax_stats() reports these per keyframe. Without this append the
        # gating log's parallax columns are identically zero.
        self.parallax_log.append(float(max_parallax))

        # Validate: landmark must be in front of ALL observing cameras + collect depths
        pt_gtsam = gtsam.Point3(*pt3_world)
        observed_depths = []
        for s_idx, _uv in buf["observations"]:
            world_T_cam = self._camera_pose(s_idx)
            if world_T_cam is None:
                continue
            pt_cam = world_T_cam.transformTo(pt_gtsam)
            if pt_cam[2] < 0.2:  # Behind camera or too close
                # Keep in buffer — pose estimates may improve later
                self.gate_stats["p_cheirality"] += 1
                self._log_promotion_event(landmark_id, buf, "retry_cheirality",
                                          pt3_world, max_parallax)
                return
            observed_depths.append(float(pt_cam[2]))

        # --- Depth filter: update regardless of parallax, check convergence ---
        if self.depth_filter_enabled and len(observed_depths) >= 2:
            self._depth_stats["attempted"] += 1
            self._update_depth_filter(buf, observed_depths)
            if buf["df_a"] < self.depth_outlier_thresh and buf["df_n"] >= 3:
                # Landmark is likely an outlier — remove permanently
                self._depth_stats["outlier"] += 1
                self.gate_stats["p_outlier"] += 1
                self._log_promotion_event(landmark_id, buf, "reject_outlier",
                                          pt3_world, max_parallax)
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
                self.gate_stats["p_pending"] += 1
                self._log_promotion_event(landmark_id, buf, "retry_depth_pending",
                                          pt3_world, max_parallax)
                return

        # Parallax gate (checked after depth filter so filter can accumulate)
        if max_parallax < self.parallax_threshold:
            self.gate_stats["p_parallax"] += 1
            self._log_promotion_event(landmark_id, buf, "retry_parallax",
                                      pt3_world, max_parallax)
            return

        # All checks passed — pop from buffer and commit to graph
        self.gate_stats["p_promoted"] += 1
        self._log_promotion_event(landmark_id, buf, "promoted", pt3_world, max_parallax)
        self.landmark_buffer.pop(landmark_id)
        if self.depth_filter_enabled:
            self._depth_stats["promoted"] += 1
            # Log depth filter state at promotion for visualization
            self.promoted_depth_log.append((
                landmark_id, pt3_world.copy(),
                buf["df_mu"], buf["df_sigma2"], buf["df_a"], buf["df_n"],
            ))

        self._commit_landmark(landmark_id, buf, pt3_world, count_flushed=True)

    def _commit_landmark(self, landmark_id: int, buf: dict, pt3_world: np.ndarray,
                         count_flushed: bool = False):
        """Insert a landmark variable, its prior, and all its buffered factors.

        The regularization prior is what keeps the linear system non-singular; the
        buffered observations are flushed together so the landmark enters the graph
        already constrained by every view that earned it.

        Args:
            landmark_id: Landmark being committed.
            buf: Its buffer entry (observations and depth-filter state).
            pt3_world: (3,) world position to initialize at.
            count_flushed: Whether to count the flushed factors in gate stats.

        Returns:
            None.
        """
        self.initial.insert(L(landmark_id), gtsam.Point3(*pt3_world))
        self.landmark_initialized.add(landmark_id)
        self.landmark_obs_count[landmark_id] = 0

        reg_noise = self._landmark_prior_noise(pt3_world, buf)
        self.graph.add(
            gtsam.PriorFactorPoint3(L(landmark_id), gtsam.Point3(*pt3_world), reg_noise)
        )

        for s_idx, uv in buf["observations"]:
            measurement = gtsam.Point2(float(uv[0]), float(uv[1]))
            self.graph.add(
                self._make_projection_factor(measurement, s_idx, landmark_id)
            )
            self.landmark_obs_count[landmark_id] += 1
            if count_flushed:
                self.gate_stats["p_flushed_factors"] += 1

        # Seed the incremental-parallax reference with the most recent observing view
        cam_pos = self._camera_position(buf["observations"][-1][0])
        if cam_pos is not None:
            self.landmark_last_factor_cam[landmark_id] = cam_pos

    def add_loop_closure_observation(
        self, landmark_id: int, state_idx: int, uv: np.ndarray
    ):
        """Add a re-projection factor for a landmark re-detected from a distant pose.

        Uses the tighter loop_pixel_noise: one re-observation has to compete against
        the whole accumulated IMU+visual chain to move the trajectory at all.

        Args:
            landmark_id: Existing landmark in the graph.
            state_idx: Pose state index observing it.
            uv: 2D pixel measurement [u, v].

        Returns:
            None.
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

    def _landmark_prior_noise(self, pt3_world: np.ndarray, buf: dict):
        """Anisotropic regularization prior matched to triangulation uncertainty.

        Projection factors pin bearing to sub-pixel while depth along the ray stays
        ill-conditioned, so an isotropic prior is backwards: negligible laterally, yet
        tight enough along the ray to assert depth confidence the geometry lacks. The
        along-ray sigma is widened to the depth filter's sigma_z, clipped from below so
        it is never tighter than the isotropic prior it replaces.

        Args:
            pt3_world: (3,) landmark position in the world frame.
            buf: Buffer entry supplying first_state and depth-filter state.

        Returns:
            A gtsam noise model — anisotropic, or isotropic if the ray is degenerate.
        """
        sigma_perp = float(self.landmark_reg_sigma)
        iso = gtsam.noiseModel.Isotropic.Sigma(3, sigma_perp)

        ref_pos = self._camera_position(buf.get("first_state"))
        if ref_pos is None:
            return iso
        ray = np.asarray(pt3_world, dtype=float) - np.asarray(ref_pos, dtype=float)
        ray_norm = np.linalg.norm(ray)
        if ray_norm < 1e-9:
            return iso
        f_hat = ray / ray_norm

        # sigma_z = sigma_rho * z^2  (uncertainty propagated from inverse depth)
        depth = 1.0 / max(abs(buf["df_mu"]), 1e-10)
        sigma_z = float(np.sqrt(max(buf["df_sigma2"], 0.0))) * depth * depth
        sigma_along = float(np.clip(
            self.landmark_reg_along_k * sigma_z,
            sigma_perp,
            self.landmark_reg_along_max,
        ))
        if not np.isfinite(sigma_along):
            return iso

        fft = np.outer(f_hat, f_hat)
        cov = sigma_perp ** 2 * (np.eye(3) - fft) + sigma_along ** 2 * fft
        try:
            return gtsam.noiseModel.Gaussian.Covariance(cov)
        except Exception:
            return iso

    def _update_depth_filter(self, buf: dict, observed_depths: List[float]):
        """Fold depth measurements into a landmark's Gaussian+uniform depth filter.

        Filtering happens in inverse depth, where stereo triangulation noise is roughly
        Gaussian and depth-independent; in metric depth it is neither.

        Args:
            buf: Buffer entry holding df_mu, df_sigma2, df_a, df_n (mutated in place).
            observed_depths: Metric depths of this landmark from each observing camera.

        Returns:
            None.
        """
        d_min, d_max = self.depth_d_min, self.depth_d_max
        p_uniform = 1.0 / (1.0 / d_min - 1.0 / d_max)
        tau2 = self.depth_tau ** 2

        for depth in observed_depths:
            if depth <= d_min or depth > d_max:
                continue
            buf["df_mu"], buf["df_sigma2"], buf["df_a"] = gaussian_uniform_update(
                buf["df_mu"], buf["df_sigma2"], buf["df_a"],
                z=1.0 / depth, tau2=tau2, p_uniform=p_uniform,
            )
            buf["df_n"] += 1

    def _max_parallax_deg(self, pt3_world: np.ndarray, observations: list) -> float:
        """Widest parallax angle (deg) between any two cameras observing a landmark.

        This is the single strongest quality signal for a landmark: triangulation
        variance scales as ~1/sin(parallax)^2, so a low-parallax point has near-parallel
        rays, near-zero Jacobians, and can render the linear system indeterminate.
        Gating on it markedly improves both landmark quality and ISAM2 stability.

        Args:
            pt3_world: (3,) landmark position in the world frame.
            observations: [(state_idx, uv), ...] views of the landmark.

        Returns:
            Maximum pairwise parallax in degrees, 0.0 with fewer than two known poses.
        """
        cam_positions = []
        for s_idx, _ in observations:
            world_T_cam = self._camera_pose(s_idx)
            if world_T_cam is not None:
                cam_positions.append(np.array(world_T_cam.translation()))

        if len(cam_positions) < 2:
            return 0.0

        max_angle = 0.0
        for i in range(len(cam_positions)):
            ray_i = pt3_world - cam_positions[i]
            for j in range(i + 1, len(cam_positions)):
                max_angle = max(
                    max_angle, angle_between_deg(ray_i, pt3_world - cam_positions[j])
                )
        return max_angle

    def _camera_position(self, state_idx: int) -> Optional[np.ndarray]:
        """World position of the camera center for pose X(state_idx).

        Args:
            state_idx: Pose state index.

        Returns:
            (3,) camera center in world coordinates, or None if the pose is unknown.
        """
        world_T_cam = self._camera_pose(state_idx)
        if world_T_cam is None:
            return None
        return np.array(world_T_cam.translation())

    def _obs_parallax_deg(self, landmark_id: int, state_idx: int) -> Optional[float]:
        """Incremental parallax (deg) since the last view that contributed a factor.

        Args:
            landmark_id: Landmark being re-observed.
            state_idx: Pose state index of the new observation.

        Returns:
            Angle in degrees, or None if there is no reference view yet — which tells
            the caller not to gate on parallax rather than to reject.
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
        """Check that a landmark sits in front of the camera at X(state_idx).

        A landmark behind the camera gives the projection factor a singular Jacobian,
        so it must be rejected rather than merely down-weighted. Unknown geometry
        returns True: this gate should not drop observations it cannot evaluate.

        Args:
            landmark_id: Landmark to test.
            state_idx: Pose state index to test from.

        Returns:
            True if depth exceeds 10 cm, or if the check cannot be made.
        """
        estimate = self.get_current_estimate()
        if estimate is None or not estimate.exists(L(landmark_id)):
            return True
        world_T_cam = self._camera_pose(state_idx)
        if world_T_cam is None:
            return True
        pt_world = estimate.atPoint3(L(landmark_id))
        return world_T_cam.transformTo(gtsam.Point3(*pt_world))[2] > 0.1

    def _landmark_to_world(self, point_cam: np.ndarray, state_idx: int) -> np.ndarray:
        """Lift a camera-frame point into the world using pose X(state_idx).

        Args:
            point_cam: (3,) point in the camera frame.
            state_idx: Pose the point was observed from.

        Returns:
            (3,) point in world coordinates; identity pose is assumed if X is unknown.
        """
        world_T_cam = self._camera_pose(state_idx, default=gtsam.Pose3())
        pt_world = world_T_cam.transformFrom(gtsam.Point3(*point_cam))
        return np.array([pt_world[0], pt_world[1], pt_world[2]])

    # ==================== Optimization ====================

    def optimize(self):
        """Fold the pending factors and values into the solver and re-solve.

        A second ISAM2 pass is triggered only after a large relinearization, where one
        Gauss-Newton step is unlikely to have converged.

        Returns:
            The updated gtsam.Values, or None without gtsam.
        """
        if gtsam is None:
            return None
        self._cached_estimate = None  # Invalidate cache
        if self.isam is not None:
            try:
                update_result = self.isam.update(self.graph, self.initial)
                self.graph.resize(0)
                self.initial.clear()
                if update_result.getVariablesRelinearized() > 50:
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
        """Latest solved values, cached until the next optimize() invalidates it.

        Returns:
            gtsam.Values, or None if nothing has been solved yet.
        """
        if gtsam is None:
            return None
        if self._cached_estimate is not None:
            return self._cached_estimate
        if self.isam is not None:
            self._cached_estimate = self.isam.calculateEstimate()
            return self._cached_estimate
        return None

    def get_optimized_landmarks(self) -> Dict[int, np.ndarray]:
        """Read back every initialized landmark's optimized position.

        Returns:
            {landmark_id: (3,) world position} for landmarks present in the estimate.
        """
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
        """One-line graph size summary for per-keyframe logging.

        Returns:
            Formatted string with state, landmark, observation and factor counts.
        """
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
        """Graph quality metrics, with the expensive term refreshed periodically.

        factors.error() re-evaluates the whole graph, so its cost grows linearly with
        trajectory length for telemetry alone. It refreshes every diagnostics_interval
        keyframes and is held in between, keyed on state_index so repeated calls within
        one keyframe cannot skip a refresh; the counts stay exact every call.

        Returns:
            Dict of error, variable/factor/landmark counts, and error_age_kf staleness.
        """
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

        n_vars = result.size()
        n_factors = factors.size()

        due = (
            self._diag_last_idx is None
            or self.state_index - self._diag_last_idx >= self.diagnostics_interval
        )
        if due:
            self._diag_total_error = factors.error(result)
            self._diag_last_idx = self.state_index
        total_error = self._diag_total_error
        avg_error = total_error / max(n_factors, 1)

        return {
            "total_error": total_error,
            "n_variables": n_vars,
            "n_factors": n_factors,
            "avg_error_per_factor": avg_error,
            "n_landmarks": len(self.landmark_initialized),
            "error_age_kf": self.state_index - self._diag_last_idx,
        }

    def drain_parallax_stats(self):
        """Return and clear parallax angles recorded since the last call.

        Returns:
            Dict with parallax_max_deg, parallax_median_deg and parallax_count.
        """
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
        """Return and clear depth-filter outcome counts since the last call.

        Returns:
            Dict with attempted, promoted, outlier and pending counts.
        """
        stats = dict(self._depth_stats)
        self._depth_stats = {"attempted": 0, "promoted": 0, "outlier": 0, "pending": 0}
        return stats

    def landmark_promotion_report(self, with_covariance: bool = True) -> Dict:
        """Bundle the promotion trace with the final graph state for offline analysis.

        Pairs each landmark's state at promotion with where the optimizer ultimately put
        it, which is what makes the depth uncertainty checkable: a well-calibrated sigma
        should bracket how far the solution actually moved.

        Args:
            with_covariance: Also query each landmark's marginal covariance. One
                marginals solve per landmark, so it is only affordable at shutdown.

        Returns:
            Dict with events, params, final landmark states, poses and leftover buffer.
        """
        report = {
            "events": list(self.promotion_events),
            "params": {
                "parallax_threshold": self.parallax_threshold,
                "min_obs_parallax_deg": self.min_obs_parallax_deg,
                "depth_d_min": self.depth_d_min,
                "depth_d_max": self.depth_d_max,
                "depth_tau": self.depth_tau,
                "depth_convergence_rel": self.depth_convergence_rel,
                "depth_max_sigma_m": self.depth_max_sigma_m,
                "depth_min_inlier": self.depth_min_inlier,
                "depth_outlier_thresh": self.depth_outlier_thresh,
                "buffer_max_age_kf": self.buffer_max_age_kf,
                "obs_cell_size": self.obs_cell_size,
                "landmark_reg_sigma": self.landmark_reg_sigma,
                "landmark_reg_along_k": self.landmark_reg_along_k,
                "landmark_reg_along_max": self.landmark_reg_along_max,
            },
            "landmarks": {},
            "poses": {},
            "buffer_at_end": [],
            "n_states": int(self.state_index) + 1,
        }
        if gtsam is None:
            return report

        estimate = self.get_current_estimate()
        if estimate is not None:
            for idx in range(self.state_index + 1):
                if estimate.exists(X(idx)):
                    t = estimate.atPose3(X(idx)).translation()
                    report["poses"][idx] = np.array([t[0], t[1], t[2]])

        for lm_id in self.landmark_initialized:
            entry = {
                "obs_count": int(self.landmark_obs_count.get(lm_id, 0)),
                "frozen": lm_id in self.landmark_frozen,
                "frozen_kf": self.landmark_frozen_kf.get(lm_id, (None, None))[0],
                "position": None,
                "cov_trace": None,
            }
            try:
                if estimate is not None and estimate.exists(L(lm_id)):
                    p = estimate.atPoint3(L(lm_id))
                    entry["position"] = np.array([p[0], p[1], p[2]])
            except Exception:
                pass
            if with_covariance and self.isam is not None:
                try:
                    entry["cov_trace"] = float(np.trace(self.isam.marginalCovariance(L(lm_id))))
                except Exception:
                    pass
            report["landmarks"][int(lm_id)] = entry

        # Landmarks that never escaped the buffer: the silent majority of the funnel
        for lm_id, buf in self.landmark_buffer.items():
            report["buffer_at_end"].append({
                "lm_id": int(lm_id),
                "df_mu": float(buf["df_mu"]),
                "df_sigma2": float(buf["df_sigma2"]),
                "df_a": float(buf["df_a"]),
                "df_n": int(buf["df_n"]),
                "n_obs": len(buf["observations"]),
                "n_poses": len(buf.get("poses", set())),
                "first_kf": int(buf["first_state"]),
                "last_kf": int(buf["observations"][-1][0]),
            })
        return report

    def prune_stale_buffer(self, current_kf_idx: int) -> int:
        """Drop buffered landmarks the camera has stopped seeing.

        Age is measured from the LAST observation: a landmark first seen 50 keyframes
        ago but still tracked is alive, whereas one unseen for 20 is gone, since the
        frontend only re-observes what KLT still follows. Without pruning the buffer is
        monotonic — every landmark that never reaches 3 poses stays forever.

        Args:
            current_kf_idx: Current keyframe index.

        Returns:
            Number of pruned entries.
        """
        if self.buffer_max_age_kf <= 0:
            return 0
        cutoff = current_kf_idx - self.buffer_max_age_kf
        stale = [lm_id for lm_id, buf in self.landmark_buffer.items()
                 if buf["observations"][-1][0] < cutoff]
        for lm_id in stale:
            self._log_promotion_event(lm_id, self.landmark_buffer[lm_id], "buffer_pruned")
            del self.landmark_buffer[lm_id]
        return len(stale)

    # ==================== Landmark Uncertainty Filtering ====================

    def filter_uncertain_landmarks(self, max_trace: float = 2.0) -> int:
        """Freeze landmarks whose marginal position uncertainty is too large.

        Frozen landmarks stop receiving projection factors, so a badly-constrained
        point cannot keep dragging poses. Costs one marginal covariance per landmark,
        which is why it is not called every keyframe.

        Args:
            max_trace: Maximum allowed trace of the 3x3 position covariance (m^2).

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
                    self.landmark_frozen_kf[lm_id] = (self.state_index, float(np.trace(cov)))
                    newly_frozen += 1
            except Exception:
                # Covariance computation can fail for poorly connected variables
                self.landmark_frozen.add(lm_id)
                self.landmark_frozen_kf[lm_id] = (self.state_index, float("nan"))
                newly_frozen += 1

        if newly_frozen > 0:
            print(
                f"  Froze {newly_frozen} high-uncertainty landmarks "
                f"(total frozen: {len(self.landmark_frozen)}/{len(self.landmark_initialized)})"
            )
        return newly_frozen

    # ==================== Covariance & Degeneracy ====================

    def get_pose_covariance(self, state_idx: int) -> Optional[np.ndarray]:
        """Marginal covariance of pose X(state_idx).

        Args:
            state_idx: Pose state index.

        Returns:
            (6, 6) covariance ordered [rx, ry, rz, tx, ty, tz], or None if it fails.
        """
        if gtsam is None or self.isam is None:
            return None
        try:
            return self.isam.marginalCovariance(X(state_idx))
        except Exception:
            return None

    def get_position_covariance(self, state_idx: int) -> Optional[np.ndarray]:
        """Translation block of the pose marginal covariance.

        Args:
            state_idx: Pose state index.

        Returns:
            (3, 3) position covariance, or None if it could not be computed.
        """
        cov6 = self.get_pose_covariance(state_idx)
        if cov6 is None:
            return None
        return cov6[3:6, 3:6]

    def detect_degeneracy(self, state_idx: int, pos_cov: Optional[np.ndarray] = None) -> Dict:
        """Classify motion degeneracy from the shape of the position covariance.

        The eigenvectors say WHICH direction is unobservable and the condition number
        says how badly, which is what distinguishes a genuinely degenerate motion from
        merely large uncertainty.

        Args:
            state_idx: Pose state index to analyze.
            pos_cov: Pre-computed 3x3 position covariance; computed internally if None.

        Returns:
            Dict with position_eigenvalues/eigenvectors, condition_number, degenerate,
            degenerate_direction and motion_type.
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
        """Render the factor graph in 3D: poses, landmarks, and typed factor edges.

        The usual VIO factor-graph figure but in real coordinates — poses in red,
        landmarks coloured by depth-filter sigma, projection factors grey, IMU orange,
        between-factors purple.

        Args:
            output_path: Path to write the interactive HTML file.

        Returns:
            The plotly Figure, or None if plotly is missing or nothing is solved.
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


