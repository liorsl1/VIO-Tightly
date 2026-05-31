import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import gtsam
    from gtsam.symbol_shorthand import X, V, B, L
except ImportError:  # Allow file existence without immediate dependency
    gtsam = None  # type: ignore


class GraphOptimizer:
    """Tightly coupled VIO/SLAM backend using GTSAM with explicit landmarks.

    Uses GenericProjectionFactorPose3Point3 for visual factors, allowing:
      - Incremental observation addition to existing landmarks in ISAM2
      - Direct access to optimized 3D landmark positions
      - Natural loop closure integration via re-observation factors
      - Robust Huber loss on reprojection errors
    """
    def __init__(self, use_isam: bool = True, body_P_sensor: np.ndarray = None):
        if gtsam is None:
            self.isam = None
            self.graph = None
            self.initial = None
            return

        # body_P_sensor: 4x4 transform T_body_cam (pose of camera in body/IMU frame)
        if body_P_sensor is not None:
            R = gtsam.Rot3(body_P_sensor[:3, :3])
            t = gtsam.Point3(body_P_sensor[:3, 3])
            self.body_P_sensor = gtsam.Pose3(R, t)
        else:
            self.body_P_sensor = None

        self.use_isam = use_isam
        self.isam_params = gtsam.ISAM2Params()
        self.isam_params.setRelinearizeThreshold(0.01)
        self.isam_params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(self.isam_params) if use_isam else None
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.state_index = 0

        # --- Landmark management ---
        self.landmark_initialized = set()  # L(id) already in ISAM2 Values
        self.landmark_obs_count = {}       # {landmark_id: number of observations added}
        # Buffer: landmarks wait here until they have 2+ observations from different poses
        # {landmark_id: {"point_cam": np.array, "first_state": int, "observations": [(state_idx, uv), ...]}}
        self.landmark_buffer = {}
        self.cal = None                    # gtsam.Cal3_S2, set on first observation

        # Cached estimate (invalidated after each optimize call)
        self._cached_estimate = None

        # --- Noise models ---
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
        self.prior_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.prior_bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-2)
        # Landmark regularization prior (sigma=1.5m).
        # Tight enough to prevent ISAM2 from pushing landmarks to degenerate positions
        # during relinearization, but loose enough not to bias converged estimates.
        self.landmark_regularization_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.5)

        # Robust pixel noise (Huber) for projection factors
        pixel_sigma = 2.0  # pixels
        pixel_noise_base = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
        self.pixel_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), pixel_noise_base)

    def _make_projection_factor(self, measurement, state_idx: int, landmark_id: int):
        """Create a projection factor with throwCheirality=False.

        Setting throwCheirality=False makes the factor return zero error (and
        zero Jacobians) when the landmark projects behind the camera, preventing
        IndeterminateLinearSystemException during ISAM2 relinearization.
        """
        if self.body_P_sensor is not None:
            # Signature: (measured, noise, poseKey, pointKey, K, throwCheirality, verboseCheirality, body_P_sensor)
            return gtsam.GenericProjectionFactorCal3_S2(
                measurement, self.pixel_noise,
                X(state_idx), L(landmark_id),
                self.cal, False, False, self.body_P_sensor)
        else:
            # Signature: (measured, noise, poseKey, pointKey, K, throwCheirality, verboseCheirality)
            return gtsam.GenericProjectionFactorCal3_S2(
                measurement, self.pixel_noise,
                X(state_idx), L(landmark_id),
                self.cal, False, False)

    # ==================== Initialization ====================

    def add_initial_state(self, pose_wb, vel_w: np.ndarray, bias, set_priors=True):
        if gtsam is None:
            return
        if isinstance(pose_wb, gtsam.Pose3):
            pose3 = pose_wb
        elif isinstance(pose_wb, np.ndarray) and pose_wb.shape == (4, 4):
            pose3 = gtsam.Pose3(gtsam.Rot3(pose_wb[:3, :3]), gtsam.Point3(*pose_wb[:3, 3]))
        else:
            pose3 = gtsam.Pose3()

        self.initial.insert(X(self.state_index), pose3)
        self.initial.insert(V(self.state_index), gtsam.Point3(*vel_w))
        self.initial.insert(B(self.state_index), bias)
        if set_priors and self.state_index == 0:
            self.graph.add(gtsam.PriorFactorPose3(X(0), pose3, self.prior_pose_noise))
            self.graph.add(gtsam.PriorFactorVector(V(0), gtsam.Point3(*vel_w), self.prior_vel_noise))
            self.graph.add(gtsam.PriorFactorConstantBias(B(0), bias, self.prior_bias_noise))

    # ==================== IMU Factors ====================

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
        fac = gtsam.ImuFactor(X(prev_idx), V(prev_idx),
                              X(curr_idx), V(curr_idx),
                              B(prev_idx), preint)
        self.graph.add(fac)
        # Bias random walk between consecutive states
        # Sigma should be loose enough to accommodate actual bias drift over the
        # inter-keyframe interval. Reference: GTSAM official example uses variance=0.1
        # (sigma~0.316). For EuRoC with large frame steps, sigma=0.3 is appropriate.
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.3)
        self.graph.add(gtsam.BetweenFactorConstantBias(
            B(prev_idx), B(curr_idx),
            gtsam.imuBias.ConstantBias(), bias_noise))

    # ==================== Visual Factors (Explicit Landmarks) ====================

    def add_landmark_observation(self, landmark_id: int, state_idx: int,
                                  uv: np.ndarray, K: np.ndarray,
                                  landmark_3d: np.ndarray = None):
        """Add a projection factor between pose X(state_idx) and landmark L(landmark_id).

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
            # Depth check: skip if landmark is behind the camera from this pose
            if not self._is_landmark_in_front(landmark_id, state_idx):
                return
            measurement = gtsam.Point2(float(uv[0]), float(uv[1]))
            self.graph.add(self._make_projection_factor(measurement, state_idx, landmark_id))
            self.landmark_obs_count[landmark_id] = self.landmark_obs_count.get(landmark_id, 0) + 1
            return

        # --- Case 2: Landmark in buffer — add observation and maybe promote ---
        if landmark_id in self.landmark_buffer:
            buf = self.landmark_buffer[landmark_id]
            buf["observations"].append((state_idx, np.array(uv, dtype=float)))
            # Check if we have observations from 2+ distinct poses
            distinct_poses = set(s for s, _ in buf["observations"])
            if len(distinct_poses) >= 2:
                self._promote_landmark(landmark_id)
            return

        # --- Case 3: First time seeing this landmark — add to buffer ---
        if landmark_3d is None:
            return  # Cannot initialize without 3D position
        self.landmark_buffer[landmark_id] = {
            "point_cam": np.array(landmark_3d, dtype=float),
            "first_state": state_idx,
            "observations": [(state_idx, np.array(uv, dtype=float))],
        }

    def _promote_landmark(self, landmark_id: int):
        """Move a landmark from the buffer into the factor graph (ISAM2).

        Called when a buffered landmark has 2+ observations from different poses.
        Adds the landmark variable + all buffered projection factors at once.
        Validates that the landmark is geometrically visible (positive depth) from
        all observing poses before committing — prevents indeterminate systems from
        zero-Jacobian projection factors.
        """
        buf = self.landmark_buffer.pop(landmark_id)
        # Validate: reject landmarks with degenerate camera-frame depth
        depth = buf["point_cam"][2]
        if depth < 0.3 or depth > 30.0:
            # Bad triangulation — don't add to graph
            return
        # Transform landmark from camera frame to world frame using the first observing pose
        pt3_world = self._landmark_to_world(buf["point_cam"], buf["first_state"])

        # Validate: minimum baseline between observing poses (poor triangulation otherwise)
        obs_states = [s for s, _ in buf["observations"]]
        if len(obs_states) >= 2:
            estimate = self.get_current_estimate()
            poses = []
            for s_idx in obs_states:
                if estimate is not None and estimate.exists(X(s_idx)):
                    poses.append(estimate.atPose3(X(s_idx)).translation())
                elif self.initial.exists(X(s_idx)):
                    poses.append(self.initial.atPose3(X(s_idx)).translation())
            if len(poses) >= 2:
                max_baseline = max(
                    np.linalg.norm(np.array(poses[i]) - np.array(poses[j]))
                    for i in range(len(poses)) for j in range(i+1, len(poses))
                )
                if max_baseline < 0.05:  # Less than 5cm baseline → unreliable depth
                    return

        # Validate: landmark must be in front of ALL observing cameras
        pt_gtsam = gtsam.Point3(*pt3_world)
        for (s_idx, _uv) in buf["observations"]:
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
            if pt_cam[2] < 0.5:  # Behind camera or too close
                return  # Reject this landmark entirely

        self.initial.insert(L(landmark_id), gtsam.Point3(*pt3_world))
        self.landmark_initialized.add(landmark_id)
        self.landmark_obs_count[landmark_id] = 0

        # Regularization prior to prevent indeterminate linear system
        self.graph.add(gtsam.PriorFactorPoint3(
            L(landmark_id), gtsam.Point3(*pt3_world),
            self.landmark_regularization_noise))

        # Add all buffered projection factors
        for (s_idx, uv) in buf["observations"]:
            measurement = gtsam.Point2(float(uv[0]), float(uv[1]))
            self.graph.add(self._make_projection_factor(measurement, s_idx, landmark_id))
            self.landmark_obs_count[landmark_id] += 1

    def add_loop_closure_observation(self, landmark_id: int, state_idx: int,
                                      uv: np.ndarray):
        """Add a reprojection factor for a loop closure re-observation.

        This is for when you re-detect an existing landmark from a distant frame.
        The landmark must already be initialized in the graph.
        """
        if gtsam is None or self.cal is None:
            return
        if landmark_id not in self.landmark_initialized:
            return  # Can't add factor to non-existent landmark

        measurement = gtsam.Point2(float(uv[0]), float(uv[1]))
        self.graph.add(self._make_projection_factor(measurement, state_idx, landmark_id))
        self.landmark_obs_count[landmark_id] = self.landmark_obs_count.get(landmark_id, 0) + 1

    def add_loop_closure_pose_constraint(self, current_idx: int, matched_idx: int,
                                          relative_pose,
                                          noise_sigmas: np.ndarray = None):
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
            noise_sigmas = np.array([0.05, 0.05, 0.05, 0.15, 0.15, 0.15])
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
        self.graph.add(gtsam.BetweenFactorPose3(
            X(matched_idx), X(current_idx), relative_pose, noise_model))

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
                self.isam.update(self.graph, self.initial)
                self.graph.resize(0)
                self.initial.clear()
                # One extra iteration helps convergence
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
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial)
            result = optimizer.optimize()
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
        return (f"States: {self.state_index+1}, Landmarks: {n_landmarks} "
                f"(+{n_buffered} buffered), "
                f"Observations: {n_total_obs}, Factors(pending): {self.graph.size()}")

    def diagnostics(self):
        """Return optimization quality metrics."""
        if gtsam is None or self.isam is None:
            return {}
        result = self.isam.calculateEstimate()
        total_error = self.isam.getFactorsUnsafe().error(result)
        n_vars = result.size()
        n_factors = self.isam.getFactorsUnsafe().size()
        avg_error = total_error / max(n_factors, 1)
        metrics = {
            "total_error": total_error,
            "n_variables": n_vars,
            "n_factors": n_factors,
            "avg_error_per_factor": avg_error,
            "n_landmarks": len(self.landmark_initialized),
        }
        return metrics
