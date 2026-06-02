import numpy as np
import cv2
import torch
from lightglue import LightGlue, SuperPoint
from collections import defaultdict
import plotly.graph_objects as go
import plotly.io as pio
import hnswlib
from scipy.spatial import cKDTree

pio.renderers.default = "browser"

class vFeature:
    def __init__(self, matcher_type="superpoint", device="cpu", baseline=None, 
                 intrinsics=None, dist_coeffs=None, T_cam1_cam0=None, max_features=2048):
        # --- Existing attributes ---
        self.matcher_type = matcher_type
        self.device = device
        self.baseline = baseline
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        self.max_features = max_features
        self.T_cam1_cam0 = T_cam1_cam0  # Camera 1 to Camera 0 transformation
        self.t_cam1_cam0 = T_cam1_cam0[:3, 3]  # Translation from Camera 1 to Camera 0
        self.R_cam1_cam0 = T_cam1_cam0[:3, :3]  # Rotation from Camera 1 to Camera 0
        self.T_cam0_cam1 = np.linalg.inv(T_cam1_cam0)  # Camera 0 to Camera 1 transformation
        self.P1 = None  # Rectified projection matrix for Camera 0
        self.P2 = None  # Rectified projection matrix for Camera 1

        # --- New tightly-coupled attributes ---
        self.feature_tracks = {}  # {track_id: [observations]}
        self.landmarks = {}  # {landmark_id: 3D_position}
        self.landmark_observations = defaultdict(list)  # {landmark_id: [(frame_id, keypoint)]}
        self.current_frame_id = 0
        self.next_track_id = 0
        self.next_landmark_id = 0
        
        # --- Feature tracking state ---
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None
        self.prev_track_ids = None  # Track IDs from the previous frame
        
        # --- Optimization interface ---
        self.new_landmarks = []  # Landmarks ready for optimization
        self.new_observations = []  # New feature observations
        
        # --- HNSW retrieval structures ---
        self.hnsw_index = None
        self.hnsw_dim = 256          # SuperPoint descriptor dimension
        self.hnsw_max_elements = 200000
        self.hnsw_elements = 0
        self.hnsw_inited = False
        self.hnsw_space = 'l2'       # or 'cosine'
        self.hnsw_new_buffer = []    # (landmark_id, descriptor)
        self.landmark_desc = {}      # landmark_id -> running mean descriptor (np.float32, shape (D,))
        self.landmark_desc_counts = {}  # landmark_id -> num updates
        self.landmark_last_frame = {}   # landmark_id -> last seen frame
        self.landmark_to_frame = {}     # landmark_id -> first frame_id
        self.frame_landmarks = {}       # frame_id -> set(landmark_ids)

        # Initialize matchers (keep existing)
        self.initialize()

    def initialize(self):
        """Initialize feature extractors and matchers (keep existing implementation)"""
        
        print("Initializing SuperPoint and LightGlue...")
        self.superpoint = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.lg_matcher = LightGlue(features="superpoint").eval().to(self.device)

        if self.intrinsics is not None:
            if isinstance(self.intrinsics, (list, tuple)):
                self.fx, self.fy, self.cx, self.cy = self.intrinsics
            elif isinstance(self.intrinsics, np.ndarray) and self.intrinsics.shape == (
                3,
                3,
            ):
                self.fx = self.intrinsics[0, 0]
                self.fy = self.intrinsics[1, 1]
                self.cx = self.intrinsics[0, 2]
                self.cy = self.intrinsics[1, 2]
            else:
                raise ValueError(
                    "Invalid intrinsics format. Must be a list/tuple or a 3x3 numpy array."
                )


    def skew(self, t):
        tx, ty, tz = t.flatten()
        return np.array([[0, -tz, ty],
                        [tz, 0, -tx],
                        [-ty, tx, 0]])


    def stereo_match_rectified(self, cam0_points, cam1_points):
        """
        Simplified stereo matching on rectified images.
        """
        if len(cam0_points) == 0:
            return [], []
        
        # # Points are already in rectified space, no need for complex rectification
        # cam1_points, inlier_markers, _ = cv2.calcOpticalFlowPyrLK(
        #     cam0_rect,
        #     cam1_rect, 
        #     cam0_points.astype(np.float32),
        #     cam0_points.astype(np.float32),  # Initial guess: same position
        #     **self.config.lk_params
        # )
        E = self.skew(self.t_cam1_cam0) @ self.R_cam1_cam0
        # make cam_points homogeneous
        cam0_points = np.hstack([cam0_points, np.ones((cam0_points.shape[0], 1))])
        cam1_points = np.hstack([cam1_points, np.ones((cam1_points.shape[0], 1))])
        # Apply epipolar constraint
        epipolar_line = (E @ cam0_points.T).T  # Epipolar line in cam1 frame
        # Numerator: x2^T * l2  -> for each correspondence, scalar
        numerators = np.abs(np.sum(cam1_points * epipolar_line, axis=1))  # (N,)

        # Denominator: sqrt(a^2 + b^2) where line is [a, b, c]
        denominators = np.linalg.norm(epipolar_line[:, :2], axis=1)  # (N,)

        # Avoid division by zero
        valid = denominators > 0
        errors = np.zeros_like(denominators)
        errors[valid] = numerators[valid] / denominators[valid]

        # Simple disparity check (epipolar constraint is just horizontal)
        disparity = cam0_points[:, 0] - cam1_points[:, 0]  # Horizontal disparity
        vertical_error = np.abs(cam0_points[:, 1] - cam1_points[:, 1])  # Should be ~0
        #print(disparity, vertical_error, errors.mean())
        # Filter based on reasonable disparity and epipolar constraint
        valid_matches = np.logical_and.reduce([
            disparity > 0,  # Positive disparity
            disparity < 40,  # Reasonable disparity limit
            vertical_error < 2.5  # Tight epipolar constraint
        ])

        return valid_matches

    def visualize_stereo_matches(self, left_img, right_img, left_points, right_points):
        """
        Visualize stereo matches between rectified images
        Args:
            left_img: Left rectified image
            right_img: Right rectified image
            left_points: Keypoints from left image (Nx2 array)
            right_points: Corresponding keypoints from right image (Nx2 array)
        Returns:
            matched_img: Visualization image with matches drawn
        """
        # Convert points to KeyPoint objects for visualization
        left_kps = [cv2.KeyPoint(x=p[0], y=p[1], size=10) for p in left_points]
        right_kps = [cv2.KeyPoint(x=p[0], y=p[1], size=10) for p in right_points]

        # Create DMatch objects (simple 1-to-1 matching)
        matches = [cv2.DMatch(i, i, 0) for i in range(len(left_kps))]

        # Draw matches
        matched_img = cv2.drawMatches(
            left_img, left_kps,
            right_img, right_kps,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0),  # Green color for matches
            singlePointColor=None,
            matchesMask=None
        )

        # Add epipolar line visualization (horizontal lines for rectified images)
        h, w = left_img.shape[:2]
        for pt in left_points:
            cv2.line(matched_img, 
                    (int(pt[0]), int(pt[1])), 
                    (w + int(pt[0]), int(pt[1])), 
                    (255, 0, 0), 1)  # Blue epipolar lines

        return matched_img

    # ==================== CORE TIGHTLY-COUPLED FUNCTIONS ====================
    
    # def process_stereo_frame(self, left_img, right_img, imu_pose=None):
    #     """
    #     Main processing function for tightly-coupled SLAM.
        
    #     Args:
    #         left_img: Left stereo image
    #         right_img: Right stereo image  
    #         imu_pose: Current IMU pose estimate (4x4 matrix)
            
    #     Returns:
    #         observations: List of feature observations for optimization
    #         new_landmarks: List of newly triangulated landmarks
    #     """
    #     # 1. Extract features from left image
    #     # 3. Track features from previous frame
    #     # 2. Perform stereo matching for depth
    #     # 4. Triangulate new landmarks
    #     # 5. Update feature tracks
    #     # 6. Return observations for optimization
    #     observations = []
    #     tracked_ids = []
    #     features_left, matches = self.extract_and_match(left_img, right_img)
    #     if len(features_left) == 0:
    #         return [], []
    #     left_points = matches[0]
    #     right_points = matches[1]
    #     matches = matches[2]
    #     # 2. Perform stereo matching for depth
    #     # And filter matches based on epipolar constraint
    #     valid_matches = self.stereo_match_rectified(left_points, right_points)
    #     if len(valid_matches.nonzero()[0]) == 0:
    #         return [], []
    #     # visualize matches over left right images
    #     left_points_valid = left_points[valid_matches]
    #     right_points_valid = right_points[valid_matches]
    #     # matched_img = self.visualize_stereo_matches(left_img, right_img, left_points_valid, right_points_valid)
    #     # cv2.imshow("Stereo Matches", matched_img)
    #     # cv2.waitKey(0)
    #     # 4. Triangulate new landmarks
    #     # 3. Track features from previous frame
    #     if self.prev_keypoints is not None:
    #         tracked_ids = self.track_features_temporal(
    #             self.prev_frame, left_img, self.prev_keypoints
    #         )
    #         if self.prev_track_ids is not None:
    #             tracked_ids = self.prev_track_ids[tracked_ids]

    #     points_3d, valid_mask = self.stereo_triangulation(left_points_valid, right_points_valid, imu_pose)
        
    #     # get valid points after reprojection error filtering
    #     valid_curr_keypts = left_points_valid[valid_mask]

    #     if len(points_3d) == 0:
    #         return [], []
        
            
    #         # 5. Update feature tracks
    #         track_ids = self.manage_landmarks(points_3d, tracked_ids, self.current_frame_id)
    #         # 6. Create observations for optimization
    #         observations = self.create_observations_for_optimization(features_left, track_ids, self.current_frame_id)
        
    #     # 7. Update previous frame state
    #     self.prev_keypoints = valid_curr_keypts
    #     #self.prev_descriptors = current_descriptors
    #     self.prev_frame = left_img.copy()
    #     self.current_frame_id += 1
    #     # 8. Return observations and new landmarks
    #     new_landmarks = points_3d
    #     return observations, new_landmarks

    def preprocess_for_matching(self, img):
        """
        Preprocess image for feature matching.
        For 'disk': expects (H, W, 3) RGB, normalized to [0,1], shape (1, 3, H, W).
        For 'loftr' and 'superpoint': expects (H, W) grayscale, normalized to [0,1], shape (1, 1, H, W).
        """
        if self.matcher_type == "disk":
            if img.ndim == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None]  # (1, 3, H, W)
        elif self.matcher_type in ["loftr", "superpoint"]:
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img)[None, None]  # (1, 1, H, W)
        else:
            raise ValueError("Unknown matcher_type for preprocessing.")
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def extract_features(self, image, num_features=2048):
        """Extract features from an image using the selected model."""
        if self.matcher_type == "disk":
            return self._extract_features_disk(image, num_features)
        elif self.matcher_type == "superpoint":
            return self._extract_features_superpoint(image)
        else:
            raise NotImplementedError(
                "Feature extraction not implemented for this matcher_type."
            )

    def _extract_features_disk(self, image, num_features):
        with torch.inference_mode():
            features = self.disk(image, num_features, pad_if_not_divisible=True)
        return features

    def _extract_features_superpoint(self, image):
        with torch.inference_mode():
            feats = self.superpoint({"image": image})
        return feats

    def _match_features_loftr(self, image0, image1, num_features):
        with torch.inference_mode():
            input_dict = {"image0": image0, "image1": image1}
            features = self.matcher(input_dict)
        return features

    def match_features(self, features1, features2, img_shape1, img_shape2, device):
        """Compute tentative matches between features using LightGlueMatcher."""
        if self.matcher_type == "superpoint":
            with torch.inference_mode():
                matches = self.lg_matcher({"image0": features1, "image1": features2})
            matches01 = matches["matches"][0]
            dists = matches["scores"][0]
            return matches01, dists, features1

    def extract_and_match(self, left_img, right_img, confidence_threshold=0.5):
        """
        Extract features from left and right images, then match them.
        """
        # Preprocess images for matching
        left_tensor = self.preprocess_for_matching(left_img)
        right_tensor = self.preprocess_for_matching(right_img)
        # Extract features
        left_features = self.extract_features(left_tensor)
        right_features = self.extract_features(right_tensor)
        # Match features
        matches, scores, features_left = self.match_features(
                left_features,
                right_features,
                left_tensor.shape[1:],
                right_tensor.shape[1:],
                self.device,
            )
        points0 = left_features["keypoints"][0][matches[:, 0]]  # shape [K, 2]
        points1 = right_features["keypoints"][0][matches[:, 1]]  # shape [K, 2]
        # get descriptors
        descriptors0 = left_features["descriptors"][0][matches[:, 0]]  # shape [K, D]
        descriptors1 = right_features["descriptors"][0][matches[:, 1]]  # shape [K, D]
        # filter descriptors based on confidence threshold
        descriptors0 = descriptors0[scores > confidence_threshold]
        # matches0: indices in left, -1 if no match; matches1: indices in right, -1 if no match
        valid = scores > confidence_threshold
        mkpts_left = points0[valid].cpu().numpy()
        mkpts_right = points1[valid].cpu().numpy()
        return (mkpts_left, descriptors0.cpu().numpy()), (mkpts_left, mkpts_right, valid)

    def triangulate_rectified_points(self, left_points, right_points):
        """
        Triangulate 3D points using rectified stereo correspondences
        """
        # Use the rectified projection matrices P1 and P2
        points_4d = cv2.triangulatePoints(
            self.P1, self.P2,  # Use rectified projection matrices
            left_points.T, right_points.T
        )
        
        # Convert from homogeneous to 3D
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T
        
        return points_3d

    def stereo_triangulation(self, left_kpts, right_kpts, imu_pose=None):
        """
        Triangulate 3D points from stereo correspondences in rectified images
        """
        if len(left_kpts) == 0:
            return np.array([]), np.array([])
        
        # Triangulate in rectified camera coordinates
        points_3d_cam = self.triangulate_rectified_points(left_kpts, right_kpts)
        
        # Filter points based on depth and reprojection error
        valid_mask = self.filter_triangulated_points(points_3d_cam, left_kpts, right_kpts)
        points_3d_cam = points_3d_cam[valid_mask]
        # plot 3D points in plotly
        # if len(points_3d_cam) > 0:
        #     self.visualize_3d_points(points_3d_cam, 
        #                             title=f"Frame {self.current_frame_id} - {len(points_3d_cam)} points")
    
        # Transform to world coordinates if IMU pose is provided
        if imu_pose is not None:
            # Transform from camera to world frame
            # points_3d_cam is in the rectified left camera frame
            points_3d_world = self.transform_camera_to_world(points_3d_cam, imu_pose)
            return points_3d_world, valid_mask
        
        return points_3d_cam, valid_mask

    def filter_triangulated_points(self, points_3d, left_pts, right_pts):
        """Filter triangulated points based on quality metrics"""
        valid = np.ones(len(points_3d), dtype=bool)
        
        # Remove points with negative or very large depth
        valid &= (np.abs(points_3d[:, 2]) > 0.1)  # Minimum depth
        valid &= (np.abs(points_3d[:, 2]) < 10.0)  # Maximum depth

        # Make homogeneous points
        points_homog = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        
        # LEFT CAMERA: Use the FULL projection matrix P1
        projected_left = self.P1 @ points_homog.T
        projected_left = projected_left[:2] / projected_left[2:3]
        projected_left = projected_left.T
        
        # Compute left reprojection error
        reproj_error_left = np.linalg.norm(projected_left - left_pts, axis=1)
        
        # RIGHT CAMERA: Use the FULL projection matrix P2
        projected_right = self.P2 @ points_homog.T
        projected_right = projected_right[:2] / projected_right[2:3]
        projected_right = projected_right.T
        
        # Compute right reprojection error
        reproj_error_right = np.linalg.norm(projected_right - right_pts, axis=1)
        
        # Apply threshold to both errors
        threshold = 1.0
        valid &= (reproj_error_left < threshold)
        valid &= (reproj_error_right < threshold)
        
        if valid.sum() > 0:
            print(f"Filtered {(~valid).sum()}/{len(points_3d)} points with reprojection error and depth constraints.")
            print(f"Left reproj mean error: {reproj_error_left[valid].mean():.2f}px, "
                f"Right reproj mean error: {reproj_error_right[valid].mean():.2f}px")

        return valid

    def manage_landmarks(self, new_3d_points, track_ids, frame_id):
        """
        Manage landmark database - add new landmarks and update existing ones.
        
        Args:
            new_3d_points: Newly triangulated 3D points
            track_ids: Track IDs corresponding to the points
            frame_id: Current frame ID
            
        Returns:
            landmark_ids: IDs of landmarks (new and existing)
        """
        # 1. Check if tracks already have associated landmarks
        # 2. Create new landmarks for unassociated tracks
        # 3. Update landmark positions using bundle adjustment or averaging
        # 4. Add observations to landmark database
        landmark_ids = []
        for i, track_id in enumerate(track_ids):
            # Check if the track ID already has an associated landmark
            if track_id in self.feature_tracks:
                # Use the existing landmark ID
                landmark_id = self.feature_tracks[track_id]
            else:
                # Create a new landmark ID
                landmark_id = self.next_landmark_id
                self.next_landmark_id += 1
                
                # Add the new landmark to the database
                self.landmarks[landmark_id] = new_3d_points[i]
                
                # Associate the track ID with the new landmark ID
                self.feature_tracks[track_id] = landmark_id
            
            # Add the observation to the landmark's observation list
            self.landmark_observations[landmark_id].append((frame_id, track_id))
            
            # Append the landmark ID to the result list
            landmark_ids.append(landmark_id)
        
        return landmark_ids

    def create_observations_for_optimization(self, keypoints, track_ids, frame_id):
        """
        Create feature observation data structure for optimization backend.
        
        Args:
            keypoints: 2D keypoints in current frame
            track_ids: Track IDs of the keypoints
            frame_id: Current frame ID
            
        Returns:
            observations: List of observations in format expected by optimizer
                         [(landmark_id, frame_id, pixel_coords, uncertainty), ...]
        """
        # 1. Map track IDs to landmark IDs
        # 2. Create observation tuples
        # 3. Estimate pixel-level uncertainty
        # 4. Filter outliers and low-quality observations
        pass

    def get_landmarks_for_optimization(self):
        """
        Get landmark data for optimization backend.
        
        Returns:
            landmarks: Dictionary {landmark_id: (3d_position, uncertainty)}
            observations: List of all observations for current optimization window
        """
        # 1. Return current landmark estimates
        # 2. Include uncertainty/covariance estimates
        # 3. Filter landmarks with insufficient observations
        pass

    def update_landmarks_from_optimization(self, optimized_landmarks):
        """
        Update landmark positions after optimization.
        
        Args:
            optimized_landmarks: Dictionary {landmark_id: optimized_3d_position}
        """
        # 1. Update landmark positions
        # 2. Update landmark uncertainties if provided
        # 3. Remove outlier landmarks based on optimization results
        pass

    def track_features_temporal(self, prev_frame, curr_frame, prev_keypoints, R_prev_curr=None):
        """
        Track features between consecutive frames using optical flow.
        Uses forward-backward consistency check for robust outlier rejection.
        Optionally uses IMU-predicted rotation as initial guess for better convergence.
        
        Args:
            prev_frame: Previous frame (grayscale image).
            curr_frame: Current frame (grayscale image).
            prev_keypoints: Keypoints from the previous frame (Nx2 array).
            R_prev_curr: 3x3 rotation matrix from prev camera to curr camera (optional).
                         If provided, used to predict keypoint locations as initial guess.
            
        Returns:
            valid: Boolean mask of successfully tracked keypoints.
            curr_keypoints: Tracked keypoint positions in current frame (only valid ones).
        """
        # Convert keypoints to float32 for optical flow
        prev_keypoints = np.array(prev_keypoints, dtype=np.float32)
        
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        # Compute initial guess from IMU rotation if available
        prev_kp_cv = prev_keypoints.reshape(-1, 1, 2)
        initial_guess = None
        if R_prev_curr is not None:
            K = np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1]], dtype=np.float64)
            pts_h = np.hstack([prev_keypoints, np.ones((len(prev_keypoints), 1))])
            pts_norm = (np.linalg.inv(K) @ pts_h.T).T
            pts_rotated = (R_prev_curr @ pts_norm.T).T
            pts_proj = (K @ pts_rotated.T).T
            initial_guess = (pts_proj[:, :2] / pts_proj[:, 2:3]).astype(np.float32)
            initial_guess = np.ascontiguousarray(initial_guess.reshape(-1, 1, 2))
            lk_params["flags"] = cv2.OPTFLOW_USE_INITIAL_FLOW
        
        # Forward tracking: prev → curr
        curr_keypoints, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, prev_kp_cv, initial_guess, **lk_params
        )
        
        # Backward tracking: curr → prev (for consistency check)
        lk_params_bwd = dict(lk_params)
        lk_params_bwd.pop("flags", None)
        prev_keypoints_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_frame, prev_frame, curr_keypoints, None, **lk_params_bwd
        )
        
        # Reshape outputs from (N,1,2) to (N,2)
        curr_keypoints = curr_keypoints.reshape(-1, 2)
        prev_keypoints_back = prev_keypoints_back.reshape(-1, 2)
        
        # Forward-backward consistency check
        fb_dist = np.linalg.norm(prev_keypoints - prev_keypoints_back, axis=1)
        valid = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1) & (fb_dist < 3.0)
        
        n = len(prev_keypoints)
        mode = "IMU+K" if R_prev_curr is not None else "No IMU"
        print(f"Optical flow [{mode}]: valid={valid.sum()}/{n}", end="")
        if valid.sum() > 0:
            print(f"  mean_fb_err={fb_dist[valid].mean():.4f}px")
        else:
            print()
        return valid, curr_keypoints[valid]
        

    def stereo_matching_for_depth(self, left_features, right_img):
        """
        Find stereo correspondences for depth estimation.
        
        Args:
            left_features: Features detected in left image
            right_img: Right stereo image
            
        Returns:
            stereo_matches: Correspondences between left and right images
            disparities: Computed disparities for depth calculation
        """
        # 1. Extract features from right image
        # 2. Match left-right using epipolar constraints
        # 3. Compute disparities and filter outliers
        pass

    def landmark_culling(self, max_landmarks=5000):
        """
        Remove old or low-quality landmarks to maintain performance.
        
        Args:
            max_landmarks: Maximum number of landmarks to maintain
        """
        # 1. Score landmarks based on observation count, age, reprojection error
        # 2. Remove lowest-scoring landmarks
        # 3. Update observation database
        pass

    def get_map_for_visualization(self):
        """
        Get current map state for visualization.
        
        Returns:
            landmark_positions: Array of 3D landmark positions
            feature_tracks: Current active feature tracks
            map_statistics: Statistics about map quality
        """
        # 1. Return all landmark positions
        # 2. Include track information for visualization
        # 3. Provide map quality metrics
        pass

    def reproject_landmarks(self, landmarks, camera_pose, K):
        """
        Reproject 3D landmarks to image coordinates for validation.
        
        Args:
            landmarks: 3D landmark positions
            camera_pose: Camera pose (4x4 matrix)
            K: Camera intrinsics
            
        Returns:
            projected_points: 2D projected coordinates
            visibility_mask: Which landmarks are visible
        """
        # 1. Transform landmarks to camera coordinate system
        # 2. Project using camera model
        # 3. Check visibility (within image bounds, positive depth)
        pass

    def estimate_feature_uncertainty(self, keypoint, descriptor_quality=None):
        """
        Estimate uncertainty for feature observations.
        
        Args:
            keypoint: 2D keypoint location
            descriptor_quality: Quality metric from feature detector
            
        Returns:
            uncertainty: 2x2 covariance matrix for pixel coordinates
        """
        # 1. Base uncertainty on detector confidence
        # 2. Consider image gradients and corner strength
        # 3. Account for matching quality and track length
        pass

    # ==================== INTEGRATION FUNCTIONS ====================
    
    def initialize_from_stereo(self, left_img, right_img):
        """
        Initialize the mapping system from first stereo pair.
        
        Args:
            left_img: First left image
            right_img: First right image
            
        Returns:
            initial_landmarks: Initial set of triangulated landmarks
        """
        # 1. Extract features from both images
        # 2. Perform stereo matching
        # 3. Triangulate initial landmarks
        # 4. Initialize feature tracks
        pass

    def get_features_for_pose_estimation(self, min_matches=20):
        """
        Get feature correspondences for IMU pose refinement.
        
        Args:
            min_matches: Minimum number of matches required
            
        Returns:
            landmarks_3d: 3D landmark positions
            pixels_2d: Corresponding 2D observations
            track_ids: Track identifiers for outlier handling
        """
        # 1. Get landmarks visible in current frame
        # 2. Find 2D-3D correspondences
        # 3. Filter based on track quality and age
        pass

    def handle_optimization_outliers(self, outlier_track_ids):
        """
        Handle features marked as outliers by optimization.
        
        Args:
            outlier_track_ids: Track IDs marked as outliers
        """
        # 1. Remove outlier observations
        # 2. Terminate tracks if too many outliers
        # 3. Update landmark quality scores
        pass

    # ==================== UTILITY FUNCTIONS (Keep existing) ====================
    
    def preprocess_for_matching(self, img):
        """
        Preprocess image for feature matching.
        For 'disk': expects (H, W, 3) RGB, normalized to [0,1], shape (1, 3, H, W).
        For 'loftr' and 'superpoint': expects (H, W) grayscale, normalized to [0,1], shape (1, 1, H, W).
        """
        if self.matcher_type == "disk":
            if img.ndim == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None]  # (1, 3, H, W)
        elif self.matcher_type in ["loftr", "superpoint"]:
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img)[None, None]  # (1, 1, H, W)
        else:
            raise ValueError("Unknown matcher_type for preprocessing.")
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def set_alignment_matrix(self, R_align):
        """Set the alignment matrix for transforming poses to gravity-aligned frame."""
        self.R_align = R_align

    def visualize_3d_points(self, points_3d, camera_poses=None, title="3D Triangulated Points"):
        """
        Visualize triangulated 3D points in an interactive 3D scatter plot.
        
        Args:
            points_3d: Nx3 array of 3D points
            camera_poses: List of camera poses as 4x4 matrices (optional)
            title: Plot title
        """

        # Create figure
        fig = go.Figure()
        
        # Add points
        fig.add_trace(go.Scatter3d(
            x=points_3d[:, 0],
            z=-points_3d[:, 1],
            y=points_3d[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points_3d[:, 2],  # Color by depth
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Depth")
            ),
            name='3D Points'
        ))
        
        # Add coordinate axes at origin
        axis_length = 1.0
        axes = np.array([
            [0, 0, 0, axis_length, 0, 0],  # X-axis
            [0, 0, 0, 0, axis_length, 0],  # Y-axis
            [0, 0, 0, 0, 0, axis_length]   # Z-axis
        ])
        
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        for i, (axis, color, label) in enumerate(zip(axes, colors, labels)):
            fig.add_trace(go.Scatter3d(
                x=[axis[0], axis[3]],
                y=[axis[1], axis[4]],
                z=[axis[2], axis[5]],
                mode='lines',
                line=dict(color=color, width=5),
                name=f'{label}-axis'
            ))
        
        # Add camera positions if available
        if camera_poses is not None:
            camera_positions = np.array([pose[:3, 3] for pose in camera_poses])
            fig.add_trace(go.Scatter3d(
                x=camera_positions[:, 0],
                y=camera_positions[:, 1],
                z=camera_positions[:, 2],
                mode='markers+lines',
                marker=dict(
                    size=5,
                    color='red',
                    symbol='square'
                ),
                line=dict(color='red', width=2),
                name='Camera Path'
            ))
        
        # Set figure layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'  # Keep the true scale
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Show the figure
        fig.show()
        
    # --------------- HNSW CORE ---------------
    def _init_hnsw(self):
        if self.hnsw_inited:
            return
        self.hnsw_index = hnswlib.Index(space=self.hnsw_space, dim=self.hnsw_dim)
        self.hnsw_index.init_index(max_elements=self.hnsw_max_elements,
                                   ef_construction=200,
                                   M=16)
        self.hnsw_index.set_ef(64)
        self.hnsw_inited = True

    def _update_landmark_descriptor(self, lid, desc):
        desc = desc.astype(np.float32)
        if lid not in self.landmark_desc:
            self.landmark_desc[lid] = desc.copy()
            self.landmark_desc_counts[lid] = 1
        else:
            c = self.landmark_desc_counts[lid]
            self.landmark_desc[lid] = (self.landmark_desc[lid] * c + desc) / (c + 1)
            self.landmark_desc_counts[lid] = c + 1

    def _buffer_new_landmark(self, lid):
        # Add to index buffer (only when descriptor exists)
        if lid in self.landmark_desc:
            self.hnsw_new_buffer.append((lid, self.landmark_desc[lid]))

    def _flush_hnsw_buffer(self, batch_size=256):
        if not self.hnsw_new_buffer:
            return
        self._init_hnsw()
        batch = self.hnsw_new_buffer[:batch_size]
        self.hnsw_new_buffer = self.hnsw_new_buffer[batch_size:]
        if not batch:
            return
        ids = np.array([b[0] for b in batch], dtype=np.int64)
        vecs = np.vstack([b[1] for b in batch]).astype(np.float32)
        self.hnsw_index.add_items(vecs, ids)
        self.hnsw_elements += len(ids)

    def query_similar_landmarks(self, query_descs, k=50, exclude_ids=None, min_frame_gap=0):
        if (not self.hnsw_inited) or self.hnsw_elements == 0 or len(query_descs) == 0:
            return []
        query_descs = query_descs.astype(np.float32)
        labels, dists = self.hnsw_index.knn_query(query_descs, k=min(k, self.hnsw_index.element_count))
        # Aggregate votes by landmark id, filtering out current and recent landmarks
        exclude_ids = exclude_ids or set()
        current_frame = self.current_frame_id
        counts = {}
        for row in labels:
            for lid in row:
                if lid in exclude_ids:
                    continue
                # Skip landmarks last seen within the temporal gap
                last_seen = self.landmark_last_frame.get(lid, current_frame)
                if current_frame - last_seen < min_frame_gap:
                    continue
                counts[lid] = counts.get(lid, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def query_similar_frames(self, query_descs, k_landmarks=50, top_frames=5,
                             exclude_ids=None, min_frame_gap=0):
        lm_candidates = self.query_similar_landmarks(
            query_descs, k=k_landmarks, exclude_ids=exclude_ids, min_frame_gap=min_frame_gap
        )
        if not lm_candidates:
            return []
        # Vote using the last frame a landmark was seen (not first frame)
        # This gives more meaningful temporal locality for loop closure
        frame_votes = {}
        for lid, score in lm_candidates:
            last_frame = self.landmark_last_frame.get(lid, None)
            if last_frame is not None:
                frame_votes[last_frame] = frame_votes.get(last_frame, 0) + score
        ranked = sorted(frame_votes.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_frames]

    def _build_kdtree(self, curr_kp, prev_kp, radius=2.0):
        """
        Build a KD-Tree for fast nearest neighbor search of SuperPoint keypoints.
        Args:
            features_left: Extracted features from the left image
            keypoints: (N,2) array of keypoints to map
            radius: Search radius for nearest neighbor"""
        tree_sp = cKDTree(prev_kp)
        dists, idxs = tree_sp.query(curr_kp, distance_upper_bound=radius)
        valid = idxs < len(prev_kp)
        return idxs, dists, valid
        

    # --------------- Descriptor Mapping ---------------
    def _map_keypoints_to_descriptors(self, features_left, keypoints, radius=2.0,
                                      base_image=None, show=False):
        """
        Map (N,2) keypoints to nearest extracted SuperPoint keypoints -> descriptors.
        Minimal visualization of:
          - previous frame keypoints (red)
          - current input keypoints (green)
          - chosen SuperPoint keypoints (blue)
          - line current -> chosen SP keypoint
        Returns:
            descriptors: (N,D)
            sp_indices: (N,) index of chosen SuperPoint kp (=-1 if none)
        """
        if features_left is None or len(keypoints) == 0:
            return np.empty((0, self.hnsw_dim), dtype=np.float32), np.empty((0,), dtype=int)

        keypoints = np.asarray(keypoints, dtype=np.float32)
        # sp_kp = features_left["keypoints"][0].cpu().numpy()
        # sp_desc = features_left["descriptors"][0].cpu().numpy()
        sp_kp = features_left[0]
        sp_desc = features_left[1]

        idxs, dists, valid = self._build_kdtree(sp_kp, keypoints)

        descriptors = sp_desc[idxs[valid]]

        print("Descriptor assoc mean dist:",
              float(dists[valid].mean()) if valid.any() else None,
              "valid", valid.sum(), "/", len(keypoints))

        if show and base_image is not None:
            if base_image.ndim == 2:
                vis = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
            else:
                vis = base_image.copy()

            # Previous keypoints (all) in red
            if self.prev_keypoints is not None:
                for p in self.prev_keypoints:
                    cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

            # Current keypoints (all) in green
            for p in sp_kp:
                cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 200, 0), -1)

            # Chosen SuperPoint keypoints (matched) in blue + lines
            matched_sp = sp_kp[idxs[valid]]
            matched_curr = keypoints[valid]
            for c, s in zip(matched_curr, matched_sp):
                cv2.line(vis, (int(c[0]), int(c[1])), (int(s[0]), int(s[1])), (255, 255, 0), 1)
                cv2.circle(vis, (int(s[0]), int(s[1])), 3, (255, 0, 0), 1)

            cv2.putText(vis,
                        f"Prev:{0 if self.prev_keypoints is None else len(self.prev_keypoints)} "
                        f"Curr:{len(keypoints)} Matched:{valid.sum()}",
                        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Prev (red) / Curr (green) / SP matched (blue)", vis)
            cv2.waitKey(1)

        # idxs where invalid -> -1
        sp_indices = idxs.copy()
        sp_indices[~valid] = -1
        return descriptors, sp_indices

    # --------------- Integrate into manage_landmarks ---------------
    def manage_landmarks(self, new_3d_points, track_ids, frame_id):
        landmark_ids = []
        for i, track_id in enumerate(track_ids):
            if track_id in self.feature_tracks:
                landmark_id = self.feature_tracks[track_id]
            else:
                landmark_id = self.next_landmark_id
                self.next_landmark_id += 1
                self.landmarks[landmark_id] = new_3d_points[i]
                self.feature_tracks[track_id] = landmark_id
                self.landmark_to_frame[landmark_id] = frame_id
            self.landmark_observations[landmark_id].append((frame_id, track_id))
            # Track last seen
            self.landmark_last_frame[landmark_id] = frame_id
            landmark_ids.append(landmark_id)
        # Register frame->landmarks
        self.frame_landmarks[frame_id] = set(landmark_ids)
        return np.asarray(landmark_ids)

    # --------------- Hook in process_stereo_frame ---------------
    def process_stereo_frame2(self, left_img, right_img, imu_pose=None, R_prev_curr=None):
        """Main stereo processing pipeline with temporal tracking and landmark management.
        
        Args:
            left_img: Left stereo image (grayscale).
            right_img: Right stereo image (grayscale).
            imu_pose: Optional world pose for transforming points to world frame.
            R_prev_curr: Optional 3x3 rotation from previous to current camera frame
                         (from IMU preintegration). Improves optical flow tracking.
        """
        observations = []
        tracked_ids = []
        
        # ========== 1. STEREO FEATURE EXTRACTION & MATCHING ==========
        # Extract SuperPoint features and match between left/right images
        features_left, matches = self.extract_and_match(left_img, right_img)
        if len(features_left) == 0:
            return [], []
        left_points = matches[0]; right_points = matches[1]
        
        # Filter matches using epipolar constraints (rectified stereo)
        valid_matches = self.stereo_match_rectified(left_points, right_points)
        if valid_matches.sum() == 0:
            return [], []
        left_points_valid = left_points[valid_matches]
        right_points_valid = right_points[valid_matches]
        valid_descs = features_left[1][valid_matches]  # Descriptors aligned with valid matches

        # ========== 2. TEMPORAL TRACKING (OPTICAL FLOW) ==========
        # Track features from previous frame to current frame using KLT
        if self.prev_keypoints is not None:
            valid_prev_mask, curr_tracked_keypoints = self.track_features_temporal(
                self.prev_frame, left_img, self.prev_keypoints, R_prev_curr=R_prev_curr
            )
            if self.prev_track_ids is not None:
                tracked_ids_prev = self.prev_track_ids[valid_prev_mask]
        else:
            curr_tracked_keypoints = np.empty((0, 2), dtype=np.float32)
            tracked_ids_prev = np.empty((0,), dtype=int)

        # ========== 3. STEREO TRIANGULATION ==========
        # Triangulate 3D points from valid stereo matches and filter by reprojection error
        points_3d, valid_mask = self.stereo_triangulation(left_points_valid, right_points_valid, imu_pose)
        if len(points_3d) == 0:
            return [], []
        valid_curr_keypts = left_points_valid[valid_mask]
        valid_descs = valid_descs[valid_mask]

        # ========== 4. TRACK ID ASSOCIATION ==========
        # Associate tracked keypoints with triangulated keypoints to preserve track IDs
        assoc_ids = np.full(len(valid_curr_keypts), -1, dtype=int)
        if len(curr_tracked_keypoints) and len(valid_curr_keypts):
            # Find nearest neighbors between tracked and triangulated keypoints
            idxs, dists, valid = self._build_kdtree(curr_tracked_keypoints, valid_curr_keypts, radius = 5)
            print("Prev to Current temporal tracked, to current stereo triangulated. mean dist:", float(dists[valid].mean()) if valid.any() else None,
              "valid", valid.sum(), "/", len(valid_curr_keypts))
            
            good = idxs < len(valid_curr_keypts)
            valid_landmarks_track = idxs[good]
            # Assign existing track IDs to associated triangulated keypoints
            for t_idx, tri_i in zip(np.where(good)[0], valid_landmarks_track):
                if assoc_ids[tri_i] == -1:  # first assignment wins
                    assoc_ids[tri_i] = tracked_ids_prev[t_idx]
            final_descs = valid_descs[valid_landmarks_track]
        else:
            final_descs = valid_descs
    
        # Assign new track IDs to unassociated triangulated keypoints
        need_new = assoc_ids == -1
        n_new = need_new.sum()
        if n_new:
            new_ids = np.arange(self.next_track_id, self.next_track_id + n_new, dtype=int)
            self.next_track_id += n_new
            assoc_ids[need_new] = new_ids
        track_ids = assoc_ids

        # ========== 5. LANDMARK MANAGEMENT ==========
        # Create new landmarks or update existing ones using track IDs
        landmark_ids = self.manage_landmarks(points_3d, track_ids, self.current_frame_id)

        # ========== 6. DESCRIPTOR & HNSW UPDATE ==========
        # Update landmark descriptors for loop closure detection (periodic)
        # Only index temporally tracked landmarks (verified multi-frame consistency)
        if self.prev_keypoints is not None and self.current_frame_id % 5 == 0:
            landmark_relevant = landmark_ids[valid_landmarks_track]
            for lid, d in zip(landmark_relevant, final_descs):
                if d.sum() != 0.0:
                    self._update_landmark_descriptor(int(lid), d)
                    self._buffer_new_landmark(int(lid))
            self._flush_hnsw_buffer(batch_size=512)

        # ========== 7. LOOP CLOSURE DETECTION ==========
        # Query similar frames using current descriptors (periodic)
        # Exclude current landmarks and require a temporal gap
        loop_candidates = []
        if self.current_frame_id % 5 == 0 and self.prev_keypoints is not None:
            current_lm_set = set(int(lid) for lid in landmark_ids)
            min_gap = 9  # Minimum frame gap to avoid trivial matches
            loop_candidates = self.query_similar_landmarks(
                valid_descs, k=20, exclude_ids=current_lm_set, min_frame_gap=min_gap
            )
            candidates = self.query_similar_frames(
                valid_descs, k_landmarks=20, top_frames=3,
                exclude_ids=current_lm_set, min_frame_gap=min_gap
            )
            self.prev_descriptors_for_lc = valid_descs  # Store for main loop LC queries
            self.lc_matched_frames = candidates  # Store matched frame candidates
            print(f"** Loop closure (frame[{self.current_frame_id}]) candidates:", candidates, "**")
        else:
            self.lc_matched_frames = []

        # ========== 8. OBSERVATIONS OUTPUT FOR OPTIMIZER ==========
        # Format: observations = [(landmark_id, frame_id, uv_pixel_coords), ...]
        # new_landmarks_3d = {landmark_id: np.array([x,y,z])} for newly created landmarks
        observations = []
        new_landmarks_3d = {}
        for idx, lid in enumerate(landmark_ids):
            uv = valid_curr_keypts[idx]
            observations.append((int(lid), self.current_frame_id, uv.astype(np.float32)))
            if need_new[idx]:
                new_landmarks_3d[int(lid)] = points_3d[idx]

        # ========== 9. STATE UPDATE ==========
        # Save current frame data for next iteration
        self.prev_keypoints = valid_curr_keypts
        self.prev_frame = left_img.copy()
        self.prev_track_ids = track_ids
        self.current_frame_id += 1

        print(f"[Frame {self.current_frame_id-1}] Obs:{len(observations)} NewLM:{len(new_landmarks_3d)}")
        return observations, new_landmarks_3d, loop_candidates
        


if __name__ == "__main__":
    # Example usage
    # use data_manager to get intrinsics and baseline
    from data_manager import DataManager
    import os
    data_dir = "f:/Code/exercise_10/data/MH_01_easy/mav0"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")

    # --- Data Loading and Calibration ---
    data_manager = DataManager(data_dir)
    data_manager.load_data()

    # Camera calibration
    K1, dist_coeffs1, new_K1 = data_manager.load_camera_calib("cam0")
    K2, dist_coeffs2, new_K2 = data_manager.load_camera_calib("cam1")
    baseline = data_manager.get_baseline(K1, K2)
    print(f"Baseline between cam0 and cam1: {baseline} meters")

    vfeature = vFeature(
        matcher_type="superpoint",
        baseline=baseline,
        intrinsics=K1,
        dist_coeffs=dist_coeffs1,
        T_cam1_cam0=data_manager.T_cam1_cam0,  # Camera 1 to Camera 0 transformation
        device="cpu"  # Use CPU for processing
    )

    vfeature.initialize()
    first_run = True
    # get the first stereo pair from data manager
    frame_step = 10
    for id, left_img, right_img in data_manager.iter_stereo_frames(step=frame_step):
        if first_run:
            # Initialize the mapping system with the first stereo pair
            vfeature.P1 = data_manager.P1  # Rectified projection matrix for cam0
            vfeature.P2 = data_manager.P2  # Rectified projection matrix for cam1
            first_run = False

        # Process the first stereo frames
        observations, new_landmarks = vfeature.process_stereo_frame2(left_img, right_img)


    # # Process stereo frame
    # observations, new_landmarks = vfeature.process_stereo_frame(left_img, right_img)
    #     # Print results
    # print("Observations:", observations)
    # print("New Landmarks:", new_landmarks)
