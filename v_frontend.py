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
        self.matcher_type = matcher_type  # "superpoint" or "xfeat"
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
        self.hnsw_dim = 256 if matcher_type == "superpoint" else 64  # SuperPoint=256D, XFeat=64D
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
        """Initialize feature extractors and matchers."""
        
        if self.matcher_type == "xfeat":
            import torch as _torch
            print("Initializing XFeat...")
            self.xfeat = _torch.hub.load(
                'verlab/accelerated_features', 'XFeat',
                pretrained=True, top_k=1024, trust_repo=True
            ).eval().to(self.device)
            print(f"XFeat loaded on {self.device} (64D descriptors)")
        else:
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
            vertical_error < 2.0  # Tight epipolar constraint
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

    def preprocess_for_matching(self, img):
        """
        Preprocess image for feature matching.
        For 'xfeat': expects (H, W) grayscale or (H, W, 3) RGB, output (1, 3, H, W).
        For 'disk': expects (H, W, 3) RGB, normalized to [0,1], shape (1, 3, H, W).
        For 'loftr' and 'superpoint': expects (H, W) grayscale, normalized to [0,1], shape (1, 1, H, W).
        """
        if self.matcher_type == "xfeat":
            # XFeat expects (B, 3, H, W) RGB float [0,1]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None]  # (1, 3, H, W)
        elif self.matcher_type == "disk":
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
        if self.matcher_type == "xfeat":
            return self._extract_features_xfeat(image)
        elif self.matcher_type == "disk":
            return self._extract_features_disk(image, num_features)
        elif self.matcher_type == "superpoint":
            return self._extract_features_superpoint(image)
        else:
            raise NotImplementedError(
                "Feature extraction not implemented for this matcher_type."
            )

    def _extract_features_xfeat(self, image):
        """Extract features using XFeat. Returns dict compatible with SuperPoint format."""
        with torch.inference_mode():
            output = self.xfeat.detectAndCompute(image, top_k=2048)[0]
        # Return in same format as SuperPoint for compatibility
        return {
            "keypoints": [output["keypoints"]],      # list of (N, 2)
            "descriptors": [output["descriptors"]],  # list of (N, 64)
            "scores": [output["scores"]],            # list of (N,)
        }

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
        """Compute tentative matches between features."""
        if self.matcher_type == "xfeat":
            # Use mutual nearest neighbor matching on descriptors
            desc0 = features1["descriptors"][0]  # (N, 64)
            desc1 = features2["descriptors"][0]  # (M, 64)
            # Compute similarity and find mutual nearest neighbors
            with torch.inference_mode():
                sim = desc0 @ desc1.T  # (N, M)
                nn01 = sim.argmax(dim=1)  # best match in feat2 for each feat1
                nn10 = sim.argmax(dim=0)  # best match in feat1 for each feat2
                ids0 = torch.arange(len(desc0), device=sim.device)
                mutual = nn10[nn01] == ids0  # mutual nearest neighbor check
                # Build match indices like LightGlue format
                valid_ids0 = ids0[mutual]
                valid_ids1 = nn01[mutual]
                matches01 = torch.stack([valid_ids0, valid_ids1], dim=1)
                # Score = cosine similarity of matched pairs
                scores = sim[valid_ids0, valid_ids1]
            return matches01, scores, features1
        elif self.matcher_type == "superpoint":
            with torch.inference_mode():
                matches = self.lg_matcher({"image0": features1, "image1": features2})
            matches01 = matches["matches"][0]
            dists = matches["scores"][0]
            return matches01, dists, features1

    def extract_and_match(self, left_img, right_img, confidence_threshold=0.5):
        """
        Extract features from left and right images, then match them.
        """
        # For XFeat, use lower confidence threshold (cosine sim range differs)
        if self.matcher_type == "xfeat":
            confidence_threshold = 0.9

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
        else:
            print(f"* All points filtered out! Left reproj error: {reproj_error_left.mean():.2f}px, "
                  f"Right reproj error: {reproj_error_right.mean():.2f}px *")

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
        


    def get_map_for_visualization(self):
        """
        Future : create another 3D representation for the visualization
        """
        pass


    # ==================== UTILITY FUNCTIONS (Keep existing) ====================
    
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
    MIN_TRACKED_FEATURES = 200  # Trigger new detection when tracks drop below this

    def process_stereo_frame2(self, left_img, right_img, imu_pose=None, R_prev_curr=None):
        """Main stereo processing pipeline — KLT-first, detect only when needed.

        Per-frame (cheap, ~5-15ms):
            1. KLT track existing features prev→curr
            2. Output 2D observations for tracked landmarks

        When tracked count < MIN_TRACKED_FEATURES (expensive, occasional):
            3. SuperPoint detection on left image
            4. LightGlue stereo match with right image (new features only)
            5. Triangulate new landmarks
            6. Add to tracking pool

        Args:
            left_img: Left stereo image (grayscale).
            right_img: Right stereo image (grayscale).
            imu_pose: Optional world pose for transforming points to world frame.
            R_prev_curr: Optional 3x3 rotation from previous to current camera frame
                         (from IMU preintegration). Improves optical flow tracking.
        """
        # ========== 1. TEMPORAL TRACKING (KLT) ==========
        # Track existing features from previous frame to current frame
        tracked_keypoints = np.empty((0, 2), dtype=np.float32)
        tracked_landmark_ids = np.empty((0,), dtype=int)

        if self.prev_keypoints is not None and len(self.prev_keypoints) > 0:
            valid_mask, curr_tracked = self.track_features_temporal(
                self.prev_frame, left_img, self.prev_keypoints, R_prev_curr=R_prev_curr
            )
            tracked_keypoints = curr_tracked
            tracked_landmark_ids = self.prev_track_ids[valid_mask]
        
        n_tracked = len(tracked_keypoints)
        need_new_features = n_tracked < self.MIN_TRACKED_FEATURES

        # ========== 2. DETECT & STEREO MATCH NEW FEATURES (only when needed) ==========
        new_keypoints = np.empty((0, 2), dtype=np.float32)
        new_points_3d = np.empty((0, 3), dtype=np.float64)
        new_descs = np.empty((0, self.hnsw_dim), dtype=np.float32)
        new_landmark_ids = np.empty((0,), dtype=int)
        new_landmarks_3d = {}

        if need_new_features:
            is_first_frame = (self.current_frame_id == 0)
            print(f"  Tracked {n_tracked} < {self.MIN_TRACKED_FEATURES}, detecting new features"
                  f" ({'INIT: full stereo match' if is_first_frame else 'KLT stereo'})...")

            if is_first_frame:
                # --- FIRST FRAME: Full SuperPoint + LightGlue stereo matching ---
                features_left, matches = self.extract_and_match(left_img, right_img)
                if len(features_left[0]) > 0:
                    left_points = matches[0]
                    right_points = matches[1]
                    descs_all = features_left[1]
                else:
                    left_points = np.empty((0, 2), dtype=np.float32)
                    right_points = np.empty((0, 2), dtype=np.float32)
                    descs_all = np.empty((0, self.hnsw_dim), dtype=np.float32)
            else:
                # --- SUBSEQUENT FRAMES: SuperPoint left only + KLT left→right ---
                left_tensor = self.preprocess_for_matching(left_img)
                sp_feats = self.extract_features(left_tensor)
                sp_kp = sp_feats["keypoints"][0].cpu().numpy().astype(np.float32)
                sp_desc = sp_feats["descriptors"][0].cpu().numpy()

                # Exclude detections too close to already-tracked features early
                if n_tracked > 0:
                    tree_tracked = cKDTree(tracked_keypoints)
                    d_tracked, _ = tree_tracked.query(sp_kp)
                    far_mask = d_tracked > 9.0
                    sp_kp = sp_kp[far_mask]
                    sp_desc = sp_desc[far_mask]

                if len(sp_kp) == 0:
                    left_points = np.empty((0, 2), dtype=np.float32)
                    right_points = np.empty((0, 2), dtype=np.float32)
                    descs_all = np.empty((0, self.hnsw_dim), dtype=np.float32)
                else:
                    # KLT from left → right image (stereo matching via optical flow)
                    lk_params_stereo = dict(
                        winSize=(21, 5),  # Wide horizontal, narrow vertical (rectified)
                        maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                    )
                    sp_kp_cv = sp_kp.reshape(-1, 1, 2)
                    right_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        left_img, right_img, sp_kp_cv, None, **lk_params_stereo
                    )
                    right_pts = right_pts.reshape(-1, 2)
                    status = status.flatten() == 1

                    left_points = sp_kp[status]
                    right_points = right_pts[status]
                    descs_all = sp_desc[status]

            # --- Common path: epipolar filter + triangulate ---
            if len(left_points) > 0:
                valid_stereo = self.stereo_match_rectified(left_points, right_points)
                if hasattr(valid_stereo, 'sum') and valid_stereo.sum() > 0:
                    left_valid = left_points[valid_stereo]
                    right_valid = right_points[valid_stereo]
                    descs_valid = descs_all[valid_stereo]

                    # Triangulate
                    points_3d, tri_mask = self.stereo_triangulation(
                        left_valid, right_valid, imu_pose
                    )

                    if len(points_3d) > 0:
                        new_kpts_candidate = left_valid[tri_mask]
                        new_descs_candidate = descs_valid[tri_mask]

                        # Exclude points too close to already-tracked features
                        # (first frame has no tracked, subsequent already filtered above)
                        if n_tracked > 0 and is_first_frame:
                            tree = cKDTree(tracked_keypoints)
                            dists_t, _ = tree.query(new_kpts_candidate)
                            far_enough = dists_t > 9.0
                        else:
                            far_enough = np.ones(len(new_kpts_candidate), dtype=bool)

                        new_kpts_filtered = new_kpts_candidate[far_enough]
                        new_3d_filtered = points_3d[far_enough]
                        new_descs_filtered = new_descs_candidate[far_enough]

                        if len(new_kpts_filtered) > 0:
                            # Assign new track IDs
                            n_new = len(new_kpts_filtered)
                            new_ids = np.arange(
                                self.next_track_id, self.next_track_id + n_new, dtype=int
                            )
                            self.next_track_id += n_new

                            # Create landmarks
                            lm_ids = self.manage_landmarks(
                                new_3d_filtered, new_ids, self.current_frame_id
                            )

                            new_keypoints = new_kpts_filtered
                            new_points_3d = new_3d_filtered
                            new_descs = new_descs_filtered
                            new_landmark_ids = lm_ids

                            # Record new landmarks for optimizer
                            for idx, lid in enumerate(lm_ids):
                                new_landmarks_3d[int(lid)] = new_3d_filtered[idx]

                            print(f"  Added {n_new} new features (total will be {n_tracked + n_new})")

        # ========== 3. UPDATE LANDMARK LAST-SEEN FOR TRACKED ==========
        for lid in tracked_landmark_ids:
            self.landmark_last_frame[int(lid)] = self.current_frame_id

        # ========== 4. OBSERVATIONS OUTPUT FOR OPTIMIZER ==========
        # All tracked features produce observations (no re-triangulation needed)
        observations = []
        for idx, lid in enumerate(tracked_landmark_ids):
            uv = tracked_keypoints[idx]
            observations.append((int(lid), self.current_frame_id, uv.astype(np.float32)))

        # New features also produce observations
        for idx, lid in enumerate(new_landmark_ids):
            uv = new_keypoints[idx]
            observations.append((int(lid), self.current_frame_id, uv.astype(np.float32)))

        # ========== 5. DESCRIPTOR & HNSW UPDATE (periodic) ==========
        # Periodically extract descriptors for TRACKED landmarks too (not just new ones)
        # This enables loop closure to find revisited landmarks
        if self.current_frame_id % 5 == 0 and n_tracked > 0:
            # Run SuperPoint extraction only (no LightGlue stereo match — cheap ~30ms)
            left_tensor = self.preprocess_for_matching(left_img)
            sp_feats = self.extract_features(left_tensor)
            sp_kp = sp_feats["keypoints"][0].cpu().numpy()
            sp_desc = sp_feats["descriptors"][0].cpu().numpy()

            # Map tracked keypoints to nearest SuperPoint detections for descriptors
            idxs, dists, valid_sp = self._build_kdtree(tracked_keypoints, sp_kp, radius=3.0)
            matched_lids = tracked_landmark_ids[valid_sp]
            matched_descs = sp_desc[idxs[valid_sp]]

            for lid, d in zip(matched_lids, matched_descs):
                if d.sum() != 0.0:
                    self._update_landmark_descriptor(int(lid), d)
                    self._buffer_new_landmark(int(lid))

            # Also index new features if we have them
            if len(new_descs) > 0:
                for lid, d in zip(new_landmark_ids, new_descs):
                    if d.sum() != 0.0:
                        self._update_landmark_descriptor(int(lid), d)
                        self._buffer_new_landmark(int(lid))

            self._flush_hnsw_buffer(batch_size=512)
            print(f"  HNSW update: indexed {valid_sp.sum()} tracked + {len(new_descs)} new descriptors")
        elif self.current_frame_id % 5 == 0 and len(new_descs) > 0:
            # Only new features available (first frame edge case)
            for lid, d in zip(new_landmark_ids, new_descs):
                if d.sum() != 0.0:
                    self._update_landmark_descriptor(int(lid), d)
                    self._buffer_new_landmark(int(lid))
            self._flush_hnsw_buffer(batch_size=512)

        # ========== 6. LOOP CLOSURE DETECTION (periodic) ==========
        loop_candidates = []
        if self.current_frame_id % 5 == 0 and n_tracked > 0:
            # Use tracked landmark descriptors from the HNSW index (just updated above)
            # Gather descriptors for current frame's landmarks from the running mean
            all_lm_ids = np.concatenate([tracked_landmark_ids, new_landmark_ids]) if len(new_landmark_ids) > 0 else tracked_landmark_ids
            current_lm_set = set(int(lid) for lid in all_lm_ids)

            # Collect descriptors we have for current landmarks
            lc_desc_list = []
            for lid in all_lm_ids:
                lid_int = int(lid)
                if lid_int in self.landmark_desc:
                    lc_desc_list.append(self.landmark_desc[lid_int])
            
            if len(lc_desc_list) > 0:
                lc_descs = np.vstack(lc_desc_list)
                min_gap = 9
                loop_candidates = self.query_similar_landmarks(
                    lc_descs, k=35, exclude_ids=current_lm_set, min_frame_gap=min_gap
                )
                candidates = self.query_similar_frames(
                    lc_descs, k_landmarks=20, top_frames=3,
                    exclude_ids=current_lm_set, min_frame_gap=min_gap
                )
                self.lc_matched_frames = candidates
                print(f"** Loop closure (frame[{self.current_frame_id}]) queried {len(lc_descs)} descs, candidates:", candidates, "**")
            else:
                self.lc_matched_frames = []
        else:
            self.lc_matched_frames = []

        # ========== 7. STATE UPDATE ==========
        # Merge tracked + new keypoints for next frame's KLT
        all_keypoints = np.concatenate([tracked_keypoints, new_keypoints], axis=0) if len(new_keypoints) > 0 else tracked_keypoints
        all_track_ids = np.concatenate([tracked_landmark_ids, new_landmark_ids]) if len(new_landmark_ids) > 0 else tracked_landmark_ids

        self.prev_keypoints = all_keypoints if len(all_keypoints) > 0 else None
        self.prev_frame = left_img.copy()
        self.prev_track_ids = all_track_ids if len(all_track_ids) > 0 else None
        self.current_frame_id += 1

        print(f"[Frame {self.current_frame_id-1}] Tracked:{n_tracked} New:{len(new_keypoints)} Obs:{len(observations)} NewLM:{len(new_landmarks_3d)}")
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
