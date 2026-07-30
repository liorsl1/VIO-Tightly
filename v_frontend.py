import numpy as np
import cv2
import torch
# from lightglue import LightGlue, SuperPoint
import hnswlib
from scipy.spatial import cKDTree

from vio_utils import skew, pixels_to_bearings, project_points, unit_rows


class vFeature:
    """Stereo frontend: KLT tracking, on-demand detection, and descriptor retrieval.

    Landmark identity is carried by KLT alone — a dropped track is not recoverable,
    and the same physical point re-detected later becomes a new landmark.
    """

    def __init__(self, matcher_type="superpoint", device="cpu", baseline=None,
                 intrinsics=None, dist_coeffs=None, T_cam1_cam0=None):
        # --- Existing attributes ---
        self.matcher_type = matcher_type  # "superpoint" or "xfeat"
        self.device = device
        self.baseline = baseline
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        self.T_cam1_cam0 = T_cam1_cam0  # Camera 1 to Camera 0 transformation
        self.P1 = None  # Rectified projection matrix for Camera 0
        self.P2 = None  # Rectified projection matrix for Camera 1

        # --- New tightly-coupled attributes ---
        self.feature_tracks = {}  # {track_id: landmark_id}
        self.landmarks = {}  # {landmark_id: 3D_position}
        self.current_frame_id = 0
        self.next_track_id = 0
        self.next_landmark_id = 0

        # --- Feature tracking state ---
        self.prev_keypoints = None
        self.prev_frame = None
        self.prev_track_ids = None  # Track IDs from the previous frame
        # Detection trigger: spatial coverage, not a raw count. SVO buckets the image into
        # a grid and treats empty cells as what needs filling; a count threshold instead
        # depends on resolution, texture density and how clustered the tracks happen to be,
        # so it does not transfer between sequences or cameras.
        self.detect_cell_size = 25       # px per coverage cell (SVO uses ~30)
        self.min_occupied_fraction = 0.18  # detect when occupied cells fall below this
        self.MIN_TRACKED_DIST = 0  # Min per-feature pixel displacement to keep a track

        # --- HNSW retrieval structures ---
        self.hnsw_index = None
        self.hnsw_dim = 256 if matcher_type == "superpoint" else 64  # SuperPoint=256D, XFeat=64D
        self.hnsw_max_elements = 200000
        self.hnsw_elements = 0
        self.hnsw_inited = False
        self.hnsw_space = 'l2'       # or 'cosine'
        self.hnsw_new_buffer = []    # (landmark_id, descriptor)
        self.lc_matched_frames = []  # [(frame_idx, vote_count), ...] from query_similar_frames
        self.lc_min_cosine = 0.9     # similarity floor for retrieval, tunable in one place
        self.landmark_desc = {}      # landmark_id -> running mean descriptor (np.float32, shape (D,))
        self.landmark_desc_counts = {}  # landmark_id -> num updates
        self.landmark_last_frame = {}   # landmark_id -> last seen frame

        self.initialize()

    def initialize(self):
        """Load the detector/matcher selected by matcher_type.

        Returns:
            None.
        """
        if self.matcher_type == "xfeat":
            import torch as _torch
            print("Initializing XFeat...")
            self.xfeat = _torch.hub.load(
                'verlab/accelerated_features', 'XFeat',
                pretrained=True, top_k=1024, trust_repo=True
            ).eval().to(self.device)
            # XFeat caches its own device (defaults to CUDA if available) and uses it
            # to move input tensors in preprocess_tensor. Override it to match self.device,
            # otherwise inputs land on CUDA while weights are on CPU -> dtype/device mismatch.
            self.xfeat.dev = _torch.device(self.device)
            print(f"XFeat loaded on {self.device} (64D descriptors)")
        else:
            print("Initializing SuperPoint and LightGlue...")
            self.superpoint = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
            self.lg_matcher = LightGlue(features="superpoint").eval().to(self.device)


    def stereo_match_rectified(self, cam0_points, cam1_points):
        """Validate stereo correspondences on rectified images.

        Rectification makes the epipolar geometry axis-aligned, so the full essential
        matrix test collapses to two scalar checks: disparity must be positive and
        bounded, and the vertical offset must be near zero.

        Args:
            cam0_points: (N, 2) left-image pixels.
            cam1_points: (N, 2) right-image pixels, index-aligned with cam0_points.

        Returns:
            (N,) boolean mask of geometrically valid matches.
        """
        if len(cam0_points) == 0:
            return np.zeros(0, dtype=bool)

        disparity = cam0_points[:, 0] - cam1_points[:, 0]
        vertical_error = np.abs(cam0_points[:, 1] - cam1_points[:, 1])
        return np.logical_and.reduce([
            disparity > 0.1,       # Positive disparity
            disparity < 110,       # Reasonable disparity limit
            vertical_error < 1.0,  # Tight epipolar constraint
        ])

    # ==================== CORE TIGHTLY-COUPLED FUNCTIONS ====================

    def preprocess_for_matching(self, img):
        """Convert a raw image to the tensor layout the active extractor expects.

        Args:
            img: (H, W) grayscale or (H, W, 3) RGB uint8 image.

        Returns:
            Float tensor on self.device — (1, 3, H, W) for XFeat, (1, 1, H, W) for
            SuperPoint.
        """
        if self.matcher_type == "xfeat":
            # XFeat expects (B, 3, H, W) RGB float [0,1]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None]
        elif self.matcher_type == "superpoint":
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img)[None, None]
        else:
            raise ValueError("Unknown matcher_type for preprocessing.")
        return img_tensor.to(self.device)

    def extract_features(self, image):
        """Detect keypoints and descriptors with the active extractor.

        Args:
            image: Preprocessed image tensor from preprocess_for_matching.

        Returns:
            Dict with 'keypoints', 'descriptors' and 'scores', each a single-element
            list (SuperPoint's layout, which XFeat output is adapted to).
        """
        if self.matcher_type == "xfeat":
            with torch.inference_mode():
                output = self.xfeat.detectAndCompute(image, top_k=2048)[0]
            return {
                "keypoints": [output["keypoints"]],      # list of (N, 2)
                "descriptors": [output["descriptors"]],  # list of (N, 64)
                "scores": [output["scores"]],            # list of (N,)
            }
        if self.matcher_type == "superpoint":
            with torch.inference_mode():
                return self.superpoint({"image": image})
        raise NotImplementedError(
            "Feature extraction not implemented for this matcher_type."
        )

    def match_features(self, features1, features2):
        """Match two feature sets, returning LightGlue-style index pairs.

        Args:
            features1: Feature dict for the first image.
            features2: Feature dict for the second image.

        Returns:
            (matches01, scores) — (K, 2) index pairs and their per-match scores.
        """
        if self.matcher_type == "xfeat":
            # Use mutual nearest neighbor matching on descriptors
            desc0 = features1["descriptors"][0]  # (N, 64)
            desc1 = features2["descriptors"][0]  # (M, 64)
            # Compute similarity and find mutual nearest neighbors
            # Could be improved with ZSSD or ZNCC
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
            return matches01, scores
        if self.matcher_type == "superpoint":
            with torch.inference_mode():
                matches = self.lg_matcher({"image0": features1, "image1": features2})
            return matches["matches"][0], matches["scores"][0]
        raise NotImplementedError("Matching not implemented for this matcher_type.")

    def extract_and_match(self, left_img, right_img, confidence_threshold=0.5):
        """Full stereo detect-and-match, used only to bootstrap the first frames.

        Args:
            left_img: Left rectified image.
            right_img: Right rectified image.
            confidence_threshold: Min match score; overridden to 0.95 for XFeat,
                whose cosine-similarity scores live on a different scale.

        Returns:
            ((left_pixels, left_descriptors), (left_pixels, right_pixels)) for the
            matches that passed the threshold.
        """
        if self.matcher_type == "xfeat":
            confidence_threshold = 0.95

        left_features = self.extract_features(self.preprocess_for_matching(left_img))
        right_features = self.extract_features(self.preprocess_for_matching(right_img))
        matches, scores = self.match_features(left_features, right_features)

        valid = scores > confidence_threshold
        points0 = left_features["keypoints"][0][matches[:, 0]]
        points1 = right_features["keypoints"][0][matches[:, 1]]
        descriptors0 = left_features["descriptors"][0][matches[:, 0]][valid]
        mkpts_left = points0[valid].cpu().numpy()
        mkpts_right = points1[valid].cpu().numpy()
        return (mkpts_left, descriptors0.cpu().numpy()), (mkpts_left, mkpts_right)

    def stereo_triangulation(self, left_kpts, right_kpts):
        """Triangulate rectified stereo correspondences into camera-frame points.

        Args:
            left_kpts: (N, 2) left-image pixels.
            right_kpts: (N, 2) right-image pixels, index-aligned.

        Returns:
            (points_3d, valid_mask) — points already filtered by valid_mask, which is
            indexed against the input arrays so callers can subset them in step.
        """
        if len(left_kpts) == 0:
            return np.array([]), np.array([])

        points_4d = cv2.triangulatePoints(self.P1, self.P2, left_kpts.T, right_kpts.T)
        points_3d_cam = (points_4d[:3] / points_4d[3]).T

        valid_mask = self.filter_triangulated_points(points_3d_cam, left_kpts, right_kpts)
        return points_3d_cam[valid_mask], valid_mask

    def filter_triangulated_points(self, points_3d, left_pts, right_pts):
        """Reject triangulations by depth band and stereo re-projection error.

        Args:
            points_3d: (N, 3) triangulated points in the rectified left camera frame.
            left_pts: (N, 2) left-image pixels that produced them.
            right_pts: (N, 2) right-image pixels that produced them.

        Returns:
            (N,) boolean mask of accepted points.
        """
        valid = np.ones(len(points_3d), dtype=bool)
        valid &= (np.abs(points_3d[:, 2]) > 0.1)   # Minimum depth
        valid &= (np.abs(points_3d[:, 2]) < 12.0)  # Maximum depth

        reproj_error_left = np.linalg.norm(
            project_points(points_3d, self.P1) - left_pts, axis=1
        )
        reproj_error_right = np.linalg.norm(
            project_points(points_3d, self.P2) - right_pts, axis=1
        )
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

    # ==================== GEOMETRIC OUTLIER REJECTION ====================

    def geometric_outlier_rejection_2pt(
        self, prev_pts, curr_pts, R_prev_curr, K,
        ransac_iters=100, ransac_threshold=1.0, min_inliers=8
    ):
        """2-point RANSAC for translation direction when rotation is known from IMU.

        The epipolar constraint p2^T [t]x R p1 = 0 rearranges by the scalar triple
        product to t . (p2 x R p1) = 0, so with R fixed each correspondence gives one
        linear constraint on t and two suffice — t is the null space of the 2x3 system.
        Candidates are scored by Sampson distance, a first-order approximation of
        geometric re-projection error that is tight in the small-error regime.

        Args:
            prev_pts: (N, 2) pixels in the previous frame.
            curr_pts: (N, 2) pixels in the current frame.
            R_prev_curr: (3, 3) rotation from previous to current camera frame.
            K: (3, 3) camera intrinsic matrix.
            ransac_iters: Number of RANSAC iterations.
            ransac_threshold: Inlier threshold in pixels.
            min_inliers: Below this the model is rejected and nothing is filtered.

        Returns:
            (N,) boolean inlier mask.
        """
        N = len(prev_pts)
        if N < 2:
            return np.ones(N, dtype=bool)

        prev_norm = pixels_to_bearings(prev_pts, K)
        curr_norm = pixels_to_bearings(curr_pts, K)
        R = R_prev_curr
        rotated_prev = (R @ prev_norm.T).T
        # q_i = p2_i x (R p1_i) — the vector each correspondence constrains t against
        q = np.cross(curr_norm, rotated_prev)

        # A 1px error in pixel space is ~1/f in normalized coordinates.
        norm_threshold_sq = (ransac_threshold / min(K[0, 0], K[1, 1])) ** 2

        best_inlier_count = 0
        best_inlier_mask = np.zeros(N, dtype=bool)
        rng = np.random.default_rng()

        for _ in range(ransac_iters):
            idx = rng.choice(N, size=2, replace=False)
            _, S, Vt = np.linalg.svd(q[idx])
            t_candidate = Vt[-1]  # smallest singular vector = null space of the 2x3 system

            # Nearly parallel constraints leave the null space ill-defined.
            if len(S) >= 2 and S[-1] > 0.1 * S[0]:
                continue

            E = skew(t_candidate) @ R
            Ep1 = (E @ prev_norm.T).T
            ETp2 = (E.T @ curr_norm.T).T
            epipolar_err = np.sum(curr_norm * Ep1, axis=1)  # p2^T E p1

            denom = np.maximum(
                Ep1[:, 0] ** 2 + Ep1[:, 1] ** 2 + ETp2[:, 0] ** 2 + ETp2[:, 1] ** 2,
                1e-12,
            )
            inlier_mask = (epipolar_err ** 2) / denom < norm_threshold_sq
            n_inliers = inlier_mask.sum()

            if n_inliers > best_inlier_count:
                best_inlier_count = n_inliers
                best_inlier_mask = inlier_mask.copy()

        n_outliers = N - best_inlier_count
        if best_inlier_count < min_inliers:
            # RANSAC failed to find a good model — don't reject anything,
            # let the backend's robust cost function handle it
            print(f"  [2pt-RANSAC] FAILED: only {best_inlier_count}/{N} inliers "
                  f"(need {min_inliers}). Keeping all features.")
            return np.ones(N, dtype=bool)

        print(f"  [2pt-RANSAC] {best_inlier_count}/{N} inliers, "
              f"{n_outliers} outliers rejected (threshold={ransac_threshold:.1f}px)")
        return best_inlier_mask

    def geometric_outlier_rejection_5pt(self, prev_pts, curr_pts, K, ransac_threshold=1.0):
        """5-point RANSAC fallback when no IMU rotation prior is available.

        Nister's 5-point algorithm (via cv2.findEssentialMat) solves for the full
        essential matrix, so it needs no prior but must search 5 DOF instead of 2.

        Args:
            prev_pts: (N, 2) pixels in the previous frame.
            curr_pts: (N, 2) pixels in the current frame.
            K: (3, 3) camera intrinsic matrix.
            ransac_threshold: Inlier threshold in pixels.

        Returns:
            (N,) boolean inlier mask.
        """
        N = len(prev_pts)
        if N < 5:
            return np.ones(N, dtype=bool)

        _E, mask = cv2.findEssentialMat(
            prev_pts, curr_pts, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ransac_threshold,
        )
        if mask is None:
            return np.ones(N, dtype=bool)

        inlier_mask = mask.ravel().astype(bool)
        n_inliers = inlier_mask.sum()
        print(f"  [5pt-RANSAC] {n_inliers}/{N} inliers, "
              f"{N - n_inliers} outliers rejected (threshold={ransac_threshold:.1f}px)")
        return inlier_mask

    def track_features_temporal(self, prev_frame, curr_frame, prev_keypoints, R_prev_curr=None):
        """Track features prev->curr with KLT, rejecting forward-backward mismatches.

        The IMU rotation, when available, is projected through K to predict where each
        keypoint lands and seeds the flow — a better starting point than the previous
        position whenever the camera rotated between frames.

        Args:
            prev_frame: Previous grayscale image.
            curr_frame: Current grayscale image.
            prev_keypoints: (N, 2) keypoints in the previous frame.
            R_prev_curr: Optional (3, 3) rotation from previous to current camera frame.

        Returns:
            (valid, curr_keypoints, median_displacement) — mask over the input, the
            tracked positions of the valid subset, and their median pixel motion.
        """
        prev_keypoints = np.array(prev_keypoints, dtype=np.float32)
        median_displacement = 0.0
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        prev_kp_cv = prev_keypoints.reshape(-1, 1, 2)
        initial_guess = None
        if R_prev_curr is not None:
            K = self.P1[:, :3]  # rectified left-camera intrinsics
            pts_rotated = (R_prev_curr @ pixels_to_bearings(prev_keypoints, K).T).T
            pts_proj = (K @ pts_rotated.T).T
            initial_guess = (pts_proj[:, :2] / pts_proj[:, 2:3]).astype(np.float32)
            initial_guess = np.ascontiguousarray(initial_guess.reshape(-1, 1, 2))
            lk_params["flags"] = cv2.OPTFLOW_USE_INITIAL_FLOW

        curr_keypoints, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, prev_kp_cv, initial_guess, **lk_params
        )

        # Backward pass for the consistency check; the seed must not be reused here.
        lk_params_bwd = dict(lk_params)
        lk_params_bwd.pop("flags", None)
        prev_keypoints_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_frame, prev_frame, curr_keypoints, None, **lk_params_bwd
        )

        curr_keypoints = curr_keypoints.reshape(-1, 2)
        prev_keypoints_back = prev_keypoints_back.reshape(-1, 2)

        fb_dist = np.linalg.norm(prev_keypoints - prev_keypoints_back, axis=1)
        valid = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1) & (fb_dist < 1)

        displacements = np.linalg.norm(curr_keypoints - prev_keypoints, axis=1)
        valid = valid & (displacements > self.MIN_TRACKED_DIST)
        if valid.sum() > 0:
            median_displacement = float(np.median(displacements[valid]))

        mode = "IMU+P1" if R_prev_curr is not None else "No IMU"
        print(f"Optical flow [{mode}]: valid={valid.sum()}/{len(prev_keypoints)}", end="")
        if valid.sum() > 0:
            print(f"  mean_fb_err={fb_dist[valid].mean():.4f}px  median_disp={median_displacement:.2f}px")
        else:
            print()
        return valid, curr_keypoints[valid], median_displacement

    # --------------- HNSW CORE ---------------
    def _init_hnsw(self):
        """Lazily build the HNSW index on first use.

        Returns:
            None.
        """
        if self.hnsw_inited:
            return
        self.hnsw_index = hnswlib.Index(space=self.hnsw_space, dim=self.hnsw_dim)
        self.hnsw_index.init_index(max_elements=self.hnsw_max_elements,
                                   ef_construction=200,
                                   M=16)
        self.hnsw_index.set_ef(64)
        self.hnsw_inited = True

    def _update_landmark_descriptor(self, lid, desc):
        """Fold an observation's descriptor into a landmark's running mean.

        Args:
            lid: Landmark id.
            desc: (D,) descriptor for this observation.

        Returns:
            None.
        """
        desc = desc.astype(np.float32)
        if lid not in self.landmark_desc:
            self.landmark_desc[lid] = desc.copy()
            self.landmark_desc_counts[lid] = 1
        else:
            c = self.landmark_desc_counts[lid]
            self.landmark_desc[lid] = (self.landmark_desc[lid] * c + desc) / (c + 1)
            self.landmark_desc_counts[lid] = c + 1

    def _buffer_new_landmark(self, lid):
        """Queue a landmark's descriptor for the next HNSW flush, unit-normalized.

        The running mean of unit descriptors has norm < 1, shrinking as views are
        averaged, which makes L2 distances incomparable across landmarks and breaks the
        cos = 1 - d^2/2 conversion. Indexing the normalized direction restores both. The
        mean itself is left un-normalized so further averaging stays correct.

        Args:
            lid: Landmark id; ignored if it has no descriptor yet.

        Returns:
            None.
        """
        if lid in self.landmark_desc:
            normalized = unit_rows(self.landmark_desc[lid]).astype(np.float32)
            self.hnsw_new_buffer.append((lid, normalized))

    def _index_descriptors(self, landmark_ids, descriptors) -> int:
        """Fold descriptors into their landmarks' running means and queue them for HNSW.

        Args:
            landmark_ids: Iterable of landmark ids.
            descriptors: (N, D) descriptors aligned with landmark_ids.

        Returns:
            Number of descriptors actually indexed (all-zero rows are skipped).
        """
        n_indexed = 0
        for lid, desc in zip(landmark_ids, descriptors):
            if desc.sum() == 0.0:
                continue
            self._update_landmark_descriptor(int(lid), desc)
            self._buffer_new_landmark(int(lid))
            n_indexed += 1
        return n_indexed

    def _flush_hnsw_buffer(self, batch_size=256):
        """Insert up to batch_size queued descriptors into the HNSW index.

        Args:
            batch_size: Maximum descriptors inserted per call.

        Returns:
            None.
        """
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

    def query_similar_landmarks(self, query_descs, k=50, exclude_ids=None,
                                min_frame_gap=0, min_cosine=None):
        """Retrieve previously-seen landmarks by descriptor similarity, ranked by score.

        A vote is weighted two ways rather than counted. By similarity: hnswlib's 'l2'
        space returns SQUARED distance and both sides are unit-norm, so cos = 1 - d^2/2
        is exact. And by inverse frequency, as in DBoW2: a landmark retrieved by many of
        this frame's descriptors is generically-textured, not a revisit, so it is
        discounted by log1p(M / df) instead of accumulating votes.

        Args:
            query_descs: (M, D) descriptors of the current frame's landmarks.
            k: Neighbours retrieved per query, and cap on returned candidates.
            exclude_ids: Landmark ids to ignore (typically those seen right now).
            min_frame_gap: Minimum frames since a landmark was last seen.
            min_cosine: Similarity floor; defaults to self.lc_min_cosine.

        Returns:
            [(landmark_id, score), ...] sorted by score descending.
        """
        if (not self.hnsw_inited) or self.hnsw_elements == 0 or len(query_descs) == 0:
            return []
        floor = self.lc_min_cosine if min_cosine is None else min_cosine
        query_descs = unit_rows(np.asarray(query_descs, dtype=np.float32)).astype(np.float32)
        labels, sq_dists = self.hnsw_index.knn_query(
            query_descs, k=min(k, self.hnsw_index.element_count)
        )
        exclude_ids = exclude_ids or set()
        current_frame = self.current_frame_id

        # Pass 1: keep neighbours clearing the gates and the similarity floor, counting
        # how many distinct query descriptors retrieved each landmark.
        kept, doc_freq = [], {}
        for row_labels, row_sq in zip(labels, sq_dists):
            for lid, sq in zip(row_labels, row_sq):
                lid = int(lid)
                if lid in exclude_ids:
                    continue
                last_seen = self.landmark_last_frame.get(lid, current_frame)
                if current_frame - last_seen < min_frame_gap:
                    continue
                cosine = 1.0 - float(sq) / 2.0
                if cosine < floor:
                    continue
                kept.append((lid, cosine))
                doc_freq[lid] = doc_freq.get(lid, 0) + 1
        if not kept:
            return []

        # Pass 2: score. The ramp spreads [floor, 1] over [0, 1] so the floor is a soft
        # boundary rather than making every survivor count the same. log1p keeps the IDF
        # weight strictly positive, so a single-descriptor query can still rank.
        n_queries = len(query_descs)
        span = max(1.0 - floor, 1e-6)
        scores = {}
        for lid, cosine in kept:
            weight = ((cosine - floor) / span) * np.log1p(n_queries / doc_freq[lid])
            scores[lid] = scores.get(lid, 0.0) + weight
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def query_similar_frames(self, query_descs, k_landmarks=50, top_frames=5,
                             exclude_ids=None, min_frame_gap=0):
        """Aggregate landmark votes into frame-level place-recognition candidates.

        Votes are attributed to the frame where each landmark was LAST seen, which
        localizes the revisit in time better than its first sighting.

        Args:
            query_descs: (M, D) descriptors of the current frame's landmarks.
            k_landmarks: Neighbours retrieved per query descriptor.
            top_frames: Number of candidate frames to return.
            exclude_ids: Landmark ids to ignore.
            min_frame_gap: Minimum frames since a landmark was last seen.

        Returns:
            [(frame_idx, score), ...] sorted by score descending.
        """
        lm_candidates = self.query_similar_landmarks(
            query_descs, k=k_landmarks, exclude_ids=exclude_ids, min_frame_gap=min_frame_gap
        )
        if not lm_candidates:
            return []
        frame_votes = {}
        for lid, score in lm_candidates:
            last_frame = self.landmark_last_frame.get(lid, None)
            if last_frame is not None:
                frame_votes[last_frame] = frame_votes.get(last_frame, 0) + score
        return sorted(frame_votes.items(), key=lambda x: x[1], reverse=True)[:top_frames]

    def coverage_fraction(self, keypoints, img_shape):
        """Fraction of detection-grid cells holding at least one keypoint.

        Measures spatial spread rather than quantity: a thousand features crowded on one
        textured wall constrain the pose far worse than two hundred spread over the frame,
        and only the second reading is comparable across resolutions and scenes.

        Args:
            keypoints: (N, 2) pixel coordinates.
            img_shape: Image shape; only the leading (height, width) is used.

        Returns:
            (fraction, n_cells) — occupied fraction in [0, 1] and the grid size.
        """
        height, width = img_shape[:2]
        n_cols = max(1, int(np.ceil(width / self.detect_cell_size)))
        n_rows = max(1, int(np.ceil(height / self.detect_cell_size)))
        n_cells = n_rows * n_cols
        if len(keypoints) == 0:
            return 0.0, n_cells
        kp = np.asarray(keypoints, dtype=float)
        cols = np.clip(kp[:, 0] // self.detect_cell_size, 0, n_cols - 1).astype(int)
        rows = np.clip(kp[:, 1] // self.detect_cell_size, 0, n_rows - 1).astype(int)
        occupied = np.unique(rows * n_cols + cols).size
        return occupied / n_cells, n_cells

    def _build_kdtree(self, query_kp, target_kp, radius=2.0):
        """Nearest-neighbour lookup from query keypoints into a target keypoint set.

        Args:
            query_kp: (N, 2) keypoints to look up.
            target_kp: (M, 2) keypoints to search within.
            radius: Maximum match distance in pixels.

        Returns:
            (idxs, dists, valid) — target index per query, distance, and a mask that is
            False where no target fell inside the radius.
        """
        dists, idxs = cKDTree(target_kp).query(query_kp, distance_upper_bound=radius)
        return idxs, dists, idxs < len(target_kp)

    def manage_landmarks(self, new_3d_points, track_ids, frame_id):
        """Register newly triangulated points as landmarks and record observations.

        Args:
            new_3d_points: (N, 3) triangulated positions in the camera frame.
            track_ids: (N,) track ids the points belong to.
            frame_id: Frame the observations come from.

        Returns:
            (N,) array of landmark ids, new or existing.
        """
        landmark_ids = []
        for i, track_id in enumerate(track_ids):
            if track_id in self.feature_tracks:
                landmark_id = self.feature_tracks[track_id]
            else:
                landmark_id = self.next_landmark_id
                self.next_landmark_id += 1
                self.landmarks[landmark_id] = new_3d_points[i]
                self.feature_tracks[track_id] = landmark_id
            self.landmark_last_frame[landmark_id] = frame_id
            landmark_ids.append(landmark_id)
        return np.asarray(landmark_ids, dtype=int)

    def process_stereo_frame2(self, left_img, right_img, R_prev_curr=None):
        """Per-frame frontend: KLT-track the existing pool, detect only when it thins.

        Tracking is cheap and runs every frame; detection, stereo matching and
        triangulation only fire when grid coverage drops. Landmark identity is
        carried by KLT alone, so a dropped track cannot be recovered later.

        Args:
            left_img: Left rectified grayscale image.
            right_img: Right rectified grayscale image.
            R_prev_curr: Optional (3, 3) IMU-predicted rotation to seed optical flow.

        Returns:
            (observations, new_landmarks_3d, loop_candidates, median_displacement).
        """
        # ========== 1. TEMPORAL TRACKING (KLT) ==========
        tracked_keypoints = np.empty((0, 2), dtype=np.float32)
        tracked_landmark_ids = np.empty((0,), dtype=int)
        median_displacement = 0.0

        if self.prev_keypoints is not None and len(self.prev_keypoints) > 0:
            valid_mask, curr_tracked, median_displacement = self.track_features_temporal(
                self.prev_frame, left_img, self.prev_keypoints, R_prev_curr=R_prev_curr
            )

            # Epipolar RANSAC on top of the FB check: it catches geometrically
            # inconsistent tracks that pass photometrically (moving objects,
            # repetitive texture, KLT drift along low-contrast edges).
            if len(curr_tracked) >= 8:
                # prev_keypoints corresponding to the valid KLT tracks
                prev_matched = self.prev_keypoints[valid_mask]
                K = self.P1[:3, :3]  # rectified intrinsic matrix

                if R_prev_curr is not None:
                    ransac_inliers = self.geometric_outlier_rejection_2pt(
                        prev_matched, curr_tracked, R_prev_curr, K,
                        ransac_iters=100, ransac_threshold=1.0,
                    )
                else:
                    ransac_inliers = self.geometric_outlier_rejection_5pt(
                        prev_matched, curr_tracked, K,
                        ransac_threshold=1.0,
                    )

                # Apply RANSAC mask on top of the KLT-valid set
                curr_tracked = curr_tracked[ransac_inliers]
                # Update valid_mask: indices that were True now get further filtered
                valid_indices = np.where(valid_mask)[0]
                valid_mask = np.zeros_like(valid_mask)
                valid_mask[valid_indices[ransac_inliers]] = True

            tracked_keypoints = curr_tracked
            tracked_landmark_ids = self.prev_track_ids[valid_mask]
        
        n_tracked = len(tracked_keypoints)
        coverage, n_cells = self.coverage_fraction(tracked_keypoints, left_img.shape)
        need_new_features = coverage < self.min_occupied_fraction

        # ========== 2. DETECT & STEREO MATCH NEW FEATURES (only when needed) ==========
        new_keypoints = np.empty((0, 2), dtype=np.float32)
        new_descs = np.empty((0, self.hnsw_dim), dtype=np.float32)
        new_landmark_ids = np.empty((0,), dtype=int)
        new_landmarks_3d = {}
        full_initial_match = 5

        if need_new_features:
            is_first_frame = (self.current_frame_id < full_initial_match)
            print(f"  Coverage {coverage:.2f} < {self.min_occupied_fraction:.2f}"
                  f" ({n_tracked} tracks over {n_cells} cells), detecting new features"
                  f" ({'INIT: full stereo match' if is_first_frame else 'KLT stereo'})...")

            if is_first_frame:
                # Bootstrap: full detect-and-match on both images, no pool to track yet.
                features_left, matches = self.extract_and_match(left_img, right_img)
                if len(features_left[0]) > 0:
                    left_points, right_points = matches
                    descs_all = features_left[1]
                else:
                    left_points = np.empty((0, 2), dtype=np.float32)
                    right_points = np.empty((0, 2), dtype=np.float32)
                    descs_all = np.empty((0, self.hnsw_dim), dtype=np.float32)
            else:
                # Steady state: detect on the left image only, stereo-match by KLT.
                left_tensor = self.preprocess_for_matching(left_img)
                sp_feats = self.extract_features(left_tensor)
                sp_kp = sp_feats["keypoints"][0].cpu().numpy().astype(np.float32)
                sp_desc = sp_feats["descriptors"][0].cpu().numpy()

                if len(sp_kp) == 0:
                    left_points = np.empty((0, 2), dtype=np.float32)
                    right_points = np.empty((0, 2), dtype=np.float32)
                    descs_all = np.empty((0, self.hnsw_dim), dtype=np.float32)
                else:
                    # KLT from left → right image (stereo matching via optical flow)
                    lk_params_stereo = dict(
                        winSize=(17, 5),  # Wide horizontal, narrow vertical (rectified)
                        maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01),
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
                if valid_stereo.sum() > 0:
                    left_valid = left_points[valid_stereo]
                    right_valid = right_points[valid_stereo]
                    descs_valid = descs_all[valid_stereo]

                    new_3d_filtered, tri_mask = self.stereo_triangulation(
                        left_valid, right_valid
                    )

                    if len(new_3d_filtered) > 0:
                        new_keypoints = left_valid[tri_mask]
                        new_descs = descs_valid[tri_mask]

                        n_new = len(new_keypoints)
                        new_ids = np.arange(
                            self.next_track_id, self.next_track_id + n_new, dtype=int
                        )
                        self.next_track_id += n_new
                        new_landmark_ids = self.manage_landmarks(
                            new_3d_filtered, new_ids, self.current_frame_id
                        )

                        for idx, lid in enumerate(new_landmark_ids):
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
        # Tracked landmarks need descriptors too, not just new ones, or retrieval can
        # only ever match a landmark's first appearance. Re-extraction is the cost, so
        # this runs every 5th frame.
        if self.current_frame_id % 5 == 0:
            n_tracked_indexed = 0
            if n_tracked > 0:
                left_tensor = self.preprocess_for_matching(left_img)
                sp_feats = self.extract_features(left_tensor)
                sp_kp = sp_feats["keypoints"][0].cpu().numpy()
                sp_desc = sp_feats["descriptors"][0].cpu().numpy()

                # Tracked keypoints come from KLT, so they carry no descriptor —
                # borrow one from the nearest fresh detection.
                idxs, _dists, valid_sp = self._build_kdtree(
                    tracked_keypoints, sp_kp, radius=3.0
                )
                n_tracked_indexed = self._index_descriptors(
                    tracked_landmark_ids[valid_sp], sp_desc[idxs[valid_sp]]
                )

            n_new_indexed = self._index_descriptors(new_landmark_ids, new_descs)
            if n_tracked_indexed or n_new_indexed:
                self._flush_hnsw_buffer(batch_size=512)
                print(f"  HNSW update: indexed {n_tracked_indexed} tracked "
                      f"+ {n_new_indexed} new descriptors")

        # ========== 6. LOOP CLOSURE DETECTION (periodic) ==========
        loop_candidates = []
        self.lc_matched_frames = []
        if self.current_frame_id % 5 == 0 and n_tracked > 0:
            all_lm_ids = np.concatenate([tracked_landmark_ids, new_landmark_ids])
            current_lm_set = set(int(lid) for lid in all_lm_ids)
            lc_desc_list = [
                self.landmark_desc[int(lid)] for lid in all_lm_ids
                if int(lid) in self.landmark_desc
            ]
            if lc_desc_list:
                lc_descs = np.vstack(lc_desc_list)
                min_gap = 9
                loop_candidates = self.query_similar_landmarks(
                    lc_descs, k=70, exclude_ids=current_lm_set, min_frame_gap=min_gap
                )
                self.lc_matched_frames = self.query_similar_frames(
                    lc_descs, k_landmarks=70, top_frames=3,
                    exclude_ids=current_lm_set, min_frame_gap=min_gap
                )
                print(f"** Loop closure (frame[{self.current_frame_id}]) queried "
                      f"{len(lc_descs)} descs, candidates:", self.lc_matched_frames, "**")

        # ========== 7. STATE UPDATE ==========
        # Merge tracked + new keypoints for next frame's KLT
        all_keypoints = np.concatenate([tracked_keypoints, new_keypoints], axis=0)
        all_track_ids = np.concatenate([tracked_landmark_ids, new_landmark_ids])

        self.prev_keypoints = all_keypoints if len(all_keypoints) > 0 else None
        self.prev_frame = left_img.copy()
        self.prev_track_ids = all_track_ids if len(all_track_ids) > 0 else None
        self.current_frame_id += 1

        cov_after, _ = self.coverage_fraction(all_keypoints, left_img.shape)
        print(f"[Frame {self.current_frame_id-1}] Tracked:{n_tracked} New:{len(new_keypoints)}"
              f" Obs:{len(observations)} NewLM:{len(new_landmarks_3d)}"
              f" Cov:{coverage:.2f}->{cov_after:.2f} MedianDisp:{median_displacement:.2f}px")
        return observations, new_landmarks_3d, loop_candidates, median_displacement
