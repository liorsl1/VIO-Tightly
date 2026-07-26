"""
Visualize promoted landmarks as a 3D point cloud with depth uncertainty ellipsoids.

Reads the promoted_depth_log from the optimizer and renders each landmark as:
  - A point colored by inlier probability (green=confident, red=marginal)
  - An ellipsoid scaled by the depth uncertainty (σ_d in meters)

Colors of uncertainties are relative - red are still in the accepted threshold.
Usage:
    After running main_threaded.py, this script loads the saved log and visualizes.
    Or: import and call visualize_promoted_depths(optimizer.promoted_depth_log) inline.

Requires: open3d (pip install open3d)
"""
import numpy as np
import pickle
import sys
import os

def compute_depth_ellipsoid(pt_world, df_mu, df_sigma2, viewing_dir=None):
    """Compute ellipsoid radii from depth filter state.
    
    The uncertainty is anisotropic: large along the viewing direction (depth),
    small laterally (bearing is well-constrained by pixel measurement).
    
    Returns:
        radii: (3,) array [lateral, lateral, depth_sigma_m]
    """
    # Depth estimate and sigma in meters
    depth_est = 1.0 / max(abs(df_mu), 1e-10)
    sigma_invdepth = np.sqrt(df_sigma2)
    # σ_d = σ_ρ * d² (propagation from inverse depth to depth)
    sigma_depth_m = sigma_invdepth * depth_est * depth_est
    
    # Lateral uncertainty (from pixel noise ~0.5px at focal=460, depth d):
    # σ_lateral ≈ σ_pixel * depth / focal ≈ 0.5 * d / 460
    sigma_lateral = 0.001 * depth_est  # ~1mm per meter (tight from stereo)
    
    return np.array([sigma_lateral, sigma_lateral, sigma_depth_m])


def visualize_promoted_depths(promoted_depth_log, title="Depth Filter: Promoted Landmarks"):
    """Visualize promoted landmarks as point cloud + uncertainty ellipsoids.
    
    Args:
        promoted_depth_log: list of (landmark_id, pt3_world, df_mu, df_sigma2, df_a, df_n)
    """
    try:
        import open3d as o3d

    # except ImportError:
        print("open3d not installed. Install with: pip install open3d")
        print("Falling back to matplotlib 3D scatter...")
        _visualize_matplotlib(promoted_depth_log, title)
        return
    except Exception as e:
        print(f"Failed to import open3d: {e}")
        print("Falling back to matplotlib 3D scatter...")
        _visualize_matplotlib(promoted_depth_log, title)
        return

    if not promoted_depth_log:
        print("No promoted landmarks to visualize.")
        return

    geometries = []
    
    # Extract data
    points = []
    colors = []
    sigmas = []
    
    for lm_id, pt3_world, df_mu, df_sigma2, df_a, df_n in promoted_depth_log:
        points.append(pt3_world)
        
        # Color by inlier probability: green (a=1) → red (a=0.5)
        r = 1.0 - (df_a - 0.5) * 2.0  # 1.0 at a=0.5, 0.0 at a=1.0
        g = (df_a - 0.5) * 2.0         # 0.0 at a=0.5, 1.0 at a=1.0
        colors.append([max(0, min(1, r)), max(0, min(1, g)), 0.2])
        
        radii = compute_depth_ellipsoid(pt3_world, df_mu, df_sigma2)
        sigmas.append(radii)
    
    points = np.array(points)
    colors = np.array(colors)
    sigmas = np.array(sigmas)
    
    print(f"Visualizing {len(points)} promoted landmarks")
    print(f"  Depth range: {1.0/np.max([x[2] for x in promoted_depth_log]):.2f}m - "
          f"{1.0/np.min([max(x[2], 1e-10) for x in promoted_depth_log]):.2f}m")
    depth_sigmas_m = sigmas[:, 2]
    print(f"  Depth σ range: {depth_sigmas_m.min():.4f}m - {depth_sigmas_m.max():.4f}m")
    print(f"  Median depth σ: {np.median(depth_sigmas_m):.4f}m")
    
    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd)
    
    # Add ellipsoids for a subset (every Nth to avoid clutter)
    n_ellipsoids = min(200, len(points))
    indices = np.linspace(0, len(points) - 1, n_ellipsoids, dtype=int)
    
    for idx in indices:
        pt = points[idx]
        sigma = sigmas[idx]
        color = colors[idx]
        
        # Scale ellipsoid by 2σ for visibility
        scale = sigma * 2.0
        # Minimum visible size
        scale = np.maximum(scale, 0.01)
        
        # Create sphere and scale to ellipsoid
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=8)
        mesh.scale(1.0, center=mesh.get_center())
        
        # Apply anisotropic scaling
        vertices = np.asarray(mesh.vertices)
        vertices[:, 0] *= scale[0]
        vertices[:, 1] *= scale[1]
        vertices[:, 2] *= scale[2]
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        # Translate to landmark position
        mesh.translate(pt)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        geometries.append(mesh)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280, height=720,
        point_show_normal=False,
    )


def _visualize_matplotlib(promoted_depth_log, title):
    """Fallback matplotlib visualization."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if not promoted_depth_log:
        print("No promoted landmarks to visualize.")
        return
    
    points = np.array([x[1] for x in promoted_depth_log])
    inlier_probs = np.array([x[4] for x in promoted_depth_log])
    
    # Compute depth sigma for each
    depth_sigmas = []
    for _, pt, df_mu, df_sigma2, _, _ in promoted_depth_log:
        d = 1.0 / max(abs(df_mu), 1e-10)
        s = np.sqrt(df_sigma2) * d * d
        depth_sigmas.append(s)
    depth_sigmas = np.array(depth_sigmas)
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D scatter colored by depth sigma
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                     c=depth_sigmas, cmap='RdYlGn_r', s=5, alpha=0.7)
    plt.colorbar(sc, ax=ax1, label='Depth σ (m)')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title(f'{title}\n({len(points)} landmarks)')
    
    # Histogram of depth sigmas
    ax2 = fig.add_subplot(122)
    ax2.hist(depth_sigmas, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(np.median(depth_sigmas), color='red', linestyle='--', 
                label=f'Median: {np.median(depth_sigmas):.4f}m')
    ax2.set_xlabel('Depth σ (meters)')
    ax2.set_ylabel('Count')
    ax2.set_title('Depth Uncertainty Distribution at Promotion')
    ax2.legend()
    
    # Stats text
    stats_text = (f"N={len(points)}\n"
                  f"σ_d: [{depth_sigmas.min():.4f}, {np.median(depth_sigmas):.4f}, {depth_sigmas.max():.4f}]m\n"
                  f"Inlier prob: [{inlier_probs.min():.2f}, {np.median(inlier_probs):.2f}, {inlier_probs.max():.2f}]")
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('depth_filter_visualization.png', dpi=150)
    print("Saved to depth_filter_visualization.png")
    plt.show()


def save_depth_log(promoted_depth_log, path="promoted_depths.pkl"):
    """Save promoted depth log to disk."""
    with open(path, "wb") as f:
        pickle.dump(promoted_depth_log, f)
    print(f"Saved {len(promoted_depth_log)} promoted landmarks to {path}")


def load_depth_log(path="promoted_depths.pkl"):
    """Load promoted depth log from disk."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} promoted landmarks from {path}")
    return data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = os.path.join(os.path.dirname(__file__), "promoted_depths.pkl")
    
    if not os.path.exists(log_path):
        print(f"No depth log found at {log_path}")
        print("Run main_threaded.py first, then save with:")
        print("  from visualize_depth_filter import save_depth_log")
        print("  save_depth_log(optimizer.promoted_depth_log)")
        sys.exit(1)
    
    log = load_depth_log(log_path)
    visualize_promoted_depths(log)
