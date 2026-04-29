import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import open3d as o3d

# Functions definitions

def euler_to_matrix(euler):
    """
    Convert Euler angles to rotation matrices

    Inputs:
        euler (torch.Tensor): shape (V, 3)
            V = number of camera views
            Each row contains (rx, ry, rz) rotation angles in radians

    Outputs:
        R (torch.Tensor): shape (V, 3, 3)
            Rotation matrix for each camera
    """

    # Extract individual rotation angles
    rx, ry, rz = euler[:, 0], euler[:, 1], euler[:, 2]

    # Precompute sin/cos for efficiency
    cosx, sinx = torch.cos(rx), torch.sin(rx)
    cosy, siny = torch.cos(ry), torch.sin(ry)
    cosz, sinz = torch.cos(rz), torch.sin(rz)

    # Rotation around X-axis
    Rx = torch.stack([
        torch.stack([torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx)], dim=-1),
        torch.stack([torch.zeros_like(rx), cosx, -sinx], dim=-1),
        torch.stack([torch.zeros_like(rx), sinx, cosx], dim=-1)
    ], dim=-2)

    # Rotation around Y-axis
    Ry = torch.stack([
        torch.stack([cosy, torch.zeros_like(ry), siny], dim=-1),
        torch.stack([torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry)], dim=-1),
        torch.stack([-siny, torch.zeros_like(ry), cosy], dim=-1)
    ], dim=-2)

    # Rotation around Z-axis
    Rz = torch.stack([
        torch.stack([cosz, -sinz, torch.zeros_like(rz)], dim=-1),
        torch.stack([sinz, cosz, torch.zeros_like(rz)], dim=-1),
        torch.stack([torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)], dim=-1)
    ], dim=-2)

    # Combine rotations (XYZ convention)
    # Note: order matters!
    R = torch.bmm(Rx, torch.bmm(Ry, Rz))

    return R

def project(points3d, euler, trans):
    """
    Project 3D points into 2D image coordinates

    Inputs:
        points3d (torch.Tensor): shape (N, 3)
            3D coordinates of points in world space

        euler (torch.Tensor): shape (V, 3)
            Euler angles for each camera (rx, ry, rz)

        trans (torch.Tensor): shape (V, 3)
            Translation vectors for each camera

    Outputs:
        projected_points (torch.Tensor): shape (V, N, 2)
            2D pixel coordinates of each point in each view
    """

    V = euler.shape[0]
    N = points3d.shape[0]

    # Convert Euler angles to rotation matrices
    R = euler_to_matrix(euler)  # (V, 3, 3)

    # Duplicate 3D points across all views
    P = points3d.unsqueeze(0).expand(V, N, 3)  # (V, N, 3)

    # Transform points into camera coordinate system
    # Pc = R * P + T
    Pc = torch.bmm(P, R.transpose(1, 2)) + trans.unsqueeze(1)

    Xc = Pc[..., 0]
    Yc = Pc[..., 1]
    Zc = Pc[..., 2]

    # Perspective projection
    u = -f * (Xc / (Zc + 1e-6)) + cx
    v =  f * (Yc / (Zc + 1e-6)) + cy

    return torch.stack([u, v], dim=-1)

def reprojection_loss(pred, gt, vis):
    """
    Compute reprojection error between predicted and observed 2D points

    Inputs:
        pred (torch.Tensor): shape (V, N, 2)
            Predicted 2D coordinates from projection

        gt (torch.Tensor): shape (V, N, 2)
            Ground truth 2D observations

        vis (torch.Tensor): shape (V, N)
            Visibility mask (1 = visible, 0 = occluded)

    Outputs:
        loss (torch.Tensor): scalar
            Mean squared reprojection error over visible points
    """

    diff = (pred - gt) ** 2
    diff = diff.sum(dim=-1)

    # Mask out invisible points
    diff = diff * vis

    # Normalize by number of visible points
    return diff.sum() / (vis.sum() + 1e-8)

# Visualization function
def visualize_point_cloud(points, colors=None):
    """
    Visualize a 3D point cloud

    Inputs:
        points (numpy.ndarray): shape (N, 3)
            3D coordinates of points

        colors (numpy.ndarray or None): shape (N, 3)
            RGB colors for each point (range [0, 255] or [0, 1])
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":

    # Load 2D observations (50 views)
    points2d_data = np.load("data/points2d.npz")

    # Load per-point RGB colors (for visualization)
    colors = np.load("data/points3d_colors.npy")

    # Sort view keys to ensure consistent ordering
    view_keys = sorted(points2d_data.keys())

    num_views = len(view_keys)
    num_points = points2d_data[view_keys[0]].shape[0]

    # Containers
    points2d = []
    visibility = []

    # Extract (x, y) and visibility mask for each view
    for k in view_keys:
        obs = points2d_data[k]
        points2d.append(obs[:, :2])
        visibility.append(obs[:, 2])

    # Convert to PyTorch tensors
    points2d = torch.tensor(np.stack(points2d), dtype=torch.float32)
    visibility = torch.tensor(np.stack(visibility), dtype=torch.float32)

    # Initialize parameters to optimize
    device = "cuda" if torch.cuda.is_available() else "cpu"

    points2d = points2d.to(device)
    visibility = visibility.to(device)

    # 3D points (unknown -> initialize randomly near origin)
    points3d = nn.Parameter(torch.randn(num_points, 3, device=device) * 0.5)

    # Camera rotations (Euler angles, initialized as identity)
    euler_angles = nn.Parameter(torch.zeros(num_views, 3, device=device))

    # Camera translations
    translations = nn.Parameter(torch.zeros(num_views, 3, device=device))

    # Initialize cameras in front of object (negative Z direction)
    translations.data[:, 2] = -2.5

    # Shared focal length (unknown)
    f = nn.Parameter(torch.tensor(800.0, device=device))

    # Image center (from image size 1024×1024)
    cx, cy = 512.0, 512.0

    # Optimization (Bundle Adjustment)
    optimizer = optim.Adam([
        points3d,
        euler_angles,
        translations,
        f
    ], lr=1e-2)

    losses = []

    for iter in range(2000):

        optimizer.zero_grad()

        # Forward: project 3D -> 2D
        pred = project(points3d, euler_angles, translations)

        # Compute reprojection error
        loss = reprojection_loss(pred, points2d, visibility)

        # Backpropagation
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Print training progress
        if iter % 100 == 0:
            print(f"Iter {iter}: Loss = {loss.item():.4f}, f = {f.item():.2f}")

    # Save reconstructed 3D model (.obj)
    points3d_np = points3d.detach().cpu().numpy()
    colors_np = colors / 255.0  # normalize RGB to [0, 1]

    with open("result.obj", "w") as fobj:
        for p, c in zip(points3d_np, colors_np):
            fobj.write(f"v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

    print("Saved result.obj")

    # Plot training loss curve
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("loss_curve.png")
    plt.show()

    # Visualize final reconstruction
    points_np = points3d.detach().cpu().numpy()
    visualize_point_cloud(points_np, colors)
