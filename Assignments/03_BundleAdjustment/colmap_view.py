import open3d as o3d
import argparse

def load(ply_path):
    """
        Load and visualize a 3D point cloud from a PLY file.

        Input: ply_path (str): Path to the .ply file containing the point cloud.
                            The file is expected to contain 3D points (and optionally colors)
                            exported from COLMAP or similar reconstruction tools.
        """
    # Load point cloud from PLY
    pcd = o3d.io.read_point_cloud(ply_path)

    # Basic info
    print("Loaded point cloud:")
    print(pcd)
    print(f"Number of points: {len(pcd.points)}")

    # Visualize
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLY Point Cloud Viewer")
    parser.add_argument(
        "--ply",
        type=str,
        default=".\data\colmap\sparse\\0\sparse.ply",
        help="Path to .ply file"
    )

    args = parser.parse_args()
    load(args.ply)