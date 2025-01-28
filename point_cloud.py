import open3d as o3d
import numpy as np
import k3d
import torch

def load_point_cloud_from_ply(file_path):
     # Load the PLY file using Open3D
    ply_data = o3d.io.read_point_cloud(file_path)
    
    # Check if the file is loaded correctly
    if ply_data.is_empty():
        print("Failed to load PLY file.")
        return

    print("PLY file loaded successfully!")
    print(f"Number of points: {len(ply_data.points)}")

    # Extract points and colors
    coords = np.asarray(ply_data.points)  # 3D coordinates
    colors = np.asarray(ply_data.colors)
    return coords, colors


def visualize_ply_with_k3d(file_path, point_size=0.05):
    """
    Load a PLY file and visualize it using k3d in a Jupyter Notebook.

    Args:
        file_path (str): Path to the PLY file.
        point_size (float): Size of the points in the visualization.
    """
    
    colors, coords = load_point_cloud_from_ply(file_path)

    # Normalize colors to 0-255 and convert to hexadecimal
    colors = (colors * 255).astype(np.uint64)
    colors_hex = (colors[:, 0] << 16) + (colors[:, 1] << 8) + colors[:, 2]

    # Visualize with k3d
    plot = k3d.plot()
    point_cloud = k3d.points(positions=coords, point_size=point_size, colors=colors_hex)
    plot += point_cloud
    return plot


def visualize_pth_with_k3d(file_path, point_size=0.05):
    coords, colors, _ = torch.load(file_path)
    
    normalized_rgb = (colors + 1) / 2
    normalized_rgb = (normalized_rgb * 255).astype(np.uint64)
    colors_hex = (normalized_rgb[:, 0] << 16) + (normalized_rgb[:, 1] << 8) + normalized_rgb[:, 2]
    
    plot = k3d.plot()
    point_cloud = k3d.points(positions=coords, point_size=0.05, colors=colors_hex)
    plot += point_cloud
    return plot

def visualize_point_cloud_with_k3d(coords, colors, point_size=0.05, is_rgb=False, is_norm=False):
    if not is_norm:
        colors = (colors + 1) / 2
    if not is_rgb:
        colors = (colors * 255).astype(np.uint64)
    colors_hex = (colors[:, 0] << 16) + (colors[:, 1] << 8) + colors[:, 2]
    
    plot = k3d.plot()
    point_cloud = k3d.points(positions=coords, point_size=0.05, colors=colors_hex)
    plot += point_cloud
    return plot