import os
import numpy as np
import torch
import open3d as o3d
import k3d
import webbrowser
import tempfile
import sys

sys.path.append("models/openscene/dataset")
from label_constants import MATTERPORT_COLOR_MAP_160

import utils



scene_names = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]
splits = {
    "train": [0,1,2,5],
    "test": [4, 7],
    "val": [3,6]
}
POINT_SIZE = 2


openscene_feature_basepath = "experiments/run_2025-01-29-20-44-08/instance_features_with_gt/"
merged_prediction_basepath = "experiments/run_2025-01-29-20-44-08/instance_features/"
mask3d_base_path = "experiments/run_2025-01-29-20-44-08/mask3d/"
gt_basepath = "dataset/data/ground_truth/"
point_cloud_paths = utils.get_all_files_in_dir_and_subdir("experiments/run_2025-01-29-20-44-08/openscene", "input.ply")

def get_split(split):
    if split == "all":
        return scene_names
    elif split in ["train", "test", "val"]:
        return [scene_names[index] for index in splits[split]]
    else:
        raise ValueError("Only 'train', 'val', 'test' or 'all' are valid inputs!")
    
def get_palette():
    scannet_palette = []
    for _, value in MATTERPORT_COLOR_MAP_160.items():
        scannet_palette.append(np.array(value))
    palette = np.concatenate(scannet_palette)
    
    return palette


def load_point_cloud_data(pcd):
    """Load a point cloud and extract coordinates and colors."""
    coords = np.asarray(pcd.points)  # 3D coordinates
    colors = np.asarray(pcd.colors) 

    # Set base color
    #colors[:] = 0.5

    # Normalize colors to 0-255 and convert to hexadecimal
    #colors = (colors * 255).astype(np.uint64)

    return coords, colors

def apply_color_mapping(colors, indices, palette, labels):
    """Apply colors based on given indices and label values."""
    valid_indices = labels != -1  # Ignore invalid labels
    colors[indices[valid_indices]] = palette[labels[valid_indices] * 3 : labels[valid_indices] * 3 + 3]
    return colors

def visualize_point_cloud(coords, colors, title):
    """Visualize the point cloud with K3D."""
    print(f"Visualizing: {title}")
    colors_hex = (colors[:, 0] << 16) + (colors[:, 1] << 8) + colors[:, 2]
    
    plot = k3d.plot()
    point_cloud = k3d.points(positions=coords, point_size=POINT_SIZE, colors=colors_hex)
    plot += point_cloud
    # Generate standalone HTML file
    html_output = plot.get_snapshot()
    js_script = """
        <script>
            function logCameraPosition() {
                const cam = window.K3DInstance.camera;
                console.log("Camera Position:", cam.position.x, cam.position.y, cam.position.z);
                console.log("Look At:", cam.target.x, cam.target.y, cam.target.z);
                console.log("Up Vector:", cam.up.x, cam.up.y, cam.up.z);
            }

            setTimeout(() => {
                if (window.K3DInstance) {
                    logCameraPosition();  // Log position after K3D loads
                    window.K3DInstance.controls.addEventListener('change', logCameraPosition); // Update on movement
                }
            }, 2000);
        </script>
        """

    # Combine HTML and JavaScript
    html_output = html_output.replace("<head>", f"<head><title>{title}</title>", 1)
    html_output = html_output.replace("</body>", js_script + "</body>")

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        tmp_file.write(html_output.encode('utf-8'))
        tmp_file_path = tmp_file.name

    # Open in browser
    webbrowser.open("file://" + tmp_file_path)


def process_scene(scene_name, selected_visualizations, palette):
    """Process and visualize a specific scene based on user selections."""
    point_cloud_path = next((p for p in point_cloud_paths if scene_name in p), None)
    if not point_cloud_path:
        print(f"Scene {scene_name} not found.")
        return
    
    split = os.path.basename(os.path.dirname(point_cloud_path))
    print(f"Processing: {scene_name} {split}")
    
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    final_mask_path = os.path.join("dataset/data/ground_truth", f"{scene_name}.npy")
    final_mask = np.load(final_mask_path)
    final_mask = final_mask == -1
    
    if "ground_truth" in selected_visualizations:
        coords, colors = load_point_cloud_data(pcd)
        gt_labels_data = np.load(os.path.join(gt_basepath, f"{scene_name}.npy"))
        colors[:] = 0.5  # Reset color
        colors = (colors * 255).astype(np.uint64)
        for i,point in enumerate(gt_labels_data):
            if point == -1:
                continue
            colors[i] = (palette[point*3],palette[point*3+1],palette[point*3+2])
        colors[final_mask] = (127,127,127)
        visualize_point_cloud(coords, colors, f"{scene_name}: Per-Point Ground Truth")
    
    if "instance_classification" in selected_visualizations:
        coords, colors = load_point_cloud_data(pcd)
        pred_path = os.path.join(merged_prediction_basepath, split, f"{scene_name}_normalized_predicted_classes.pl")
        pred_cl = torch.load(pred_path)
        masks = torch.load(os.path.join(mask3d_base_path, split, f"{scene_name}_masks.pt"))
        colors[:] = 0.5  # Reset color
        colors = (colors * 255).astype(np.uint64)
        
        for i,mask in enumerate(masks):
            colors[mask] = (palette[pred_cl[i]*3],palette[pred_cl[i]*3+1],palette[pred_cl[i]*3+2])
        
        colors[final_mask] = (127,127,127)
        visualize_point_cloud(coords, colors, f"{scene_name}: Instance Feature Classification No Prompt Learning")
    
    if "instance_classification_prompt_learning" in selected_visualizations:
        coords, colors = load_point_cloud_data(pcd)
        pred_path = os.path.join(merged_prediction_basepath, split, f"{scene_name}_normalized_prompt_learning_predicted_classes.pl")
        pred_cl = torch.load(pred_path)
        masks = torch.load(os.path.join(mask3d_base_path, split, f"{scene_name}_masks.pt"))
        colors[:] = 0.5  # Reset color
        colors = (colors * 255).astype(np.uint64)
        for i,mask in enumerate(masks):
            colors[mask] = (palette[pred_cl[i]*3],palette[pred_cl[i]*3+1],palette[pred_cl[i]*3+2])
        
        colors[final_mask] = (127,127,127)
        visualize_point_cloud(coords, colors, f"{scene_name}: Instance Feature Classification Prompt Learning")

    if "openscene_instance_predictions" in selected_visualizations:
        coords, colors = load_point_cloud_data(pcd)
        predicted_classes = torch.load(os.path.join(openscene_feature_basepath, f"{scene_name}_normalized_predicted_classes.pl"))
        gt_mask_path = os.path.join("dataset/gt_masks", f"{scene_name}.pt")
        gt_masks = torch.load(gt_mask_path)[0].T

        colors[:] = 0.5  # Reset color
        colors = (colors * 255).astype(np.uint64)
        for i,point in enumerate(predicted_classes):
            mask = gt_masks[i] != 0
            colors[mask] = (palette[point*3],palette[point*3+1],palette[point*3+2])
        #colors = apply_color_mapping(colors, np.arange(len(predicted_classes)), palette, predicted_classes)
        
        colors[final_mask] = (127,127,127)
        visualize_point_cloud(coords, colors, f"{scene_name}: OpenScene Per-Point Predictions")

    if "openscene_instance_predictions_prompt_learning" in selected_visualizations:
        coords, colors = load_point_cloud_data(pcd)
        predicted_classes = torch.load(os.path.join(openscene_feature_basepath, f"{scene_name}_normalized_prompt_learning_predicted_classes.pl"))
        gt_mask_path = os.path.join("dataset/gt_masks", f"{scene_name}.pt")
        gt_masks = torch.load(gt_mask_path)[0].T

        colors[:] = 0.5  # Reset color
        colors = (colors * 255).astype(np.uint64)
        for i,point in enumerate(predicted_classes):
            mask = gt_masks[i] != 0
            colors[mask] = (palette[point*3],palette[point*3+1],palette[point*3+2])
        #colors = apply_color_mapping(colors, np.arange(len(predicted_classes)), palette, predicted_classes)
        colors[final_mask] = (127,127,127)
        visualize_point_cloud(coords, colors, f"{scene_name}: OpenScene Per-Point Predictions Prompt Learning")

    if "mask3d_predictions" in selected_visualizations:
        coords, colors = load_point_cloud_data(pcd)
        masks = torch.load(os.path.join(mask3d_base_path, split, f"{scene_name}_masks.pt"))
        gt_labels_data = np.load(os.path.join(gt_basepath, f"{scene_name}.npy"))
        colors[:] = 0.5  # Reset color
        colors = (colors * 255).astype(np.uint64)
        for i, mask in enumerate(masks):
            instance_id = int(gt_labels_data[mask].mean().item())  # Use majority instead of mean
            if instance_id == -1:
                continue
            colors[mask] = (palette[instance_id * 3], palette[instance_id * 3 + 1], palette[instance_id * 3 + 2])
        
        colors[final_mask] = (127,127,127)
        visualize_point_cloud(coords, colors, f"{scene_name}: Mask3D Instance Predictions")


if __name__ == "__main__":
    # User selection
    selected_scenes = ["office1"] # Replace with desired scene names
    selected_visualizations = {"ground_truth", "instance_classification", "instance_classification_prompt_learning", "openscene_instance_predictions","openscene_instance_predictions_prompt_learning", "mask3d_predictions"}  # Choose what to display
    palette = get_palette()

    for scene in selected_scenes:
        process_scene(scene, selected_visualizations, palette)