import hydra
from omegaconf import DictConfig, OmegaConf
from models.mask3d import Mask3D
import os
import torch

import MinkowskiEngine as ME
import open3d as o3d
import numpy as np
import albumentations as A

from datetime import datetime
import sys
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset'))
print(dataset_path)
if dataset_path not in sys.path:
    sys.path.append(dataset_path)

from point_loader import Point3DLoader

from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

from datasets.scannet200.scannet200_constants import (
    SCANNET_COLOR_MAP_200,
    SCANNET_COLOR_MAP_20,
    VALID_CLASS_IDS_200,
    VALID_CLASS_IDS_20,
    CLASS_LABELS_200,
    CLASS_LABELS_20,
)

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)


    def forward(self, x, raw_coordinates=None):
        return self.model(x, raw_coordinates=raw_coordinates)

@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device: ", device)
    
    # Avoid annoying o3d outpouts and warnings
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    os.chdir(hydra.utils.get_original_cwd())
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    model = model.to(device)
    # model.eval()
    #time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #output_path = os.path.join(cfg.general.save_dir, f"run_{time_stamp}")
    
    #if not os.path.exists(output_path):
     #       os.makedirs(output_path)
    output_path = cfg.general.save_dir
    print("Loading checkpoint!")
    print("Save dir: ", output_path)
    print("Data root: ", cfg.general.data_dir)
    
    val_dataset = Point3DLoader(datapath_prefix=cfg.general.data_dir,  
                        voxel_size=0.02,   
                        split=cfg.general.split, aug=False,  
                        memcache_init=False, eval_all=True,
                        input_color=False)
    
    print("Dataset: ", len(val_dataset))
    
    
    for idx, batch in enumerate(val_dataset):
        coords, feats, labels, inds_reconstruct, data, points, colors, features, unique_map, inverse_map, point2segment, point2segment_full, split_name = batch
        file_path = val_dataset.data_paths[idx]
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing batch {idx} from file {file_name} ....")
        with torch.no_grad():
            outputs = model(data, raw_coordinates=features)
            
        del data
        torch.cuda.empty_cache()

        # parse predictions
        logits = outputs["pred_logits"]
        masks = outputs["pred_masks"]


        # reformat predictions
        logits = logits[0].detach().cpu()
        masks = masks[0].detach().cpu()
        print("Shape of mask: ", masks.shape)
        print("Shape of logits: ", logits.shape)

        labels = []
        confidences = []
        masks_binary = []

        for i in range(len(logits)):
            p_labels = torch.softmax(logits[i], dim=-1)
            p_masks = torch.sigmoid(masks[:, i])
            l = torch.argmax(p_labels, dim=-1)
            c_label = torch.max(p_labels)            
            m = p_masks > 0.5
            c_m = p_masks[m].sum() / (m.sum() + 1e-8)
            c = c_label * c_m
            if l < 200 and c > 0.5:
                labels.append(l.item())
                confidences.append(c.item())
                masks_binary.append(m[inverse_map]) # mapping the mask back to the original point cloud
        
        current_output_path = os.path.join(output_path, split_name)
        os.makedirs(current_output_path, exist_ok=True)
        
        label_file = os.path.join(current_output_path, f"{file_name}_labels.txt")
        with open(label_file, "w") as file:
            for label in labels:
                file.write(f"{label}\n")
        
        confidences_file = os.path.join(current_output_path, f"{file_name}_confidences.txt")
        with open(confidences_file, "w") as file:
            for confidence in confidences:
                file.write(f"{confidence}\n")
                
        masks_binary_file = os.path.join(current_output_path, f"{file_name}_masks.pt")
        torch.save(masks_binary, masks_binary_file)
        #with open(masks_binary_file, "w") as file:
        #    for mask_binary in masks_binary:
        #        file.write(f"{mask_binary}\n")
        
        print("Shape of labels output: ", len(labels))
        print("Shape of confidences output: ", len(confidences))
        print("Shape of masks_binary output: ", len(masks_binary), masks_binary[0].shape)
        
        

        """
        # save labelled mesh
       # mesh_labelled = o3d.geometry.TriangleMesh()
        #mesh_labelled.vertices = mesh.vertices
        #mesh_labelled.triangles = mesh.triangles

        #labels_mapped = np.zeros((len(mesh.vertices), 1))
        #colors_mapped = np.zeros((len(mesh.vertices), 3))

        confidences, labels, masks_binary = zip(*sorted(zip(confidences, labels, masks_binary), reverse=False))
        #for i, (l, c, m) in enumerate(zip(labels, confidences, masks_binary)):
        #    labels_mapped[m == 1] = l
        #    if l == 0:
         #       l_ = -1 + 2 # label offset is 2 for scannet 200, 0 needs to be mapped to -1 before (see trainer.py in Mask3D)
         #   else:
         #       l_ = l + 2
            # print(VALID_CLASS_IDS_200[l_], SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l_]], l_, CLASS_LABELS_200[l_])
         #   colors_mapped[m == 1] = SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l_]]

            # colors_mapped[mask_mapped == 1] = SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l]]
        

        
        #output_dir = cfg.general.save_dir
        #mesh_labelled.vertex_colors = o3d.utility.Vector3dVector(colors_mapped.astype(np.float32) / 255.)
        #o3d.io.write_triangle_mesh(f'{output_dir}/mesh_tsdf_labelled.ply', mesh_labelled)

        mask_path = os.path.join(output_dir, 'pred_mask')
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        # sorting by confidence
        with open(os.path.join(output_dir, 'mask3d_predictions.txt'), 'w') as f:
            for i, (l, c, m) in enumerate(zip(labels, confidences, masks_binary)):
                mask_file = f'pred_mask/{str(i).zfill(3)}.txt'
                f.write(f'{mask_file} {VALID_CLASS_IDS_200[l]} {c}\n')
                np.savetxt(os.path.join(output_dir, mask_file), m.numpy(), fmt='%d')
       """

if __name__ == "__main__":
    main()