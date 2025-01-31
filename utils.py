import os
from glob import glob
import torch
import numpy as np


def get_all_files_in_dir(path, file_type="txt"):
    return sorted(glob(os.path.join(path, f"*.{file_type}")))

def get_all_files_in_dir_and_subdir(path, file_type="txt"):
    return sorted([
        os.path.join(root, file)
        for root, _, files in os.walk(path)
        for file in files
        if file.endswith(file_type)
    ])

def merge_extracted_features(output_path):

    mask3d_path = os.path.join(output_path, "mask3d")
    mask_paths = get_all_files_in_dir_and_subdir(mask3d_path, "pt")

    openscene_path = os.path.join(output_path, "openscene")
    features_paths = get_all_files_in_dir_and_subdir(openscene_path, "npy")

    print("Instance masks: ", len(mask_paths))
    print("Per point features: ", len(features_paths))

    assert len(mask_paths) == len(features_paths)

    for i in range(len(mask_paths)):

        sample_name = os.path.basename(mask_paths[i]).split('_')[0]
        print("Processing: ", sample_name)

        # Make sure that the instance masks and the point feature are from the same input sample
        assert sample_name == os.path.basename(features_paths[i]).split('_')[0]

        # Load masks and features
        masks = torch.load(mask_paths[i])
        features = np.load(features_paths[i])
        print(f"Masks shape: ({len(masks)}, {masks[0].shape[0]})")
        print(f"Features shape: {features.shape}")

        mean_instance_features = []
        # Compute average instance features
        for mask in masks:
            masked_features = features[mask,:]
            mean_instance_features.append(features[mask,:].mean(axis=0))
        mean_instance_features = np.array(mean_instance_features)
        print(f"Mean instane features: {mean_instance_features.shape}")

        folder_path = os.path.join(output_path, "instance_features")

        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, f"{sample_name}_instance_features.npy")

        np.save(file_path, mean_instance_features)

        print(f"Saved instance features for {sample_name}")
            

def merge_extracted_features_augmented(output_path, num_aug=10):

        mask3d_path = os.path.join(output_path, "mask3d")
        mask_paths = get_all_files_in_dir("dataset/OpenYOLO3D/output/replica/replica_ground_truth_masks", "pt")

        openscene_path = os.path.join(output_path, "openscene", "prompt_learning")
        
        features_paths = get_all_files_in_dir_and_subdir(openscene_path, "npy")

        print("Instance masks: ", len(mask_paths))
        print("Per point features: ", len(features_paths))

        for i in range(len(mask_paths)):
            index = i*(num_aug+1)
            sample_name = os.path.basename(features_paths[index]).split('_')[0]
            print(f"Processing {sample_name}:")
            # Make sure that the instance masks and the point feature are from the same input sample
            #assert sample_name == os.path.basename(features_paths[i]).split('_')[0]
            
            mask_path = mask_paths[i]
            for path in mask_paths:
                if sample_name in path:
                    mask_path = path
                    break
            
            # Load masks and features
            masks,_ = torch.load(mask_path)
            masks = masks.T
            print(f"Masks shape: {masks.shape}")
            #features = np.load(features_paths[i])
            
            mean_instance_feature_list = []
            for j in range(num_aug+1):
                feature_path = features_paths[i*(num_aug+1)+j]
                #print(feature_path)
                features = np.load(feature_path)
            #features = np.array(features)
                print(f"Features shape: {features.shape}")

                mean_instance_features = []
                # Compute average instance features
                for mask in masks:
                    mask = mask != 0
                    masked_features = features[mask,:]
                    mean_instance_features.append(features[mask,:].mean(axis=0))
                mean_instance_features = np.array(mean_instance_features)
                #print(mean_instance_features.shape)
                mean_instance_feature_list.append(mean_instance_features)
                
            mean_instance_features = np.array(mean_instance_feature_list)
            print("Mean instance feature list shape: ", mean_instance_features.shape)
            mean_instance_features = mean_instance_features.mean(axis=0)
            
            folder_path = os.path.join(output_path, "instance_features_prompt")

            os.makedirs(folder_path, exist_ok=True)

            file_path = os.path.join(folder_path, f"{sample_name}_instance_features.npy")

            np.save(file_path, mean_instance_features)

            print(f"Saved instance features for {sample_name}")