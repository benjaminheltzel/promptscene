'''Dataloader for 3D points.'''

from glob import glob
import multiprocessing as mp
from os.path import join, exists, basename, dirname
import os
import numpy as np
import torch
# import SharedArray as SA
# import dataset.augmentation as t
from voxelizer import Voxelizer
import MinkowskiEngine as ME
import albumentations as A
import open3d as o3d


def sa_create(name, var):
    '''Create share memory.'''

    shared_mem = SA.create(name, var.shape, dtype=var.dtype)
    shared_mem[...] = var[...]
    shared_mem.flags.writeable = False
    return shared_mem


def collation_fn(batch):
    '''
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    '''
    coords, feats, labels = list(zip(*batch))

    for i, coord in enumerate(coords):
        coord[:, 0] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels)


def collation_fn_eval_all(batch):
    '''
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    '''
    coords, feats, labels, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i, coord in enumerate(coords):
        coord[:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), torch.cat(inds_recons)

def collation_fn_eval_all_merged(batch):
    coords, feats, labels, inds_recons, data, points, colors, features, unique_map, inverse_map, point2segment, point2segment_full, split_name = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i, coord in enumerate(coords):
        coord[:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]
       
    return {
            'coords': torch.cat(coords),
            'feats': torch.cat(feats),
            'labels': torch.cat(labels),
            'inds_reconstruct': torch.cat(inds_recons),
            'data': data,
            'points': points,
            'colors': colors,
            'features': features,
            'unique_map': unique_map,
            'inverse_map': inverse_map,
            'point2segment': point2segment,
            'point2segment_full': point2segment_full,
            'split_name': split_name
            }
    

def load_mesh(pcl_file):
    
    # load point cloud
    input_mesh_path = pcl_file
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    return mesh
    
class Point3DLoader(torch.utils.data.Dataset):
    '''Dataloader for 3D points and labels.'''

    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, datapath_prefix='data', voxel_size=0.05,
                 split='train', aug=False, memcache_init=False, identifier=1233, loop=1,
                 data_aug_color_trans_ratio=0.1,
                 data_aug_color_jitter_std=0.05,
                 data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2,
                 eval_all=False, input_color=False, use_augmentation_paths=False
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"
        self.split = split
        if split is None:
            split = ''
        self.identifier = identifier
        self.use_augmentation_paths = use_augmentation_paths
        
        if split == 'all':
            self.data_paths = sorted(glob(join(datapath_prefix, 'train', '*.pth'))
                                    + glob(join(datapath_prefix, 'val', '*.pth'))
                                    + glob(join(datapath_prefix, 'test', '*.pth')))
            if self.use_augmentation_paths:
                self.data_paths = sorted([os.path.join(root, file) for root, _, files in os.walk(datapath_prefix) for file in files if file.endswith('.pth')])
                
                print(self.data_paths)
                        
        else:
            self.data_paths = sorted(glob(join(datapath_prefix, split, '*.pth')))
        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the point loader.')

        self.input_color = input_color
        self.voxel_size = voxel_size
        self.aug = aug
        self.loop = loop
        self.eval_all = eval_all
        dataset_name = datapath_prefix.split('/')[-1]
        self.dataset_name = dataset_name
        self.use_shm = memcache_init

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if aug:
            prevoxel_transform_train = [
                t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
            input_transforms = [
                t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(data_aug_color_trans_ratio),
                t.ChromaticJitter(data_aug_color_jitter_std),
                t.HueSaturationTranslation(
                    data_aug_hue_max, data_aug_saturation_max),
            ]
            self.input_transforms = t.Compose(input_transforms)

        if memcache_init and (not exists("/dev/shm/%s_%s_%06d_locs_%08d" % (dataset_name, split, identifier, 0))):
            print('[*] Starting shared memory init ...')
            print('No. CPUs: ', mp.cpu_count())
            for i, (locs, feats, labels) in enumerate(torch.utils.data.DataLoader(
                    self.data_paths, collate_fn=lambda x: torch.load(x[0]),
                    num_workers=min(16, mp.cpu_count()), shuffle=False)):
                labels[labels == -100] = 255
                labels = labels.astype(np.uint8)
                # no color in the input point cloud, e.g nuscenes
                if np.isscalar(feats) and feats == 0:
                    feats = np.zeros_like(locs)
                # Scale color to 0-255
                feats = (feats + 1.) * 127.5
                sa_create("shm://%s_%s_%06d_locs_%08d" %
                          (dataset_name, split, identifier, i), locs)
                sa_create("shm://%s_%s_%06d_feats_%08d" %
                          (dataset_name, split, identifier, i), feats)
                sa_create("shm://%s_%s_%06d_labels_%08d" %
                          (dataset_name, split, identifier, i), labels)
            print('[*] %s (%s) loading 3D points done (%d)! ' %
                  (datapath_prefix, split, len(self.data_paths)))

    def __getitem__(self, index):
        #index = index_long % len(self.data_paths)
        # get data for Openscene
        if self.eval_all:
            coords, feats, labels, inds_reconstruct = self._getitem_for_openscene(index)
        else:
            coords, feats, labels = self._getitem_for_openscene(index)

        # get data for Mask3D
        data, points, colors, features, unique_map, inverse_map, point2segment, point2segment_full = self._getitem_for_mask3d(index)
        
        
        # Get name of split (train, val, test)
        path = self.data_paths[index]
        
        split_name = basename(dirname(path))
        
        if split_name not in ["train", "val", "test"]:
            split_name = join(basename(dirname(dirname(path))), "augmented_features")
        
        #if self.eval_all:
        #    return {
        #        'coords': coords,
        #        'feats': feats,
        #        'labels': labels,
        #        'inds_reconstruct': inds_reconstruct,
        #        'data': data,
        #        'points': points,
        #        'colors': colors,
        #        'features': features,
        #        'unique_map': unique_map,
        #        'inverse_map': inverse_map,
        #        'point2segment': point2segment,
        #        'point2segment_full': point2segment_full
        #    }
        #else:
        #    return {
        #        'coords': coords,
        #        'feats': feats,
        #        'labels': labels,
        #        'data': data,
        #        'points': points,
        #        'colors': colors,
        #        'features': features,
        #        'unique_map': unique_map,
        #        'inverse_map': inverse_map,
        #        'point2segment': point2segment,
        #        'point2segment_full': point2segment_full
        #    }
        if self.eval_all:
            return coords, feats, labels, inds_reconstruct, data, points, colors, features, unique_map, inverse_map, point2segment, point2segment_full, split_name
        else:
            return coords, feats, labels, data, points, colors, features, unique_map, inverse_map, point2segment, point2segment_full, split_name
        
        
    def _getitem_for_openscene(self, index):
        if self.use_shm:
            locs_in = SA.attach("shm://%s_%s_%06d_locs_%08d" %
                                (self.dataset_name, self.split, self.identifier, index)).copy()
            feats_in = SA.attach("shm://%s_%s_%06d_feats_%08d" %
                                 (self.dataset_name, self.split, self.identifier, index)).copy()
            labels_in = SA.attach("shm://%s_%s_%06d_labels_%08d" %
                                  (self.dataset_name, self.split, self.identifier, index)).copy()
        else:
            locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            # no color in the input point cloud, e.g nuscenes
            if np.isscalar(feats_in) and feats_in == 0:
                feats_in = np.zeros_like(locs_in)
            feats_in = (feats_in + 1.) * 127.5

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs, feats, labels, inds_reconstruct = self.voxelizer.voxelize(
            locs, feats_in, labels_in)
        if self.eval_all:
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat(
            (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.
        else:
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()
        if self.eval_all:
            return coords, feats, labels, torch.from_numpy(inds_reconstruct).long()
        return coords, feats, labels

    def _getitem_for_mask3d(self, index):
        # From OpenYOLO3D source code
        filename = self.data_paths[index]
        if self.use_augmentation_paths and "augmentations" in os.path.normpath(filename):
            sample_name = os.path.basename(filename)
            new_filename = sample_name.split("_")[0] + "_mesh.ply"
            filename = os.path.join(os.path.dirname(os.path.dirname(filename)),  new_filename)
        else:
            filename = filename.replace(".pth", "_mesh.ply")
        color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
        color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
        normalize_color = A.Normalize(mean=color_mean, std=color_std)
        mesh = load_mesh(filename)
        points = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
        colors = colors * 255.
        pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
        colors = np.squeeze(normalize_color(image=pseudo_image)["image"])
        coords = np.floor(points / 0.02)
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coords,
            features=colors,
            return_index=True,
            return_inverse=True,
        )
    
        sample_coordinates = coords[unique_map]
        coordinates = [torch.from_numpy(sample_coordinates).int()]
        sample_features = colors[unique_map]
        features = [torch.from_numpy(sample_features).float()]
        point2segment = None
        point2segment_full = None
        coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
        features = torch.cat(features, dim=0)
        data = ME.SparseTensor(
            coordinates=coordinates,
            features=features,
            device=self.device,
        )
        return data, points, colors, features, unique_map, inverse_map, point2segment, point2segment_full
        

    def __len__(self):
        return len(self.data_paths) * self.loop