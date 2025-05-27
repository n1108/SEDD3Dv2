import os
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
# from utils.spilt_scenes import split_scenes
# from utils.mask_scene import mask_scene

class Kitti360Dataset(Dataset):
    def __init__(self, 
        directory,
        quantized_directory=None,
        data_argumentation=False, 
        num_frames=1, 
        mode='train', 
        prev_stage='none',
        next_stage='s_1',
        prev_data_size=(32, 32, 4),
        next_data_size=(256, 256, 16), 
        prev_scene_path=None,
        infer_data_source='dataset',
        args=None,
        ):
        self.args = args
        self._directory = directory
        self.quantized_directory = quantized_directory
        self._num_frames = num_frames
        self.data_argumentation = data_argumentation
        self.infer_data_source = infer_data_source
        self.prev_data_size = prev_data_size
        self.next_data_size = next_data_size
        self.mode = mode
        self.prev_stage = prev_stage
        self.next_stage = next_stage
        self.prev_scene_path = prev_scene_path
        self._scenes = sorted([scene for scene in os.listdir(self._directory)])
        if hasattr(args, 'crop_size'):
            self.crop_size = args.crop_size
            if isinstance(self.crop_size, int):
                self.crop_size = [self.crop_size, self.crop_size, self.crop_size]
        else:
            self.crop_size = None

        self._num_scenes = len(self._scenes)
        self._num_frames_scene = []
        self._eval_labels = []
        self._frames_list = []
        self._completion_list = []
        self._quantized_eval_labels = []

        for scene in self._scenes:
            eval_dir = os.path.join(self._directory, scene)
            quantized_dir = os.path.join(self.quantized_directory, scene)
            frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(eval_dir)) if filename.endswith('.npy')]
            
            self._num_frames_scene.append(len(os.listdir(eval_dir)))  
            self._frames_list.extend(frames_list)
            self._eval_labels.extend([os.path.join(eval_dir, str(frame).zfill(6)+'.npy') for frame in frames_list])
            self._quantized_eval_labels.extend([os.path.join(quantized_dir, str(frame).zfill(6)+'.npy') for frame in frames_list])

        if self.mode in ['inference'] and self.prev_stage!='none' and self.infer_data_source=='generation':
            for root_, _, files in os.walk(self.prev_scene_path):
                for file in files:
                    if file.endswith('.txt') or file.endswith('.npy'):
                        self._completion_list.append(os.path.join(root_, file))
            self._completion_list.sort(key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))

        self.true_length = len(self._completion_list) if self.mode in ['inference'] and self.prev_stage != 'none' and self.infer_data_source == 'generation' else sum(self._num_frames_scene)

        # self.check_data_dimensions()

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        path, include_dirs = path.split(';', 1)
        if len(include_dirs) == 0:
            include_dirs = None
        return cls(path, include_dirs=include_dirs, **kwargs)

    # def check_data_dimensions(self):
    #     expected_quantized_dims = {
    #         'kitti360_s_1': (64, 64, 16),
    #         'kitti360_s_2': (128, 128, 32),
    #     }
    #     expected_eval_dims = {
    #         'kitti360_s_1': (64, 64, 16),
    #         'kitti360_s_2': (128, 128, 32),
    #         'kitti360_s_3': (256, 256, 64),
    #     }

    #     if self.prev_stage in expected_quantized_dims:
    #         sample_quantized = np.load(self._quantized_eval_labels[0])
    #         if sample_quantized.shape != expected_quantized_dims[self.prev_stage]:
    #             raise ValueError(f"Incorrect dimensions for _quantized_eval_labels. Expected {expected_quantized_dims[self.prev_stage]}, got {sample_quantized.shape}")

    #     if self.next_stage in expected_eval_dims:
    #         sample_eval = np.load(self._eval_labels[0])
    #         if sample_eval.shape != expected_eval_dims[self.next_stage]:
    #             raise ValueError(f"Incorrect dimensions for _eval_labels. Expected {expected_eval_dims[self.next_stage]}, got {sample_eval.shape}")


    def __len__(self):
        if hasattr(self.args, 'merge_into_one_epoch') and self.args.merge_into_one_epoch:
            return self.true_length * self.args.epochs
        else:
            return self.true_length
    
    def collate_fn(self, data):
        voxel_batch = [bi[0] for bi in data]
        output_batch = [bi[1] for bi in data]
        voxel_batch = torch.from_numpy(np.asarray(voxel_batch)).long()
        output_batch = torch.from_numpy(np.asarray(output_batch)).long()
        return voxel_batch, output_batch

    def __getitem__(self, idx):
        idx = idx % self.true_length
        idx_range = self.find_horizon(idx)

        
        quantized_output = np.load(self._quantized_eval_labels[idx_range[-1]])
        next_stage_data = np.load(self._eval_labels[idx_range[-1]])
        counts = next_stage_data.copy()
        prev_stage_data = next_stage_data.copy()

        if self.data_argumentation:
            next_stage_data, counts, quantized_output = self.apply_data_augmentation(next_stage_data, counts, quantized_output)
        
        if self.crop_size:
            next_stage_data_crop, quantized_output_crop = self.apply_crop(next_stage_data, quantized_output)
            h,w,u = quantized_output_crop.shape
            while (quantized_output_crop != 0).sum() < 0.01*h*w*u:
                next_stage_data_crop, quantized_output_crop = self.apply_crop(next_stage_data, quantized_output)
            next_stage_data, quantized_output = next_stage_data_crop, quantized_output_crop
        
        if self.mode in ['inference'] and self.prev_stage != 'none' and self.infer_data_source == 'generation':
            quantized_output = self.load_generated_output(idx_range[-1])

        
        return quantized_output, next_stage_data
    
    def apply_crop(self, next_stage_data, quantized_output):
        sr = next_stage_data.shape[0] // quantized_output.shape[0]
        l1, w1, h1 = next_stage_data.shape
        l2, w2, h2 = quantized_output.shape
        crop_x = int(np.random.rand()*(1-self.crop_size[0])*quantized_output.shape[0])
        crop_y = int(np.random.rand()*(1-self.crop_size[1])*quantized_output.shape[1])
        crop_z = int(np.random.rand()*(1-self.crop_size[2])*quantized_output.shape[2])
        return next_stage_data[crop_x*sr:crop_x*sr+int(self.crop_size[0]*l1),
                               crop_y*sr:crop_y*sr+int(self.crop_size[1]*w1),
                               crop_z*sr:crop_z*sr+int(self.crop_size[2]*h1),], \
                quantized_output[crop_x:crop_x+int(self.crop_size[0]*l2),
                                 crop_y:crop_y+int(self.crop_size[1]*w2),
                                 crop_z:crop_z+int(self.crop_size[2]*h2),]

    def apply_data_augmentation(self, next_stage_data, counts, quantized_output):
        if np.random.randint(2):
            next_stage_data = np.flip(next_stage_data, axis=0)
            counts = np.flip(counts, axis=0)
            quantized_output = np.flip(quantized_output, axis=0)

        if np.random.randint(2):
            next_stage_data = np.flip(next_stage_data, axis=1)
            counts = np.flip(counts, axis=1)
            quantized_output = np.flip(quantized_output, axis=1)

        rotation_type = np.random.randint(4)
        if rotation_type > 0:
            k = rotation_type
            next_stage_data = np.rot90(next_stage_data, k, axes=(0, 1))
            counts = np.rot90(counts, k, axes=(0, 1))
            quantized_output = np.rot90(quantized_output, k, axes=(0, 1))

        return next_stage_data, counts, quantized_output

    def load_generated_output(self, idx):
        _output = np.zeros(self.prev_data_size)
        if self._completion_list[idx].endswith('txt'):
            with open(self._completion_list[idx], 'r') as file:
                for line in file:
                    label, x, y, z = map(int, map(float, line.strip().split()))
                    _output[x, y, z] = label
        elif self._completion_list[idx].endswith('npy'):
            _output = np.load(self._completion_list[idx])
        return _output.astype(np.uint8)
    
    # def split_scenes(self, next_stage_data, low_resolution, counts, sliding=False):
    #     sub_scenes_high, sub_scenes_low, sub_scenes_high_counts = split_scenes(
    #         high_resolution=next_stage_data,
    #         low_resolution=low_resolution,
    #         high_resolution_counts=counts,
    #         target_resolution_high=self.next_data_size,
    #         sliding_split=sliding,
    #         sliding_ratio=1 - self.mask_ratio
    #     )
    #     return sub_scenes_high, sub_scenes_low, sub_scenes_high_counts

    # def apply_mask(self, next_stage_data):
    #     masked_scene = mask_scene(input=next_stage_data, mask_ratio=self.mask_ratio, mask_prob=self.mask_prob)
    #     return np.concatenate((next_stage_data, masked_scene))
        
    # no enough frame
    def find_horizon(self, idx):
        end_idx = idx
        idx_range = np.arange(idx-self._num_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[end_idx]) - int(self._frames_list[i]) for i in idx_range])
        good_difs = -1 * (np.arange(-self._num_frames, 0) + 1)
        
        idx_range[good_difs != diffs] = -1

        return idx_range
