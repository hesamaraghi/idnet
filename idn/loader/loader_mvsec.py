# import h5pickle as h5py
import h5py
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from idn.utils.mvsec_utils import EventSequence
from idn.utils.dsec_utils import RepresentationType, VoxelGrid
from idn.utils.transformers import EventSequenceToVoxelGrid_Pytorch, apply_randomcrop_to_sample

from tqdm import tqdm
from pathlib import Path, PurePath
import hdf5plugin
from scipy.ndimage import gaussian_filter
from idn.loader.loader_dsec import HarrisRecursive


class MVSEC(Dataset):

    def __init__(
        self,
        config,
        training=True,
        representation_type=None,
        rate=20,
        transforms=[],
        filter=None,
        augment=True,
    ):
        self.config = config
        self.add_eigenvalues = config.get("add_eigenvalues", False)
        self.add_filter_values = config.get("add_filter_values", False)
        self.in_memory = config.get("in_memory", False)
        self.force_preprocess = config.get("force_preprocess", False)
        self.do_not_save_preprocessed = config.get("do_not_save_preprocessed", False)
        self.seq_path = config.common.get("data_root", "data/MVSEC")
        if training:
            self.seq_name = config.train.seq
        else:
            self.seq_name = config.val.seq
        self.num_bins = config.get("num_voxel_bins", None)

        assert Path(self.seq_path).is_dir(), f"{self.seq_path} is not a directory"
        
        self.preprocessed_path = Path(self.seq_path) / Path(self.seq_name) / 'preprocessed'
        if not self.preprocessed_path.exists():  
            self.preprocessed_path.mkdir(parents=True, exist_ok=True)
        
        self.num_bins = config.get("num_voxel_bins", None)
        self.dt = config.get("dt", None)
        if self.dt is None:
            self.event_h5 = h5py.File(os.path.join(self.seq_path, f"{self.seq_name}_data.hdf5"), "r")
            self.event = self.event_h5['davis']['left']['events']
            self.gt_h5 = h5py.File(os.path.join(self.seq_path, f"{self.seq_name}_gt.hdf5"), "r")
            self.gt_flow = self.gt_h5['davis']['left']['flow_dist']
            self.timestamps = self.gt_h5['davis']['left']['flow_dist_ts']
        else:
            assert self.dt == 1 or self.dt == 4
            self.h5 = h5py.File(os.path.join(self.seq_path, f"{self.seq_name}.h5"), "r")
            self.event = self.h5['events']
            self.timestamps = self.h5['flow']['dt={}'.format(self.dt)]['timestamps'][:, 0]
            self.gt_flow = list(self.h5['flow']['dt={}'.format(self.dt)].keys())
            self.gt_flow.remove('timestamps')
            assert sorted(self.gt_flow) == self.gt_flow
        
        if representation_type is None:
            self.representation_type = VoxelGrid
        else:
            self.representation_type = representation_type
        
        if filter is not None:
            assert isinstance(filter, tuple) and isinstance(filter[0], int)\
                and isinstance(filter[1], int)
            self.timestamps = self.timestamps[slice(*filter)]
            self.gt_flow = self.gt_flow[slice(*filter)]

        self.raw_gt_len = self.timestamps.shape[0]
        self.event_ts_to_idx = self.build_event_idx()
        self.voxel = EventSequenceToVoxelGrid_Pytorch(
            num_bins=self.num_bins,
            normalize=True,
            gpu=False,
        )
        self.image_width, self.image_height = 346, 260
        self.cropper = T.CenterCrop((256, 256))
        self.augment = augment
        
        if self.add_eigenvalues or self.add_filter_values:
            self.tau = config.get("tau", 15_000)
            self.filter_size = config.get("filter_size", 7)
            self.harris_recursive = HarrisRecursive(
                tau=self.tau,
                filter_size=self.filter_size,
                image_size=(self.image_height, self.image_width),
            )

        if self.in_memory:
            self.data = []
            print(f"Loading data for sequence {self.seq_name} into memory...")
            for i in tqdm(range(len(self))):
                self.data.append(self[i])
        
        pass

    def __len__(self):
        return self.raw_gt_len - 2

    def get_eigenvalues(self, x, y, t, p):
        dtype = [
            ('x', np.uint16),
            ('y', np.uint16),
            ('t', np.uint64),
            ('p', np.uint8),
        ]
        events = np.empty(x.shape[0], dtype=dtype)
        events['x'] = x
        events['y'] = y
        events['t'] = t
        events['p'] = p
        self.harris_recursive(events)

    def get_data_sample(self, idx):
        preprocessed_file_path = self.preprocessed_path / f"{idx:05d}.pt"
        if not self.force_preprocess and preprocessed_file_path.exists():
            # print(f"Loading preprocessed data for index {index} for sequence {self.seq_name} from {preprocessed_file_path}")
            loaded_file = torch.load(preprocessed_file_path)       
            if not self.add_eigenvalues and not self.add_filter_values:
                loaded_file['event_volume_new'] = loaded_file['event_volume_new'][:self.num_bins,:,:]
            return loaded_file
            
        idx += 1
        sample = {}
        if self.dt is None:
            # get events
            events = self.event[self.event_ts_to_idx[idx-1]:self.event_ts_to_idx[idx]]
            events = events[:, [2, 0, 1, 3]]  # make it (t, x, y, p)
            sample["event_volume_old"] = \
                self.voxel(EventSequence(events,
                                     params={'width': self.image_width,
                                             'height': self.image_height},
                                     timestamp_multiplier=1e6,
                                     convert_to_relative=True,
                                     features=events))
            
            # get events
            events = self.event[self.event_ts_to_idx[idx]:self.event_ts_to_idx[idx+1]]
            events = events[:, [2, 0, 1, 3]] # make it (t, x, y, p)

            sample["event_volume_new"] = \
                self.voxel(EventSequence(events, 
                                    params={'width': self.image_width, 
                                            'height': self.image_height},
                                    timestamp_multiplier=1e6,
                                    convert_to_relative=True,
                                    features = events))
            
            if self.add_eigenvalues or self.add_filter_values:
                x = events[:, 1]
                y = events[:, 2]
                t = events[:, 0]
                p = events[:, 3]
                self.get_eigenvalues(x, y, t, p)
                
                if self.add_eigenvalues:
                    eig1 = self.harris_recursive.eig1
                    eig2 = self.harris_recursive.eig2
                    print(f"Eigenvalues computed for index {idx - 1} and sequence {self.seq_name}")
                    print(f"Eigenvalue 1: min. {eig1.min()}, max. {eig1.max()}")
                    print(f"Eigenvalue 2: min. {eig2.min()}, max. {eig2.max()}")
                    eig1_representation = \
                        self.voxel(EventSequence(np.column_stack((t, x, y, eig1)),
                                    params={'width': self.image_width, 
                                            'height': self.image_height},
                                    timestamp_multiplier=1e6,
                                    convert_to_relative=True,
                                    features = events))
            
                    eig2_representation = \
                        self.voxel(EventSequence(np.column_stack((t, x, y, eig2)),
                                    params={'width': self.image_width, 
                                            'height': self.image_height},
                                    timestamp_multiplier=1e6,
                                    convert_to_relative=True,
                                    features = events))
                    sample["eigenvalues_volume_new"] = torch.cat(
                        (
                            eig1_representation,
                            eig2_representation,
                        ),
                        dim=0
                    )
                    print(f"Voxel grid representation with eigenvalues for index {idx - 1} and sequence {self.seq_name}")
                    print(f"Eigenvalue 1: min. {eig1_representation.min()}, max. {eig1_representation.max()}")
                    print(f"Eigenvalue 2: min. {eig2_representation.min()}, max. {eig2_representation.max()}")   
            
                if self.add_filter_values:
                    filter_values = self.harris_recursive.filter_value_recursive
                    print(f"Filter values computed for index {idx - 1} and sequence {self.seq_name}")
                    print(f"Filter values: min. {filter_values.min()}, max. {filter_values.max()}")
                    sample["filter_values_volume_new"] = \
                        self.voxel(EventSequence(np.column_stack((t, x, y, filter_values)),
                                    params={'width': self.image_width, 
                                            'height': self.image_height},
                                    timestamp_multiplier=1e6,
                                    convert_to_relative=True,
                                    features = events))
                    print(f"Voxel grid representation with filter values for index {idx - 1} and sequence {self.seq_name}")
                    print(f"Filter values: min. {sample['filter_values_volume_new'].min()}, max. {sample['filter_values_volume_new'].max()}")
          
            # get flow
            flow = self.gt_flow[idx] # -1 yields the same gt flow as E-RAFT, but likely incorrect
            flow_next = self.gt_flow[idx+1]
        else:
            old_p = self.event['ps'][self.event_ts_to_idx[idx-1]:self.event_ts_to_idx[idx]]
            old_t = self.event['ts'][self.event_ts_to_idx[idx-1]:self.event_ts_to_idx[idx]]
            old_x = self.event['xs'][self.event_ts_to_idx[idx-1]:self.event_ts_to_idx[idx]]
            old_y = self.event['ys'][self.event_ts_to_idx[idx-1]:self.event_ts_to_idx[idx]]

            old_events = np.column_stack((old_t, old_x, old_y, old_p))
            sample["event_volume_old"] = \
                self.voxel(EventSequence(old_events,
                                     params={'width': self.image_width,
                                             'height': self.image_height},
                                     timestamp_multiplier=1e6,
                                     convert_to_relative=True,
                                     features=old_events))
            
            new_p = self.event['ps'][self.event_ts_to_idx[idx]:self.event_ts_to_idx[idx+1]]
            new_t = self.event['ts'][self.event_ts_to_idx[idx]:self.event_ts_to_idx[idx+1]]
            new_x = self.event['xs'][self.event_ts_to_idx[idx]:self.event_ts_to_idx[idx+1]]
            new_y = self.event['ys'][self.event_ts_to_idx[idx]:self.event_ts_to_idx[idx+1]]

            new_events = np.column_stack((new_t, new_x, new_y, new_p))
            sample["event_volume_new"] = \
                self.voxel(EventSequence(new_events,
                                     params={'width': self.image_width,
                                             'height': self.image_height},
                                     timestamp_multiplier=1e6,
                                     convert_to_relative=True,
                                     features=new_events))

            if self.add_eigenvalues or self.add_filter_values:
                x = new_x
                y = new_y
                t = new_t
                p = new_p
                self.get_eigenvalues(x, y, t, p)
                
                if self.add_eigenvalues:
                    eig1 = self.harris_recursive.eig1
                    eig2 = self.harris_recursive.eig2
                    print(f"Eigenvalues computed for index {idx - 1} and sequence {self.seq_name}")
                    print(f"Eigenvalue 1: min. {eig1.min()}, max. {eig1.max()}")
                    print(f"Eigenvalue 2: min. {eig2.min()}, max. {eig2.max()}")
                    eig1_representation = \
                        self.voxel(EventSequence(np.column_stack((t, x, y, eig1)),
                                    params={'width': self.image_width, 
                                            'height': self.image_height},
                                    timestamp_multiplier=1e6,
                                    convert_to_relative=True,
                                    features=new_events))
                    eig2_representation = \
                        self.voxel(EventSequence(np.column_stack((t, x, y, eig2)),
                                    params={'width': self.image_width, 
                                            'height': self.image_height},
                                    timestamp_multiplier=1e6,
                                    convert_to_relative=True,
                                    features=new_events))
                    sample["eigenvalues_volume_new"] = torch.cat(
                        (
                            eig1_representation,
                            eig2_representation,
                        ),
                        dim=0
                    )
                    print(f"Voxel grid representation with eigenvalues for index {idx - 1} and sequence {self.seq_name}")
                    print(f"Eigenvalue 1: min. {eig1_representation.min()}, max. {eig1_representation.max()}")
                    print(f"Eigenvalue 2: min. {eig2_representation.min()}, max. {eig2_representation.max()}")    
            
                if self.add_filter_values:
                    filter_values = self.harris_recursive.filter_value_recursive
                    print(f"Filter values computed for index {idx - 1} and sequence {self.seq_name}")
                    print(f"Filter values: min. {filter_values.min()}, max. {filter_values.max()}")
                    sample["filter_values_volume_new"] = \
                        self.voxel(EventSequence(np.column_stack((t, x, y, filter_values)),
                                    params={'width': self.image_width, 
                                            'height': self.image_height},
                                    timestamp_multiplier=1e6,
                                    convert_to_relative=True,
                                    features=new_events))
                    print(f"Voxel grid representation with filter values for index {idx - 1} and sequence {self.seq_name}")
                    print(f"Filter values: min. {sample['filter_values_volume_new'].min()}, max. {sample['filter_values_volume_new'].max()}")
                    
            # get flow
            flow = np.transpose(self.h5['flow']['dt={}'.format(self.dt)][self.gt_flow[idx]][:], (2, 0, 1))
            flow_next = np.transpose(self.h5['flow']['dt={}'.format(self.dt)][self.gt_flow[idx+1]][:], (2, 0, 1))
        

        sample["flow_gt_event_volume_new"] = self.process_flow_gt(flow)
        sample["flow_gt_next"] = self.process_flow_gt(flow_next)

        sample["event_volume_old"] = self.cropper(sample["event_volume_old"])
        sample["event_volume_new"] = self.cropper(sample["event_volume_new"])
        if self.add_eigenvalues:
            sample["eigenvalues_volume_new"] = self.cropper(sample["eigenvalues_volume_new"])
        if self.add_filter_values:
            sample["filter_values_volume_new"] = self.cropper(sample["filter_values_volume_new"])

        cleaned_output = {
                k: v for k, v in sample.items() if not ("_old" in k or "_next" in k)
            }

        if self.do_not_save_preprocessed:
            return cleaned_output
        torch.save(cleaned_output, preprocessed_file_path)
        print(f"Saved preprocessed data for index {idx - 1} for sequence {self.seq_name} at {preprocessed_file_path}")
        return cleaned_output

    def __getitem__(self, idx):
        if self.in_memory:
            sample = self.data[idx]
        else:
            sample = self.get_data_sample(idx)
        if self.augment:
            # augmentation
            if random.random() > 0.5:
                for key in sample:
                    if isinstance(sample[key], torch.Tensor):
                        sample[key] = T.functional.hflip(sample[key])
                    if key.startswith("flow_gt"):
                        sample[key] = [T.functional.hflip(
                            mask) for mask in sample[key]]
                        sample[key][0][0, :] = -sample[key][0][0, :]
        return sample            



    def process_flow_gt(self, flow):
        flow_valid = (flow[0] != 0) | (flow[1] != 0)
        flow_valid[193:, :] = False
        flow = torch.from_numpy(flow)
        valid_mask = torch.from_numpy(
            np.stack([flow_valid]*1, axis=0))

        return (self.cropper(flow), self.cropper(valid_mask))
        
    def build_event_idx(self):
        if self.dt is None:
            events_ts = self.event_h5['davis']['left']['events'][:, 2]
        else:
            events_ts = self.h5['events']['ts']
        return np.searchsorted(events_ts, self.timestamps, side='left')


class MVSECRecurrent(MVSEC):
    def __init__(self, seq_name, seq_path="/scratch", representation_type=None,
                 rate=20, num_bins=15, transforms=[], filter=None, augment=True, sequence_length=1):
        super(MVSECRecurrent, self).__init__(seq_name, seq_path, representation_type,
                                             rate, num_bins, transforms, filter, augment)
        self.sequence_length = sequence_length
        self.valid_indices = self.get_continuous_sequences()

    def get_continuous_sequences(self):
        # MVSEC is continuous without breaks
        continuous_seq_idcs = list(
            (range(self.raw_gt_len - 2 - self.sequence_length)))
        return continuous_seq_idcs
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < len(self)

        valid_idx = self.valid_indices[idx]
        sequence = []
        j = valid_idx

        for i in range(self.sequence_length):
            sample = super(MVSECRecurrent, self).__getitem__(j)
            sequence.append(sample)
            j += 1

        
        # Check if the current sample is the first sample of a continuous sequence
        if idx == 0 or self.valid_indices[idx]-self.valid_indices[idx-1] != 1:
            sequence[0]['new_sequence'] = 1
        else:
            sequence[0]['new_sequence'] = 0

        return sequence

