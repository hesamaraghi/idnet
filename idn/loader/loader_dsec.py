import math
from pathlib import Path, PurePath
from random import sample
import random
from typing import Dict, Tuple
import weakref
from time import time
import cv2
# import h5pickle as h5py
import h5py
from numba import jit
import numpy as np
import os
import imageio
import hashlib
# import mkl
import torch
from torchvision.transforms import ToTensor, RandomCrop
from torchvision import transforms as tf
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt, transforms
from ..utils import transformers
from tqdm import tqdm
import hdf5plugin
from scipy.ndimage import gaussian_filter

from ..utils.dsec_utils import RepresentationType, VoxelGrid, PolarityCount, flow_16bit_to_float
from ..utils.transformers import (
    downsample_spatial,
    downsample_spatial_mask,
    apply_transform_to_field,
    apply_randomcrop_to_sample)

VISU_INDEX = 1


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(
            t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            print(f"t_start_us_idx: {t_start_ms_idx}, t_end_us_idx: {t_end_ms_idx}, "
                  f"t_end_ms: {t_end_ms}, "
                  f"max ts in events: {self.events['t'][-1]}, "
                  f"len(ms_to_idx): {len(self.ms_to_idx)}, "
                  f"max idx in ms_to_idx: {self.ms_to_idx[-1]}, "
                  f"time of max in ms_to_idx: {self.events['t'][self.ms_to_idx[-1]]} ", flush=True)
            return None

        events = dict()
        time_array_conservative = np.asarray(
            self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(
            time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(
                self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0, f"time_ms must be non-negative, got {time_ms}"
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


class HarrisRecursive:

    def __init__(self, tau: float, filter_size: int, image_size: Tuple[int, int]):

        assert filter_size % 2 == 1, "Filter size must be odd"
        print(f"Using tau={tau} and filter_size={filter_size} for Harris Recursive")
        self.tau = tau
        self.filter_size = filter_size
        self.image_size = image_size
        self.K = filter_size // 2
        self.H, self.W = image_size

        sigma = filter_size / 5.0
        self.kernel = np.zeros((filter_size, filter_size))
        self.kernel[filter_size // 2, filter_size // 2] = 1
        self.gaussian_kernel = gaussian_filter(self.kernel, sigma)
        self.gaussian_kernel = self.gaussian_kernel / np.sum(self.gaussian_kernel)

    def __call__(self, data):

        self.last_time_tensor = np.full(
            (2, self.H + 2 * self.K + 2, self.W + 2 * self.K + 2),
            float("-inf"),
            dtype=np.float32,
        )
        self.temporal_accumulation_tensor = np.full(
            (2, self.H + 2 * self.K + 2, self.W + 2 * self.K + 2),
            float("0"),
            dtype=np.float32,
        )

        self.filter_value_recursive = np.zeros(data.shape[0], dtype=np.float32)
        self.temporal_accumulation_tensor_list = []
        self.structure_tensor = np.zeros((3, data.shape[0]), dtype=np.float32)

        # sorted_indices = torch.argsort(data.pos[..., -1])
        # data.pos = data.pos[sorted_indices]
        # data.x = data.x[sorted_indices]

        data["t"] = (data["t"] - data["t"][0]).astype(np.float32)

        for i, ev in enumerate(data):

            # pp = 0 if data.x[i] <= 0 else 1
            temporal_adding_value = -1 if ev["p"] <= 0 else 1
            pp = 0
            # temporal_adding_value = 1; pp = 0
            # temporal_adding_value = 1

            h = ev["y"] + self.K + 1
            w = ev["x"] + self.K + 1
            t = ev["t"]

            h_start = h - self.K
            h_end = h + self.K
            w_start = w - self.K
            w_end = w + self.K

            # Compute the temporal lag
            temporal_lag = np.exp(
                -(
                    t
                    - self.last_time_tensor[
                        pp, h_start - 1 : h_end + 2, w_start - 1 : w_end + 2
                    ]
                )
                / self.tau
            )

            # update the last time tensor
            self.last_time_tensor[
                pp, h_start - 1 : h_end + 2, w_start - 1 : w_end + 2
            ] = t

            # update the temporal accumulation tensor

            self.temporal_accumulation_tensor[
                pp, h_start - 1 : h_end + 2, w_start - 1 : w_end + 2
            ] *= temporal_lag
            self.temporal_accumulation_tensor[pp, h, w] += temporal_adding_value

            self.h_diff_tensor = cv2.Sobel(
                self.temporal_accumulation_tensor[
                    pp, h_start - 1 : h_end + 2, w_start - 1 : w_end + 2
                ],
                -1,
                0,
                1,
                ksize=3,
                scale=1,
                delta=0,
                borderType=cv2.BORDER_DEFAULT,
            )[1:-1, 1:-1]
            self.w_diff_tensor = cv2.Sobel(
                self.temporal_accumulation_tensor[
                    pp, h_start - 1 : h_end + 2, w_start - 1 : w_end + 2
                ],
                -1,
                1,
                0,
                ksize=3,
                scale=1,
                delta=0,
                borderType=cv2.BORDER_DEFAULT,
            )[1:-1, 1:-1]

            self.structure_tensor[0, i] = np.sum(
                self.h_diff_tensor * self.h_diff_tensor * self.gaussian_kernel
            )
            self.structure_tensor[1, i] = np.sum(
                self.w_diff_tensor * self.w_diff_tensor * self.gaussian_kernel
            )
            self.structure_tensor[2, i] = np.sum(
                self.w_diff_tensor * self.h_diff_tensor * self.gaussian_kernel
            )

            self.filter_value_recursive[i] = np.sum(
                self.temporal_accumulation_tensor[
                    pp, h_start : h_end + 1, w_start : w_end + 1
                ]
                * self.gaussian_kernel
            )

        a = self.structure_tensor[0]
        b = self.structure_tensor[1]
        c = self.structure_tensor[2]
        trace = a + b
        det_term = np.sqrt((a - b) ** 2 + 4 * c**2)
        self.eig1 = 0.5 * (trace + det_term)
        self.eig2 = 0.5 * (trace - det_term)


class Sequence(Dataset):

    def __init__(
        self,
        seq_path: Path,
        representation_type: RepresentationType,
        mode: str = "test",
        delta_t_ms: int = 100,
        num_bins: int = 15,
        transforms=[],
        name_idx=0,
        visualize=False,
        load_gt=False,
        add_eigenvalues=False,
        add_filter_values=False,
        tau=15_000,
        filter_size=7,
        in_memory=False,
        force_preprocess=False,
        do_not_save_preprocessed=False,
        normalize_voxel=None,
    ):

        self.add_eigenvalues = add_eigenvalues
        self.add_filter_values = add_filter_values
        self.in_memory = in_memory
        self.force_preprocess = force_preprocess
        self.do_not_save_preprocessed = do_not_save_preprocessed
        
        # Track which samples have been force-preprocessed in this session
        # This allows force_preprocess to overwrite files once, then use them
        self.force_preprocessed_indices = set()

        assert num_bins >= 1
        # assert delta_t_ms == 100
        assert seq_path.is_dir(), f"{seq_path} is not a directory"
        assert mode in {'train', 'test'}
        '''
        Directory Structure:

        Dataset
        └── test
            ├── interlaken_00_b
            │   ├── events_left
            │   │   ├── events.h5
            │   │   └── rectify_map.h5
            │   ├── image_timestamps.txt
            │   └── test_forward_flow_timestamps.csv

        '''
        self.seq_path = PurePath(seq_path)
        self.seq_name = PurePath(seq_path).name
        self.mode = mode
        self.name_idx = name_idx
        self.visualize_samples = visualize
        self.load_gt = load_gt
        self.transforms = transforms

        self.do_not_save_preprocessed = do_not_save_preprocessed
        self.preprocessed_path = Path(self.seq_path / 'preprocessed')
        if not self.preprocessed_path.exists():  
            self.preprocessed_path.mkdir(parents=True, exist_ok=True)

        if self.mode == "test":
            # Get Test Timestamp File
            ev_dir_location = seq_path / 'events_left'
            timestamp_file = seq_path / 'test_forward_flow_timestamps.csv'
            flow_path = seq_path
            timestamps_images = np.loadtxt(
                flow_path / 'image_timestamps.txt', dtype='int64')
            self.indices = np.arange(len(timestamps_images))[::2][1:-1]
            self.timestamps_flow = timestamps_images[::2][1:-1]

        elif self.mode == "train":
            ev_dir_location = seq_path / 'events' / 'left'
            seq_name = seq_path.parts[-1]
            flow_path = seq_path.parents[1] / \
                "train_optical_flow"/seq_name/'flow'
            if not flow_path.is_dir():
                print(f"Flow path {flow_path} does not exist. Cheking the data directory 'data' for the file ...")
                flow_path = Path("data") / "train_optical_flow" / seq_name / 'flow'
                if not flow_path.is_dir():
                    raise FileNotFoundError(
                        f"Flow path {flow_path} does not exist in the data directory.")
                else:
                    print(f"Using flow path {flow_path} from the data directory.")
            timestamp_file = flow_path/'forward_timestamps.txt'
            self.flow_png = [Path(os.path.join(flow_path / 'forward', img)) for img in sorted(
                os.listdir(flow_path / 'forward'))]
            timestamps_images = np.loadtxt(
                flow_path / 'forward_timestamps.txt', delimiter=',', dtype='int64')
            self.indices = np.arange(len(timestamps_images) - 1)
            self.timestamps_flow = timestamps_images[1:, 0]
        else:
            pass
        assert timestamp_file.is_file()

        file = np.genfromtxt(
            timestamp_file,
            delimiter=','
        )

        self.idx_to_visualize = file[:, 2] if file.shape[1] == 3 else []

        # Save output dimensions
        self.height = 256 #TODO: get from config file
        self.width = 256 #TODO: get from config file
        self.num_bins = num_bins

        # Just for now, we always train with num_bins=15
        # assert self.num_bins == 15

        # Set event representation
        self.voxel_grid = None
        if representation_type == RepresentationType.VOXEL:
            normalize_voxel = True if normalize_voxel is None else normalize_voxel
            self.voxel_grid = VoxelGrid(
                (self.num_bins, self.height, self.width), normalize=True)
            self.voxel_grid_aug = VoxelGrid(
                (self.num_bins, self.height, self.width), normalize=normalize_voxel)
        if representation_type == "count":
            self.voxel_grid = "count"
        if representation_type == "pcount":
            self.voxel_grid = PolarityCount((2, self.height, self.width))

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        # Left events only
        ev_data_file = ev_dir_location / 'events.h5'
        ev_rect_file = ev_dir_location / 'rectify_map.h5'

        if not ev_data_file.is_file():
            print(f"Events file {ev_data_file} does not exist. Cheking the data directory 'data' for the file ...")
            # find self.seq_name in the address of the data directory and replace the before ones with data
            parts = ev_data_file.parts
            idx = parts.index(self.seq_name)
            ev_data_file = Path("data") 
            ev_data_file = (ev_data_file / "test") if self.mode == "test" else (ev_data_file / "train_events")
            ev_data_file = Path(ev_data_file, *parts[idx:])
            if ev_data_file.is_file():
                print(f"Found events file at {ev_data_file}")
            else:
                raise FileNotFoundError(
                    f"Events file {ev_data_file} does not exist in the data directory.")
        if not ev_rect_file.is_file():
            print(f"Rectify map file {ev_rect_file} does not exist. Cheking the data directory 'data' for the file ...")
            # find self.seq_name in the address of the data directory and replace the before ones with data
            parts = ev_rect_file.parts
            idx = parts.index(self.seq_name)
            ev_rect_file = Path("data")
            ev_rect_file = (ev_rect_file / "test") if self.mode == "test" else (ev_rect_file / "train_events")
            ev_rect_file = Path(ev_rect_file, *parts[idx:])
            if ev_rect_file.is_file():
                print(f"Found rectify map file at {ev_rect_file}")
            else:
                raise FileNotFoundError(
                    f"Rectify map file {ev_rect_file} does not exist in the data directory.")

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location)

        self.h5rect = h5py.File(str(ev_rect_file), 'r')
        self.rectify_ev_map = self.h5rect['rectify_map'][()]

        if self.add_eigenvalues or self.add_filter_values:
            self.tau = tau
            self.filter_size = filter_size
            self.harris_recursive = HarrisRecursive(
                tau=self.tau,
                filter_size=self.filter_size,
                image_size=(self.height, self.width),
            )

        if self.in_memory:
            self.data = []
            print(f"Loading data for sequence {self.seq_name} into memory...")
            for i in tqdm(range(len(self))):
                self.data.append(self.get_data_sample(i))

    def events_to_voxel_grid(self, p, t, x, y, device: str = 'cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    def value_to_voxel_grid(self, val, t, x, y, device: str = 'cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        val = val.astype('float32')
        event_data_torch = {
            'val': torch.from_numpy(val),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid_aug.convert(event_data_torch, val_type='val')

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.timestamps_flow) - 1

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (
            self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

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

    def get_data_sample(self, index, crop_window=None, flip=None):
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        names = ['event_volume_old', 'event_volume_new']
        eigenvalues_names = ['eigenvalues_volume_old', 'eigenvalues_volume_new']
        filter_values_names = ['filter_values_volume_old', 'filter_values_volume_new']
        preprocessed_file_path = self.preprocessed_path / f"{index:05d}.pt"
        
        # Determine whether to load from disk or preprocess
        # Logic:
        # 1. If force_preprocess=True and do_not_save_preprocessed=True: Always preprocess, never save/load
        # 2. If force_preprocess=True and do_not_save_preprocessed=False: 
        #    - First time seeing this index: Preprocess and overwrite existing file
        #    - Subsequent times: Load from the newly saved file
        # 3. If force_preprocess=False: Use saved file if exists, otherwise preprocess and save
        should_load_from_disk = False
        
        if self.force_preprocess and self.do_not_save_preprocessed:
            # Always preprocess from scratch, never use saved files
            should_load_from_disk = False
        elif self.force_preprocess and not self.do_not_save_preprocessed:
            # Force preprocess mode: overwrite once, then use saved files
            if index in self.force_preprocessed_indices:
                # We've already preprocessed this index in this session, use the saved file
                should_load_from_disk = True
            else:
                # First time seeing this index, force preprocess and overwrite
                should_load_from_disk = False
                # Mark this index as force-preprocessed
                self.force_preprocessed_indices.add(index)
        elif preprocessed_file_path.exists():
            # Normal mode: file exists, load it
            should_load_from_disk = True
        else:
            # Normal mode: file doesn't exist, need to preprocess
            should_load_from_disk = False
        
        if should_load_from_disk:
            # print(f"Loading preprocessed data for index {index} for sequence {self.seq_name} from {preprocessed_file_path}")
            loaded_file = torch.load(preprocessed_file_path, weights_only=False)       
            if not self.add_eigenvalues and not self.add_filter_values:
                loaded_file['event_volume_new'] = loaded_file['event_volume_new'][:self.num_bins,:,:]
            return loaded_file
        else:
            ts_start = [self.timestamps_flow[index] -
                        self.delta_t_us, self.timestamps_flow[index]]
            ts_end = [self.timestamps_flow[index],
                    self.timestamps_flow[index] + self.delta_t_us]

            file_index = self.indices[index]

            output = {
                'file_index': file_index,
                'timestamp': self.timestamps_flow[index],
                'seq_name': self.seq_name
            }
            # Save sample for benchmark submission
            output['save_submission'] = file_index in self.idx_to_visualize
            output['visualize'] = self.visualize_samples

            for i in range(len(names)):
                event_data = self.event_slicer.get_events(
                    ts_start[i], ts_end[i])

                if event_data is None:
                    raise ValueError(
                        f"i is {i}. "
                        f"Failed to get events for time window [{ts_start[i]}, {ts_end[i]}] us. "
                        f"This usually means the requested time window is outside the available data range. "
                        f"Sequence: {self.seq_name}, Index: {index}, "
                        f"Time offset: {self.event_slicer.t_offset}, "
                        f"Available time range: {self.event_slicer.events['t'][0]} to {self.event_slicer.events['t'][-1]} us"
                    )

                p = event_data['p']
                t = event_data['t']
                x = event_data['x']
                y = event_data['y']

                if self.add_eigenvalues or self.add_filter_values:
                    self.get_eigenvalues(x, y, t, p)
                    if self.add_eigenvalues:
                        eig1 = self.harris_recursive.eig1
                        eig2 = self.harris_recursive.eig2
                        print(f"Eigenvalues computed for index {index} and sequence {self.seq_name}")
                        print(f"Eigenvalue 1: min. {eig1.min()}, max. {eig1.max()}")
                        print(f"Eigenvalue 2: min. {eig2.min()}, max. {eig2.max()}")
                    if self.add_filter_values:
                        filter_values = self.harris_recursive.filter_value_recursive
                        print(f"Filter values computed for index {index} and sequence {self.seq_name}")
                        print(f"Filter values: min. {filter_values.min()}, max. {filter_values.max()}")
                xy_rect = self.rectify_events(x, y)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                if crop_window is not None:
                    # Cropping (+- 2 for safety reasons)
                    x_mask = (x_rect >= crop_window['start_x']-2) & (
                        x_rect < crop_window['start_x']+crop_window['crop_width']+2)
                    y_mask = (y_rect >= crop_window['start_y']-2) & (
                        y_rect < crop_window['start_y']+crop_window['crop_height']+2)
                    mask_combined = x_mask & y_mask
                    p = p[mask_combined]
                    t = t[mask_combined]
                    x_rect = x_rect[mask_combined]
                    y_rect = y_rect[mask_combined]
                    if self.add_eigenvalues:
                        eig1 = eig1[mask_combined]
                        eig2 = eig2[mask_combined]
                    if self.add_filter_values:
                        filter_values = filter_values[mask_combined]
                if self.voxel_grid is None:
                    raise NotImplementedError
                else:
                    event_representation = self.events_to_voxel_grid(
                        p, t, x_rect, y_rect)
                    output[names[i]] = event_representation
                    if self.add_eigenvalues:
                        eig1_representation = self.value_to_voxel_grid(
                            eig1, t, x_rect, y_rect
                        )
                        eig2_representation = self.value_to_voxel_grid(
                            eig2, t, x_rect, y_rect
                        )
                        # Add eigenvalues to the output
                        output[eigenvalues_names[i]] = torch.cat(
                            (
                                eig1_representation,
                                eig2_representation,
                            ),
                            dim=0,
                        )
                        print(f"Voxel grid representation with eigenvalues for index {index} and sequence {self.seq_name}")
                        print(f"Eigenvalue 1: min. {eig1_representation.min()}, max. {eig1_representation.max()}")
                        print(f"Eigenvalue 2: min. {eig2_representation.min()}, max. {eig2_representation.max()}")   
                    if self.add_filter_values:
                        filter_values_representation = self.value_to_voxel_grid(
                            filter_values, t, x_rect, y_rect
                        )
                        # Add filter values to the output
                        output[filter_values_names[i]] = filter_values_representation
                        print(f"Voxel grid representation with filter values for index {index} and sequence {self.seq_name}")
                        print(f"Filter values: min. {filter_values_representation.min()}, max. {filter_values_representation.max()}")
                output['name_map'] = self.name_idx

                if self.load_gt:
                    output['flow_gt_' + names[i]
                        ] = [torch.tensor(x) for x in self.load_flow(self.flow_png[index + i])]

                    output['flow_gt_' + names[i]
                        ][0] = torch.moveaxis(output['flow_gt_' + names[i]][0], -1, 0)
                    output['flow_gt_' + names[i]
                        ][1] = torch.unsqueeze(output['flow_gt_' + names[i]][1], 0)

            if self.load_gt:
                if index + 2 < len(self.flow_png):
                    output['flow_gt_next'] = [torch.tensor(
                        x) for x in self.load_flow(self.flow_png[index + 2])]
                    output['flow_gt_next'][0] = torch.moveaxis(
                        output['flow_gt_next'][0], -1, 0)
                    output['flow_gt_next'][1] = torch.unsqueeze(
                        output['flow_gt_next'][1], 0)

            cleaned_output = {
                k: v for k, v in output.items() if not ("_old" in k or "_next" in k)
            }
            if self.do_not_save_preprocessed:
                return cleaned_output
            torch.save(cleaned_output, preprocessed_file_path)
            print(f"Saved preprocessed data for index {index} for sequence {self.seq_name} at {preprocessed_file_path}")
            return cleaned_output

    def __getitem__(self, idx):
        if self.in_memory:
            sample = self.data[idx]
        else:
            sample = self.get_data_sample(idx)
        for key_t, transform in self.transforms.items():
            if key_t == "hflip":
                if random.random() > 0.5:
                    for key in sample:
                        if isinstance(sample[key], torch.Tensor):
                            sample[key] = tf.functional.hflip(sample[key])
                        if key.startswith("flow_gt"):
                            sample[key] = [tf.functional.hflip(
                                mask) for mask in sample[key]]
                            sample[key][0][0, :] = -sample[key][0][0, :]
            elif key_t == "vflip":
                if random.random() < transform:
                    for key in sample:
                        if isinstance(sample[key], torch.Tensor):
                            sample[key] = tf.functional.vflip(sample[key])
                        if key.startswith("flow_gt"):
                            sample[key] = [tf.functional.vflip(
                                mask) for mask in sample[key]]
                            sample[key][0][1, :] = -sample[key][0][1, :]
            elif key_t == "randomcrop":
                apply_randomcrop_to_sample(sample, crop_size=transform)
            else:
                apply_transform_to_field(sample, transform, key_t)

        return sample

    def get_voxel_grid(self, idx):

        if idx == 0:
            event_data = self.event_slicer.get_events(
                self.timestamps_flow[0] - self.delta_t_us, self.timestamps_flow[0])
        elif idx > 0 and idx <= self.__len__():
            event_data = self.event_slicer.get_events(
                self.timestamps_flow[idx-1], self.timestamps_flow[idx-1] + self.delta_t_us)
        else:
            raise IndexError

        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']

        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]
        return self.events_to_voxel_grid(p, t, x_rect, y_rect)

    def get_event_count_image(self, ts_start, ts_end, num_bins=15, normalize=True):
        assert ts_end > ts_start
        delta_t_bin = (ts_end - ts_start) / num_bins
        ts_start_bin = np.linspace(
            ts_start, ts_end, num=num_bins, endpoint=False)
        ts_end_bin = ts_start_bin + delta_t_bin
        assert abs(ts_end_bin[-1] - ts_end) < 10.
        ts_end_bin[-1] = ts_end

        event_count = torch.zeros(
            (num_bins, self.height, self.width), dtype=torch.float, requires_grad=False)

        for i in range(num_bins):
            event_data = self.event_slicer.get_events(
                ts_start_bin[i], ts_end_bin[i])
            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            t = (t - t[0]).astype('float32')
            t = (t/t[-1])
            x = x.astype('float32')
            y = y.astype('float32')
            pol = p.astype('float32')
            event_data_torch = {
                'p': torch.from_numpy(pol),
                't': torch.from_numpy(t),
                'x': torch.from_numpy(x),
                'y': torch.from_numpy(y),
            }
            x = event_data_torch['x']
            y = event_data_torch['y']
            xy_rect = self.rectify_events(x.int(), y.int())
            x_rect = torch.from_numpy(xy_rect[:, 0]).long()
            y_rect = torch.from_numpy(xy_rect[:, 1]).long()
            value = 2*event_data_torch['p']-1
            index = self.width*y_rect + x_rect
            mask = (x_rect < self.width) & (y_rect < self.height)
            event_count[i].put_(index[mask], value[mask], accumulate=True)

        return event_count

    @staticmethod
    def normalize_tensor(event_count):
        mask = torch.nonzero(event_count, as_tuple=True)
        if mask[0].size()[0] > 0:
            mean = event_count[mask].mean()
            std = event_count[mask].std()
            if std > 0:
                event_count[mask] = (event_count[mask] - mean) / std
            else:
                event_count[mask] = event_count[mask] - mean
        return event_count


class SequenceRecurrent(Sequence):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str = 'test', delta_t_ms: int = 100,
                 num_bins: int = 15, transforms=None, sequence_length=1, name_idx=0, visualize=False, load_gt=False):
        super(SequenceRecurrent, self).__init__(seq_path, representation_type, mode, delta_t_ms, transforms=transforms,
                                                name_idx=name_idx, visualize=visualize, load_gt=load_gt)
        self.crop_size = self.transforms['randomcrop'] if 'randomcrop' in self.transforms else None
        self.sequence_length = sequence_length
        self.valid_indices = self.get_continuous_sequences()

    def get_continuous_sequences(self):
        continuous_seq_idcs = []
        if self.sequence_length > 1:
            for i in range(len(self.timestamps_flow)-self.sequence_length+1):
                diff = self.timestamps_flow[i +
                                            self.sequence_length-1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length-1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        else:
            for i in range(len(self.timestamps_flow)-1):
                diff = self.timestamps_flow[i+1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length-1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        return continuous_seq_idcs

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < len(self)

        # Valid index is the actual index we want to load, which guarantees a continuous sequence length
        valid_idx = self.valid_indices[idx]

        sequence = []
        j = valid_idx

        ts_cur = self.timestamps_flow[j]
        # Add first sample
        sample = self.get_data_sample(j)
        sequence.append(sample)

        # Data augmentation according to first sample
        crop_window = None
        flip = None
        if 'crop_window' in sample.keys():
            crop_window = sample['crop_window']
        if 'flipped' in sample.keys():
            flip = sample['flipped']

        for i in range(self.sequence_length-1):
            j += 1
            ts_old = ts_cur
            ts_cur = self.timestamps_flow[j]
            assert(ts_cur-ts_old < 100000 + 1000)
            sample = self.get_data_sample(
                j, crop_window=crop_window, flip=flip)
            sequence.append(sample)

        # Check if the current sample is the first sample of a continuous sequence
        if idx == 0 or self.valid_indices[idx]-self.valid_indices[idx-1] != 1:
            sequence[0]['new_sequence'] = 1
            print("Timestamp {} is the first one of the next seq!".format(
                self.timestamps_flow[self.valid_indices[idx]]))
        else:
            sequence[0]['new_sequence'] = 0

        # random crop
        if self.crop_size is not None:
            i, j, h, w = RandomCrop.get_params(
                sample["event_volume_new"], output_size=self.crop_size)
            keys_to_crop = [
                # "event_volume_old",
                "event_volume_new",
                # "eigenvalues_volume_old",
                "eigenvalues_volume_new",
                # "filter_values_volume_old",
                "filter_values_volume_new",
                # "flow_gt_event_volume_old",
                "flow_gt_event_volume_new",
                # "flow_gt_next",
            ]

            for sample in sequence:
                for key, value in sample.items():
                    if key in keys_to_crop:
                        if isinstance(value, torch.Tensor):
                            sample[key] = tf.functional.crop(value, i, j, h, w)
                        elif isinstance(value, list) or isinstance(value, tuple):
                            sample[key] = [tf.functional.crop(
                                v, i, j, h, w) for v in value]
        return sequence


class DatasetProvider:
    def __init__(self, dataset_path: Path, representation_type: RepresentationType, delta_t_ms: int = 100, num_bins=15,
                 type='standard', config=None, visualize=False):
        test_path = dataset_path / 'test'
        assert dataset_path.is_dir(), str(dataset_path)
        assert test_path.is_dir(), str(test_path)
        assert delta_t_ms == 100
        self.config = config
        self.name_mapper_test = []

        test_sequences = list()
        for child in test_path.iterdir():
            self.name_mapper_test.append(str(child).split("/")[-1])
            if type == 'standard':
                test_sequences.append(Sequence(child, representation_type, 'test', delta_t_ms, num_bins,
                                               transforms=[],
                                               name_idx=len(
                                                   self.name_mapper_test)-1,
                                               visualize=visualize))
            elif type == 'warm_start':
                test_sequences.append(SequenceRecurrent(child, representation_type, 'test', delta_t_ms, num_bins,
                                                        transforms=[], sequence_length=1,
                                                        name_idx=len(
                                                            self.name_mapper_test)-1,
                                                        visualize=visualize))
            else:
                raise Exception(
                    'Please provide a valid subtype [standard/warm_start] in config file!')

        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

    def get_test_dataset(self):
        return self.test_dataset

    def get_name_mapping_test(self):
        return self.name_mapper_test

    def summary(self, logger):
        logger.write_line(
            "================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__, True)
        logger.write_line("Number of Voxel Bins: {}".format(
            self.test_dataset.datasets[0].num_bins), True)


def assemble_dsec_sequences(dataset_root, include_seq=None, exclude_seq=None, require_gt=True, config=None, representation_type="voxel", num_bins=None):
    if representation_type is None:
        representation_type = "voxel"
    representation_type = RepresentationType.VOXEL if representation_type == "voxel" else representation_type
    event_root = os.path.join(dataset_root, "train_events")
    flow_gt_root = os.path.join(dataset_root, "train_optical_flow")
    
    if not os.path.exists(flow_gt_root):
        print(f"Events file {flow_gt_root} does not exist. Cheking the data directory 'data' for the file ...")
        # find self.seq_name in the address of the data directory and replace the before ones with data
        flow_gt_root = Path("data") 
        flow_gt_root = flow_gt_root / "train_optical_flow"
    
    available_seqs = os.listdir(
        flow_gt_root) if require_gt else os.listdir(event_root)

    seqs = available_seqs
    if include_seq:
        seqs = [seq for seq in seqs if seq in include_seq]
    if exclude_seq:
        seqs = [seq for seq in seqs if seq not in exclude_seq]
    # Prepare transform list
    transforms = dict()
    if config.downsample_ratio > 1:
        transforms['(?<!flow_gt_)event_volume'] = lambda sample: downsample_spatial(
            sample, config.downsample_ratio)
        transforms['flow_gt'] = lambda sample: [downsample_spatial(
            sample[0], config.downsample_ratio) / config.downsample_ratio, downsample_spatial_mask(sample[1], config.downsample_ratio)]
        # TODO: Flow gt mask handle downsample on flags
    if config.get("horizontal_flip", None):
        p_hflip = config.horizontal_flip
        assert p_hflip >= 0 and p_hflip <= 1
        #from torchvision.transforms import RandomHorizontalFlip
        # ignore probability of hflip for now, perform when 'hflip' key exists in transforms dict
        transforms['hflip'] = None
    if config.get("vertical_flip", None):
        p_vflip = config.vertical_flip
        assert p_vflip >= 0 and p_vflip <= 1
        transforms['vflip'] = p_vflip
    if config.get("random_crop", None):
        crop_size = config.random_crop
        transforms['randomcrop'] = crop_size

    seq_dataset = []
    for seq in seqs:
        dataset_cls = SequenceRecurrent if hasattr(
            config, "recurrent") and config.recurrent else Sequence
        extra_arg = dict(
            sequence_length=config.sequence_length) if dataset_cls == SequenceRecurrent else dict(        
                num_bins=num_bins,
                delta_t_ms=config.delta_t_ms if getattr(config, "delta_t_ms", None) is not None else 100,
                add_eigenvalues=config.add_eigenvalues, 
                add_filter_values=config.add_filter_values,
                tau=config.tau,
                filter_size=config.filter_size,
                in_memory=config.in_memory,
                force_preprocess=config.force_preprocess,
                do_not_save_preprocessed=config.do_not_save_preprocessed,
                normalize_voxel=config.get("normalize_voxel", None),
            )
        seq_dataset.append(dataset_cls(Path(event_root) / seq,
                           representation_type=representation_type, mode="train",
                           load_gt=require_gt, transforms=transforms, **extra_arg))
    if config.get("concat_seq", True):
        return torch.utils.data.ConcatDataset(seq_dataset)
    else:
        return seq_dataset


def assemble_dsec_test_set(test_set_root, seq_len=None, concat_seq=False, config=None, representation_type=None):
    if representation_type is None:
        representation_type = RepresentationType.VOXEL
        print("dsec test uses representation: voxel")
    elif representation_type == "voxel":
        representation_type = RepresentationType.VOXEL
        print("dsec test uses representation: voxel")
    else:
        print("dsec test uses representation: {}".format(representation_type))
        representation_type = representation_type
    test_seqs = os.listdir(test_set_root)
    seqs = []

    transforms = dict()
    for seq in test_seqs:
        dataset_cls = SequenceRecurrent if seq_len else Sequence
        extra_arg = dict(
            sequence_length=seq_len) if dataset_cls == SequenceRecurrent else dict(
                # num_bins=num_bins, #TODO: num_bins handling
                delta_t_ms=config.delta_t_ms if getattr(config, "delta_t_ms", None) is not None else 100,
                add_eigenvalues=config.get("add_eigenvalues", False),
                add_filter_values=config.get("add_filter_values", False),
                tau=config.get("tau", None),
                filter_size=config.get("filter_size", None),
                in_memory=config.in_memory,
                force_preprocess=config.force_preprocess,
                do_not_save_preprocessed=config.do_not_save_preprocessed,
                normalize_voxel=config.get("normalize_voxel", None),
            )
        seqs.append(dataset_cls(Path(test_set_root) / seq,
                                representation_type, mode='test',
                                load_gt=False, transforms=transforms, **extra_arg))
    if concat_seq:
        return torch.utils.data.ConcatDataset(seqs)
    else:
        return seqs


def assemble_dsec_train_set(train_set_root, flow_gt_root=None, exclude_seq=None, args=None):
    train_seqs = os.listdir(
        flow_gt_root) if flow_gt_root is not None else os.listdir(train_set_root)
    if exclude_seq is not None:
        train_seqs = [seq for seq in train_seqs if seq not in exclude_seq]
    seq_dataset = []
    for seq in train_seqs:
        seq_dataset.append(Sequence(Path(train_set_root) / seq,
                           RepresentationType.VOXEL, mode='train',
                           transforms=ToTensor))
    return torch.utils.data.ConcatDataset(seq_dataset)


def train_collate(sample_list):
    batch = dict()
    for field_name in sample_list[0]:
        if field_name == 'seq_name':
            batch['seq_name'] = [sample[field_name] for sample in sample_list]
        if field_name == 'new_sequence':
            batch['new_sequence'] = [sample[field_name]
                                     for sample in sample_list]
        if (
            field_name.startswith("event_volume")
            or field_name.startswith("eigenvalues_volume")
            or field_name.startswith("filter_values_volume")
        ):
            try:
                batch[field_name] = torch.stack(
                    [sample[field_name] for sample in sample_list]
                )
            except Exception as e:
                print(f"Error stacking {field_name}: {e}")
                for sample in sample_list:
                    print(f"Sample {sample['file_index']} - seq_name: {sample['seq_name']}")
                    for k,v in sample.items():
                        print(f"{k}: {type(v)}")
                raise e
        if field_name.startswith("flow_gt"):
            if all(field_name in x for x in sample_list):
                batch[field_name] = torch.stack(
                    [sample[field_name][0] for sample in sample_list])
                batch[field_name + '_valid_mask'] = torch.stack(
                    [sample[field_name][1] for sample in sample_list])

    return batch


def rec_train_collate(sample_list):
    seq_length = len(sample_list[0])
    seq_of_batch = []
    for i in range(seq_length):
        seq_of_batch.append(train_collate(
            [sample[i] for sample in sample_list]))
    return seq_of_batch
