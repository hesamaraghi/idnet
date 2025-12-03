"""DSEC dataset format utilities.

This module provides utility functions for working with DSEC-format optical flow
and event data, including encoding/decoding flows and creating HDF5 event files.
"""

import os
import hashlib
import json
import numpy as np
import h5py
import imageio.v2 as imageio


# Default parameters for dataset generation
DEFAULT_TOTAL_FRAMES = 2_000
DEFAULT_FRAME_TIME_US = 1_000  # time between frames in microseconds
DEFAULT_IMAGE_SIZE = (256, 256)  # (H, W)
DEFAULT_SAVE_STEP = 20          # optical flow from frame k to k+save_step
DEFAULT_FLOW_DT_US = DEFAULT_FRAME_TIME_US * DEFAULT_SAVE_STEP  # e.g., 100_000 for 10 Hz
DEFAULT_START_TS_US = 0          # synthetic start timestamp
DEFAULT_SEQ_NAME = "star8"


def generate_dataset_hash(**kwargs):
    """Generate a short hash for dataset configuration to avoid conflicts in parallel generation.
    
    This function takes ALL parameters that affect the dataset generation and creates
    a unique hash. If you add new parameters in the future, they will automatically
    be included in the hash calculation.
    
    Args:
        **kwargs: All dataset generation parameters (arbitrary key-value pairs)
        
    Returns:
        str: 8-character hash of configuration
    """
    # Sort by key to ensure consistent ordering
    config_str = str(sorted(kwargs.items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return config_hash


def load_dataset_metadata(data_root: str, seq_name: str):
    """Load dataset metadata from JSON file.
    
    Args:
        data_root: Root directory (e.g., 'toy_datasets/data/star8')
        seq_name: Sequence name (e.g., 'star8' or 'star8_test')
        
    Returns:
        dict: Metadata dictionary with generation parameters
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path = os.path.join(data_root, "train_optical_flow", seq_name, "dataset_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"This dataset may have been generated before metadata support was added.\n"
            f"Please regenerate the dataset or manually specify parameters."
        )
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def encode_flow_dsec(u: np.ndarray, v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Encode flow into DSEC 16-bit PNG format.

    Channels (uint16):
    - R (0): x-component encoded as round(u*128 + 2^15)
    - G (1): y-component encoded as round(v*128 + 2^15)
    - B (2): valid mask (0 or 1)
    
    Args:
        u: Horizontal flow component (float)
        v: Vertical flow component (float)
        valid: Valid flow mask (bool or int)
        
    Returns:
        Encoded 16-bit RGB image
    """
    assert u.shape == v.shape == valid.shape
    I = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint16)
    enc_u = np.clip(np.round(u * 128.0 + 2**15), 0, 65535).astype(np.uint16)
    enc_v = np.clip(np.round(v * 128.0 + 2**15), 0, 65535).astype(np.uint16)
    I[..., 0] = enc_u
    I[..., 1] = enc_v
    I[..., 2] = valid.astype(np.uint16)
    return I


def decode_flow_dsec(png_path: str):
    """Decode DSEC-format flow PNG back to u, v, valid.
    
    Args:
        png_path: Path to DSEC-format 16-bit PNG file
        
    Returns:
        tuple: (u, v, valid) where u and v are float arrays and valid is bool array
    """
    I = imageio.imread(png_path, format="PNG-FI")
    u = (I[..., 0].astype(np.float32) - 2**15) / 128.0
    v = (I[..., 1].astype(np.float32) - 2**15) / 128.0
    valid = I[..., 2].astype(bool)
    return u, v, valid


def create_dsec_events_h5(events_array, output_path, t_offset=0):
    """Create DSEC-format h5 file from events array.
    
    Args:
        events_array: structured numpy array with fields ('x', 'y', 't', 'p')
                     where t is in microseconds
        output_path: path to save the h5 file
        t_offset: time offset in microseconds to add to event timestamps
    
    DSEC h5 structure:
        /events/p - polarity (0 or 1)
        /events/t - time in microseconds
        /events/x - column index
        /events/y - row index
        /ms_to_idx - mapping from milliseconds to event indices
        /t_offset - time offset in microseconds
    """
    # Extract and sort events by time
    x = events_array['x'].astype(np.int16)
    y = events_array['y'].astype(np.int16)
    t = events_array['t'].astype(np.int64)
    p = events_array['p'].astype(np.uint8)  # Convert bool to uint8 (0 or 1)
    
    # Sort by time
    sort_idx = np.argsort(t)
    x = x[sort_idx]
    y = y[sort_idx]
    t = t[sort_idx]
    p = p[sort_idx]
    
    # Build ms_to_idx mapping
    # ms_to_idx[ms] = index such that t[index] >= ms*1000 and t[index-1] < ms*1000
    # We need one extra entry beyond the last millisecond to allow queries at the boundary
    # E.g., if last event is at 1599000 us (1599 ms), we need ms_to_idx[1600] to be valid
    if len(t) > 0:
        max_time_us = t[-1]
        # Convert to milliseconds and add 2: one for the current ms, one extra for boundary queries
        max_time_ms = int(max_time_us / 1000) + 2
        
        ms_to_idx = np.zeros(max_time_ms, dtype=np.int64)
        
        event_idx = 0
        for ms in range(max_time_ms):
            ms_us = ms * 1000
            # Find first event with t >= ms_us
            while event_idx < len(t) and t[event_idx] < ms_us:
                event_idx += 1
            ms_to_idx[ms] = event_idx
    else:
        ms_to_idx = np.zeros(1, dtype=np.int64)
    
    # Write h5 file
    with h5py.File(output_path, 'w') as f:
        # Create events group
        events_group = f.create_group('events')
        
        # Store event data with compression
        events_group.create_dataset('x', data=x, compression='gzip', compression_opts=9)
        events_group.create_dataset('y', data=y, compression='gzip', compression_opts=9)
        events_group.create_dataset('t', data=t, compression='gzip', compression_opts=9)
        events_group.create_dataset('p', data=p, compression='gzip', compression_opts=9)
        
        # Store ms_to_idx mapping
        f.create_dataset('ms_to_idx', data=ms_to_idx, compression='gzip', compression_opts=9)
        
        # Store time offset
        f.create_dataset('t_offset', data=np.array([t_offset], dtype=np.int64))
    
    print(f"Created DSEC events h5: {output_path}")
    print(f"  Total events: {len(x)}")
    print(f"  Time range: {t[0] if len(t) > 0 else 0} - {t[-1] if len(t) > 0 else 0} us")
    print(f"  ms_to_idx size: {len(ms_to_idx)} ms")
    print(f"  t_offset: {t_offset} us")


def create_identity_rectify_map(image_size: tuple, output_path: str):
    """Create identity rectification map for synthetic dataset.
    
    Since the dataset is synthetic and has no lens distortion,
    the rectify_map is just an identity mapping where each pixel
    maps to itself.
    
    Args:
        image_size: (H, W) tuple
        output_path: path to save rectify_map.h5
    
    Structure:
        /rectify_map - (H, W, 2) array where rectify_map[y, x] = [x, y]
    """
    H, W = image_size
    
    # Create identity mapping: each pixel (x, y) maps to itself
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    rectify_map = np.stack([xx, yy], axis=-1).astype(np.float32)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('rectify_map', data=rectify_map, compression='gzip', compression_opts=9)
    
    print(f"Created identity rectify_map: {output_path}")
    print(f"  Shape: {rectify_map.shape}")
