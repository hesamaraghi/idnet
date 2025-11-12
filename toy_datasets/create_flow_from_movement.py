import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ensure repo root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("/data/idnet")

from star8 import StarMovement  # StarMovement class (same API as ShapeMovementBase)

# DSEC-format writing
import imageio.v2 as imageio
import h5py

DEFAULT_OUTDIR = "/data/idnet/toy_datasets/data/star8"

# default parameters (can be overridden via CLI)
DEFAULT_TOTAL_FRAMES = 2_000
DEFAULT_FRAME_TIME_US = 1_000  # time between frames in microseconds
DEFAULT_IMAGE_SIZE = (256, 256)  # (H, W)
DEFAULT_SAVE_STEP = 40          # optical flow from frame k to k+save_step
DEFAULT_FLOW_DT_US = DEFAULT_FRAME_TIME_US * DEFAULT_SAVE_STEP  # e.g., 100_000 for 10 Hz
DEFAULT_START_TS_US = 0          # synthetic start timestamp
DEFAULT_SEQ_NAME = "star8"


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
    import json
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
    """Decode DSEC-format flow PNG back to u, v, valid."""
    I = imageio.imread(png_path, format="PNG-FI")
    u = (I[..., 0].astype(np.float32) - 2**15) / 128.0
    v = (I[..., 1].astype(np.float32) - 2**15) / 128.0
    valid = I[..., 2].astype(bool)
    return u, v, valid


def get_vertices(star: StarMovement, frame_idx: int):
    """Get star vertices at a specific frame."""
    star.update_shape(frame_idx)
    return star.transformed_path.vertices


def generate_sanity_check_figures(
    seq_name: str,
    outdir: str,
    image_size: tuple,
    total_frames: int,
    save_step: int,
    test_size: float,
    split: str,
    face_color: str = "black",
):
    """Generate sanity check figures by loading the saved flow PNGs.
    
    This verifies the saved ground truth flows by:
    1. Loading the GT flows from the saved PNG files
    2. Visualizing start/end vertices and flow arrows
    
    Args:
        seq_name: Sequence name (e.g., 'star8' or 'star8_test')
        outdir: Root output directory
        image_size: (H, W) tuple
        total_frames: Total frames in the sequence
        save_step: Frame step used for flow generation
        test_size: Test split size (to compute split_start_frame)
        split: 'train' or 'test'
        face_color: Star face color
    """
    assert split in ("train", "test")
    
    H, W = image_size
    
    # Compute split info
    test_size = float(test_size)
    split_start_frame = int(np.floor(total_frames * (1.0 - test_size))) if test_size > 0 else total_frames
    
    # Paths
    flow_root = os.path.join(outdir, "train_optical_flow", seq_name, "flow")
    forward_dir = os.path.join(flow_root, "forward")
    sanity_dir = os.path.join(flow_root, "sanity")
    
    if not os.path.exists(forward_dir):
        print(f"[skip sanity] No forward flow directory for {seq_name}: {forward_dir}")
        return 0
    
    os.makedirs(sanity_dir, exist_ok=True)
    
    # Instantiate star movement (same parameters as generation)
    star = StarMovement(total_frames=total_frames, image_size=image_size, face_color=face_color)
    
    # Get list of PNG files
    png_files = sorted([f for f in os.listdir(forward_dir) if f.endswith('.png')])
    
    # Determine base frame
    if split == "train":
        base_from = 0
    else:
        base_from = split_start_frame
    
    saved = 0
    for idx, png_file in enumerate(png_files):
        frame_from = base_from + idx * save_step
        frame_to = frame_from + save_step
        if frame_to > total_frames:
            break
        
        # Get vertices at start and end
        verts_start = get_vertices(star, frame_from)
        verts_end = get_vertices(star, frame_to)
        
        # Load flow PNG (this is the GT we want to verify)
        flow_png = os.path.join(forward_dir, png_file)
        u, v, valid = decode_flow_dsec(flow_png)
        
        # Prepare plot
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"{seq_name} | pair {idx:06d} | frames {frame_from}->{frame_to}")
        
        # Plot start vertices (blue) and connect them
        if len(verts_start) > 0:
            ax.scatter(verts_start[:, 0], verts_start[:, 1], c='blue', s=20, label='start vertices', zorder=5)
            vs = verts_start
            if not np.allclose(vs[0], vs[-1]):
                vs = np.vstack([vs, vs[:1]])
            ax.plot(vs[:, 0], vs[:, 1], '-', color='blue', alpha=0.6, linewidth=1.5, zorder=4)
        
        # Plot end vertices (red) and connect them
        if len(verts_end) > 0:
            ax.scatter(verts_end[:, 0], verts_end[:, 1], c='red', s=20, label='end vertices', zorder=5)
            ve = verts_end
            if not np.allclose(ve[0], ve[-1]):
                ve = np.vstack([ve, ve[:1]])
            ax.plot(ve[:, 0], ve[:, 1], '-', color='red', alpha=0.6, linewidth=1.5, zorder=4)
        
        # Draw GT flow arrows at vertices (green)
        # If a vertex has no valid flow at its pixel, use nearest valid flow
        arrow_count = 0
        vy_valid, vx_valid = np.where(valid)
        has_any_valid = len(vx_valid) > 0
        
        for vx, vy in verts_start:
            px, py = int(round(vx)), int(round(vy))
            du, dv = None, None
            
            if 0 <= px < W and 0 <= py < H and valid[py, px]:
                du = u[py, px]
                dv = v[py, px]
            elif has_any_valid:
                # Nearest neighbor search among valid pixels
                dx = vx_valid.astype(np.float32) - float(px)
                dy = vy_valid.astype(np.float32) - float(py)
                idx_min = np.argmin(dx * dx + dy * dy)
                py_n = int(vy_valid[idx_min])
                px_n = int(vx_valid[idx_min])
                du = u[py_n, px_n]
                dv = v[py_n, px_n]
            
            # Draw arrow if we found a vector
            if du is not None and dv is not None:
                ax.arrow(vx, vy, du, dv, color='green', width=0.5, head_width=3.0, 
                        head_length=4.0, length_includes_head=True, alpha=0.9, zorder=6)
                arrow_count += 1
        
        ax.legend(loc='upper right', fontsize=8)
        
        out_path = os.path.join(sanity_dir, f"sanity_{idx:06d}.png")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved += 1
        print(f"[sanity] Saved {out_path} (arrows: {arrow_count})")
    
    return saved


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
    if len(t) > 0:
        max_time_us = t[-1]
        max_time_ms = int(np.ceil(max_time_us / 1000.0)) + 1
        
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


def build_star8_flow_and_events(
    seq_name: str = DEFAULT_SEQ_NAME,
    total_frames: int = DEFAULT_TOTAL_FRAMES,
    image_size: tuple = DEFAULT_IMAGE_SIZE,
    save_step: int = DEFAULT_SAVE_STEP,
    frame_time_us: int = DEFAULT_FRAME_TIME_US,
    outdir: str = DEFAULT_OUTDIR,
    start_ts_us: int = DEFAULT_START_TS_US,
    flow_dt_us: int = DEFAULT_FLOW_DT_US,
    face_color: str = "black",
    test_size: float = 0.2,
) -> dict:
    """Generate synthetic star movement optical flow + events in DSEC layout.

    Returns dict with important output paths.
    """
    # Resolve split
    test_size = float(test_size)
    if not (0.0 <= test_size < 1.0):
        raise ValueError("test_size must be in [0, 1)")
    split_start_frame = int(np.floor(total_frames * (1.0 - test_size))) if test_size > 0 else total_frames
    split_start_ts_us = split_start_frame * frame_time_us

    # Output dirs for train and test
    flow_dir_train = os.path.join(outdir, "train_optical_flow", seq_name, "flow", "forward")
    ts_path_train = os.path.join(outdir, "train_optical_flow", seq_name, "flow", "forward_timestamps.txt")
    event_dir_train = os.path.join(outdir, "train_events", seq_name, "events", "left")
    
    # Clean up existing dataset directories to prevent mixture of old and new data
    seq_flow_root = os.path.join(outdir, "train_optical_flow", seq_name)
    seq_event_root = os.path.join(outdir, "train_events", seq_name)
    
    for path in [seq_flow_root, seq_event_root]:
        if os.path.exists(path):
            print(f"[cleanup] Removing existing data: {path}")
            import shutil
            shutil.rmtree(path)
    
    # Also clean up test sequence directories if test_size > 0
    if test_size > 0:
        seq_name_test = f"{seq_name}_test"
        seq_flow_root_test = os.path.join(outdir, "train_optical_flow", seq_name_test)
        seq_event_root_test = os.path.join(outdir, "train_events", seq_name_test)
        
        for path in [seq_flow_root_test, seq_event_root_test]:
            if os.path.exists(path):
                print(f"[cleanup] Removing existing test data: {path}")
                import shutil
                shutil.rmtree(path)

    seq_name_test = f"{seq_name}_test" if test_size > 0 else None
    flow_dir_test = os.path.join(outdir, "train_optical_flow", seq_name_test, "flow", "forward") if seq_name_test else None
    ts_path_test = os.path.join(outdir, "train_optical_flow", seq_name_test, "flow", "forward_timestamps.txt") if seq_name_test else None
    event_dir_test = os.path.join(outdir, "train_events", seq_name_test, "events", "left") if seq_name_test else None

    os.makedirs(flow_dir_train, exist_ok=True)
    os.makedirs(event_dir_train, exist_ok=True)
    if seq_name_test:
        os.makedirs(flow_dir_test, exist_ok=True)
        os.makedirs(event_dir_test, exist_ok=True)

    # Instantiate movement
    star = StarMovement(
        total_frames=total_frames, image_size=image_size, face_color=face_color
    )

    print("Generating events...")
    events = star.generate_events()
    # scale frame index timestamps to microseconds
    events['t'] = events['t'] * frame_time_us

    # Split events into train/test according to time threshold
    if test_size > 0:
        train_mask = events['t'] < split_start_ts_us
        test_mask = ~train_mask
        events_train = events[train_mask]
        events_test = events[test_mask].copy()
        # Rebase test timestamps to start from 0
        if len(events_test) > 0:
            events_test['t'] -= events_test['t'].min()
    else:
        events_train = events
        events_test = None

    events_h5_train = os.path.join(event_dir_train, "events.h5")
    create_dsec_events_h5(events_train, events_h5_train, t_offset=0)
    rectify_map_train = os.path.join(event_dir_train, "rectify_map.h5")
    create_identity_rectify_map(image_size, rectify_map_train)
    
    events_h5_test = None
    rectify_map_test = None
    if events_test is not None and seq_name_test:
        events_h5_test = os.path.join(event_dir_test, "events.h5")
        create_dsec_events_h5(events_test, events_h5_test, t_offset=0)
        rectify_map_test = os.path.join(event_dir_test, "rectify_map.h5")
        create_identity_rectify_map(image_size, rectify_map_test)

    H, W = image_size
    rows = np.arange(H)
    cols = np.arange(W)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    all_coords = np.vstack([cc.ravel(), rr.ravel()]).T.astype(float)

    # Generate flows for train split
    ts_rows_train = []
    idx_train = 0
    for frame_from in range(0, split_start_frame, save_step):
        frame_to = frame_from + save_step
        if frame_to > split_start_frame:
            break
        flows = star.compute_optical_flow_between_frames(all_coords, frame_from, frame_to)
        if flows is None:
            raise RuntimeError("compute_optical_flow_between_frames returned None")
        u = np.zeros((H, W), dtype=np.float32)
        v = np.zeros((H, W), dtype=np.float32)
        valid = np.zeros((H, W), dtype=np.uint8)
        is_valid = ~np.isnan(flows[:, 0])
        coords_valid = all_coords[is_valid]
        disp = flows[is_valid]
        if coords_valid.size > 0:
            x = coords_valid[:, 0].astype(int)
            y = coords_valid[:, 1].astype(int)
            in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            x = x[in_bounds]
            y = y[in_bounds]
            d = disp[in_bounds]
            u[y, x] = d[:, 0].astype(np.float32)
            v[y, x] = d[:, 1].astype(np.float32)
            valid[y, x] = 1
        img16 = encode_flow_dsec(u, v, valid)
        png_path = os.path.join(flow_dir_train, f"{idx_train:06d}.png")
        imageio.imwrite(png_path, img16, format="PNG-FI")
        from_ts = 0 + idx_train * flow_dt_us
        to_ts = from_ts + flow_dt_us
        ts_rows_train.append((from_ts, to_ts))
        print(f"[train] Saved flow #{idx_train} : frame {frame_from} -> {frame_to} (valid: {int(valid.sum())} pixels)")
        idx_train += 1

    os.makedirs(os.path.dirname(ts_path_train), exist_ok=True)
    with open(ts_path_train, "w") as f:
        f.write("# from_timestamp_us, to_timestamp_us\n")
        for fr, to in ts_rows_train:
            f.write(f"{fr}, {to}\n")

    # Generate flows for test split
    ts_rows_test = []
    idx_test = 0
    if test_size > 0 and seq_name_test:
        for frame_from in range(split_start_frame, total_frames, save_step):
            frame_to = frame_from + save_step
            if frame_to > total_frames:
                break
            flows = star.compute_optical_flow_between_frames(all_coords, frame_from, frame_to)
            if flows is None:
                raise RuntimeError("compute_optical_flow_between_frames returned None")
            u = np.zeros((H, W), dtype=np.float32)
            v = np.zeros((H, W), dtype=np.float32)
            valid = np.zeros((H, W), dtype=np.uint8)
            is_valid = ~np.isnan(flows[:, 0])
            coords_valid = all_coords[is_valid]
            disp = flows[is_valid]
            if coords_valid.size > 0:
                x = coords_valid[:, 0].astype(int)
                y = coords_valid[:, 1].astype(int)
                in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                x = x[in_bounds]
                y = y[in_bounds]
                d = disp[in_bounds]
                u[y, x] = d[:, 0].astype(np.float32)
                v[y, x] = d[:, 1].astype(np.float32)
                valid[y, x] = 1
            img16 = encode_flow_dsec(u, v, valid)
            png_path = os.path.join(flow_dir_test, f"{idx_test:06d}.png")
            imageio.imwrite(png_path, img16, format="PNG-FI")
            # Test timestamps start at 0 independently
            from_ts = 0 + idx_test * flow_dt_us
            to_ts = from_ts + flow_dt_us
            ts_rows_test.append((from_ts, to_ts))
            print(f"[test] Saved flow #{idx_test} : frame {frame_from} -> {frame_to} (valid: {int(valid.sum())} pixels)")
            idx_test += 1

        os.makedirs(os.path.dirname(ts_path_test), exist_ok=True)
        with open(ts_path_test, "w") as f:
            f.write("# from_timestamp_us, to_timestamp_us\n")
            for fr, to in ts_rows_test:
                f.write(f"{fr}, {to}\n")

    print(
        f"Done. Wrote {idx_train} train flow PNGs to '{flow_dir_train}'"
        + (f" and {idx_test} test flow PNGs to '{flow_dir_test}'" if test_size > 0 else "")
    )
    
    # Save dataset metadata for later reference
    metadata = {
        'seq_name': seq_name,
        'total_frames': total_frames,
        'image_size': image_size,
        'save_step': save_step,
        'frame_time_us': frame_time_us,
        'flow_dt_us': flow_dt_us,
        'start_ts_us': start_ts_us,
        'face_color': face_color,
        'test_size': test_size,
        'split_start_frame': split_start_frame,
        'num_flow_pairs_train': idx_train,
        'num_flow_pairs_test': idx_test,
    }
    
    # Save metadata to train sequence
    import json
    metadata_path_train = os.path.join(outdir, "train_optical_flow", seq_name, "dataset_metadata.json")
    os.makedirs(os.path.dirname(metadata_path_train), exist_ok=True)
    with open(metadata_path_train, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved dataset metadata to {metadata_path_train}")
    
    # Save metadata to test sequence if exists
    if seq_name_test:
        metadata_test = metadata.copy()
        metadata_test['seq_name'] = seq_name_test
        metadata_path_test = os.path.join(outdir, "train_optical_flow", seq_name_test, "dataset_metadata.json")
        os.makedirs(os.path.dirname(metadata_path_test), exist_ok=True)
        with open(metadata_path_test, 'w') as f:
            json.dump(metadata_test, f, indent=2)
        print(f"Saved dataset metadata to {metadata_path_test}")
    
    return {
        "flow_dir_train": flow_dir_train,
        "timestamps_train": ts_path_train,
        "events_h5_train": events_h5_train,
        "rectify_map_train": rectify_map_train,
        "num_flow_pairs_train": idx_train,
        "flow_dir_test": flow_dir_test,
        "timestamps_test": ts_path_test,
        "events_h5_test": events_h5_test,
        "rectify_map_test": rectify_map_test,
        "num_flow_pairs_test": idx_test,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic star8 optical flow + events in DSEC layout (with optional train/test split)")
    parser.add_argument("--seq-name", default=DEFAULT_SEQ_NAME, help="Sequence name (directory name)")
    parser.add_argument("--total-frames", type=int, default=DEFAULT_TOTAL_FRAMES, help="Total frames to simulate")
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1], help="Image width")
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Image height")
    parser.add_argument("--save-step", type=int, default=DEFAULT_SAVE_STEP, help="Frame step for forward flow displacement")
    parser.add_argument("--frame-time-us", type=int, default=DEFAULT_FRAME_TIME_US, help="Microseconds between successive frames in synthetic timestamps")
    parser.add_argument("--flow-dt-us", type=int, default=DEFAULT_FLOW_DT_US, help="Time delta between forward flow pairs (e.g. 100000 for 10Hz)")
    parser.add_argument("--start-ts-us", type=int, default=DEFAULT_START_TS_US, help="Start timestamp offset (us)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Root output directory")
    parser.add_argument("--face-color", default="black", help="Star fill color")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction in (0,1) to reserve for test (suffix _test). Last part is test. Default 0.2")
    parser.add_argument("--sanity-check", action="store_true", default=True, help="Generate sanity check visualizations after dataset creation (default: True)")
    parser.add_argument("--no-sanity-check", dest="sanity_check", action="store_false", help="Skip sanity check visualizations")

    args = parser.parse_args()
    image_size = (args.image_height, args.image_width)
    
    # Generate dataset
    result = build_star8_flow_and_events(
        seq_name=args.seq_name,
        total_frames=args.total_frames,
        image_size=image_size,
        save_step=args.save_step,
        frame_time_us=args.frame_time_us,
        outdir=args.outdir,
        start_ts_us=args.start_ts_us,
        flow_dt_us=args.flow_dt_us,
        face_color=args.face_color,
        test_size=args.test_size,
    )
    
    # Run sanity checks if enabled
    if args.sanity_check:
        print("\n" + "="*60)
        print("Running sanity checks on generated flows...")
        print("="*60)
        
        # Sanity check for train split
        print(f"\n[SANITY CHECK] Train split: {args.seq_name}")
        n_train = generate_sanity_check_figures(
            seq_name=args.seq_name,
            outdir=args.outdir,
            image_size=image_size,
            total_frames=args.total_frames,
            save_step=args.save_step,
            test_size=args.test_size,
            split="train",
            face_color=args.face_color,
        )
        print(f"[SANITY CHECK] Generated {n_train} train sanity figures")
        
        # Sanity check for test split if it exists
        if args.test_size > 0:
            seq_name_test = args.seq_name + "_test"
            print(f"\n[SANITY CHECK] Test split: {seq_name_test}")
            n_test = generate_sanity_check_figures(
                seq_name=seq_name_test,
                outdir=args.outdir,
                image_size=image_size,
                total_frames=args.total_frames,
                save_step=args.save_step,
                test_size=args.test_size,
                split="test",
                face_color=args.face_color,
            )
            print(f"[SANITY CHECK] Generated {n_test} test sanity figures")
        
        print("\n" + "="*60)
        print("Sanity checks completed!")
        print("="*60)
