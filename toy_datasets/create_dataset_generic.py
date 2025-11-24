"""
Generic dataset generator that works with any ShapeMovementBase subclass.

This script can generate datasets for any shape/movement combination:
- Star with figure-eight trajectory (star8)
- Triangle with linear motion (triangle)
- Lissajous curves with various shapes (lissajous)
- Custom shapes with custom trajectories

Usage examples:

1. Generate Lissajous dataset:
   python create_dataset_generic.py --shape-class lissajous --seq-name lissajous_5_3 \\
       --freq-ratio-a 5 --freq-ratio-b 3 --shape-type star

2. Generate star8 (backward compatible):
   python create_dataset_generic.py --shape-class star8 --seq-name star8

3. Generate triangle:
   python create_dataset_generic.py --shape-class triangle --seq-name triangle_motion \\
       --triangle-base 40 --triangle-height 60

4. Auto-naming for parallel sweeps:
   python create_dataset_generic.py --shape-class lissajous --auto-name \\
       --freq-ratio-a 7 --freq-ratio-b 5
"""

import os
import sys
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt

# Import the flow generation functions from existing script
from create_flow_from_movement import (
    encode_flow_dsec,
    decode_flow_dsec,
    create_dsec_events_h5,
    create_identity_rectify_map,
    generate_dataset_hash,
    DEFAULT_TOTAL_FRAMES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SAVE_STEP,
    DEFAULT_FRAME_TIME_US,
    DEFAULT_START_TS_US,
    DEFAULT_FLOW_DT_US,
)

# Use relative path from script location (more generic than hardcoded "star8")
DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), "data")

import imageio.v2 as imageio


def create_event_animation(
    events: np.ndarray,
    image_size: tuple,
    output_path: str,
    fps: int = 20,
    accumulation_time_ms: int = 10,
    frame_time_us: int = 1000,
) -> None:
    """
    Create an animation showing event accumulation over time.
    
    Args:
        events: Structured numpy array with fields ('x', 'y', 't', 'p')
        image_size: (H, W) tuple
        output_path: Path to save the animation GIF
        fps: Frames per second for output animation
        accumulation_time_ms: Time window in milliseconds to accumulate events per frame
        frame_time_us: Microseconds per simulation frame (for timing reference)
    """
    if len(events) == 0:
        print("[warning] No events to animate")
        return
    
    H, W = image_size
    
    # Time range
    t_start = events['t'].min()
    t_end = events['t'].max()
    time_range_ms = (t_end - t_start) / 1000.0
    
    print(f"Creating event animation:")
    print(f"  Time range: {t_start:.0f} - {t_end:.0f} μs ({time_range_ms:.1f} ms)")
    print(f"  Total events: {len(events)}")
    print(f"  Accumulation window: {accumulation_time_ms} ms")
    
    # Generate frames
    accumulation_window_us = accumulation_time_ms * 1000
    current_time_us = t_start
    frames = []
    
    from matplotlib import cm
    
    while current_time_us < t_end:
        # Get events in this time window
        window_mask = (events['t'] >= current_time_us) & (events['t'] < current_time_us + accumulation_window_us)
        window_events = events[window_mask]
        
        # Create accumulation image
        # Use different colors for positive/negative polarity
        img_pos = np.zeros((H, W), dtype=np.float32)
        img_neg = np.zeros((H, W), dtype=np.float32)
        
        if len(window_events) > 0:
            for event in window_events:
                x, y, p = event['x'], event['y'], event['p']
                if 0 <= x < W and 0 <= y < H:
                    if p:  # Positive polarity
                        img_pos[y, x] += 1
                    else:  # Negative polarity
                        img_neg[y, x] += 1
        
        # Normalize and create RGB image
        # Red for positive events, Blue for negative events
        max_val = max(img_pos.max(), img_neg.max(), 1.0)
        img_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        img_rgb[..., 0] = np.clip((img_pos / max_val) * 255, 0, 255).astype(np.uint8)  # Red channel
        img_rgb[..., 2] = np.clip((img_neg / max_val) * 255, 0, 255).astype(np.uint8)  # Blue channel
        
        # Add timestamp overlay
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(img_rgb)
        ax.set_title(f"Events: {current_time_us:.0f} μs ({len(window_events)} events)", fontsize=10)
        ax.axis('off')
        
        # Convert plot to image
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame_data = frame_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame_data)
        plt.close(fig)
        
        current_time_us += accumulation_window_us
    
    print(f"  Generated {len(frames)} animation frames")
    
    # Save as GIF
    if len(frames) > 0:
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"✓ Event animation saved: {output_path}")
    else:
        print("[warning] No frames generated for event animation")


def get_shape_class(shape_class_name: str):
    """
    Dynamically import and return the shape movement class.
    
    Args:
        shape_class_name: Name of the module/class (e.g., 'star8', 'triangle', 'lissajous')
        
    Returns:
        The class object (e.g., StarMovement, TriangleMovement, LissajousMovement)
    """
    class_map = {
        'star8': ('star8', 'StarMovement'),
        'triangle': ('triangle', 'TriangleMovement'),
        'lissajous': ('lissajous', 'LissajousMovement'),
        'multi_lissajous': ('lissajous', 'MultiShapeLissajous'),
    }
    
    if shape_class_name not in class_map:
        raise ValueError(
            f"Unknown shape class: {shape_class_name}. "
            f"Available: {list(class_map.keys())}"
        )
    
    module_name, class_name = class_map[shape_class_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_vertices(shape_instance, frame_idx: int):
    """Get shape vertices at a specific frame."""
    shape_instance.update_shape(frame_idx)
    return shape_instance.transformed_path.vertices


def generate_sanity_check_figures(
    shape_instance,
    seq_name: str,
    outdir: str,
    image_size: tuple,
    total_frames: int,
    save_step: int,
    test_size: float,
    split: str,
):
    """Generate sanity check figures by loading the saved flow PNGs."""
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
        if frame_to >= total_frames:
            break
        
        # Get vertices at start and end
        verts_start = get_vertices(shape_instance, frame_from)
        verts_end = get_vertices(shape_instance, frame_to)
        
        # Load flow PNG
        flow_png = os.path.join(forward_dir, png_file)
        u, v, valid = decode_flow_dsec(flow_png)
        
        # Prepare plot
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"{seq_name} | pair {idx:06d} | frames {frame_from}->{frame_to}")
        
        # Plot start vertices (blue)
        if len(verts_start) > 0:
            ax.scatter(verts_start[:, 0], verts_start[:, 1], c='blue', s=20, label='start vertices', zorder=5)
            vs = verts_start
            if not np.allclose(vs[0], vs[-1]):
                vs = np.vstack([vs, vs[:1]])
            ax.plot(vs[:, 0], vs[:, 1], '-', color='blue', alpha=0.6, linewidth=1.5, zorder=4)
        
        # Plot end vertices (red)
        if len(verts_end) > 0:
            ax.scatter(verts_end[:, 0], verts_end[:, 1], c='red', s=20, label='end vertices', zorder=5)
            ve = verts_end
            if not np.allclose(ve[0], ve[-1]):
                ve = np.vstack([ve, ve[:1]])
            ax.plot(ve[:, 0], ve[:, 1], '-', color='red', alpha=0.6, linewidth=1.5, zorder=4)
        
        # Draw GT flow arrows at vertices (green)
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
                dx = vx_valid.astype(np.float32) - float(px)
                dy = vy_valid.astype(np.float32) - float(py)
                idx_min = np.argmin(dx * dx + dy * dy)
                py_n = int(vy_valid[idx_min])
                px_n = int(vx_valid[idx_min])
                du = u[py_n, px_n]
                dv = v[py_n, px_n]
            
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


def build_dataset(
    shape_instance,
    seq_name: str,
    total_frames: int,
    image_size: tuple,
    save_step: int,
    frame_time_us: int,
    outdir: str,
    start_ts_us: int,
    flow_dt_us: int,
    test_size: float,
) -> dict:
    """
    Generate optical flow + events dataset from any ShapeMovementBase instance.
    
    Args:
        shape_instance: Instance of a ShapeMovementBase subclass
        seq_name: Sequence name for directory
        (other parameters same as original function)
        
    Returns:
        dict with output paths and metadata
    """
    # Resolve split
    test_size = float(test_size)
    if not (0.0 <= test_size < 1.0):
        raise ValueError("test_size must be in [0, 1)")
    split_start_frame = int(np.floor(total_frames * (1.0 - test_size))) if test_size > 0 else total_frames
    split_start_ts_us = split_start_frame * frame_time_us

    # Output dirs
    flow_dir_train = os.path.join(outdir, "train_optical_flow", seq_name, "flow", "forward")
    ts_path_train = os.path.join(outdir, "train_optical_flow", seq_name, "flow", "forward_timestamps.txt")
    event_dir_train = os.path.join(outdir, "train_events", seq_name, "events", "left")
    
    # Check if dataset exists
    seq_flow_root = os.path.join(outdir, "train_optical_flow", seq_name)
    seq_event_root = os.path.join(outdir, "train_events", seq_name)
    
    dataset_exists = os.path.exists(seq_flow_root) or os.path.exists(seq_event_root)
    if dataset_exists:
        metadata_path = os.path.join(outdir, "train_optical_flow", seq_name, "dataset_metadata.json")
        if os.path.exists(metadata_path):
            print(f"[skip] Dataset already exists at {outdir}/{seq_name}")
            return {
                "flow_dir_train": flow_dir_train,
                "timestamps_train": ts_path_train,
                "events_h5_train": os.path.join(event_dir_train, "events.h5"),
                "rectify_map_train": os.path.join(event_dir_train, "rectify_map.h5"),
                "num_flow_pairs_train": 0,
                "flow_dir_test": None,
                "timestamps_test": None,
                "events_h5_test": None,
                "rectify_map_test": None,
                "num_flow_pairs_test": 0,
                "skipped": True,
            }
        else:
            print(f"[cleanup] Removing incomplete dataset: {seq_flow_root}")
            import shutil
            for path in [seq_flow_root, seq_event_root]:
                if os.path.exists(path):
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

    print("Generating events...")
    events = shape_instance.generate_events()
    events['t'] = events['t'] * frame_time_us

    # Split events
    if test_size > 0:
        train_mask = events['t'] < split_start_ts_us
        test_mask = ~train_mask
        events_train = events[train_mask]
        events_test = events[test_mask].copy()
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
        flows = shape_instance.compute_optical_flow_between_frames(all_coords, frame_from, frame_to)
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
            flows = shape_instance.compute_optical_flow_between_frames(all_coords, frame_from, frame_to)
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
    
    # Save metadata
    import json
    metadata = {
        'seq_name': seq_name,
        'shape_class': shape_instance.__class__.__name__,
        'total_frames': total_frames,
        'image_size': image_size,
        'save_step': save_step,
        'frame_time_us': frame_time_us,
        'flow_dt_us': flow_dt_us,
        'start_ts_us': start_ts_us,
        'test_size': test_size,
        'split_start_frame': split_start_frame,
        'num_flow_pairs_train': idx_train,
        'num_flow_pairs_test': idx_test,
    }
    
    metadata_path_train = os.path.join(outdir, "train_optical_flow", seq_name, "dataset_metadata.json")
    os.makedirs(os.path.dirname(metadata_path_train), exist_ok=True)
    with open(metadata_path_train, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved dataset metadata to {metadata_path_train}")
    
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
    parser = argparse.ArgumentParser(
        description="Generate synthetic optical flow + events for any shape/trajectory combination",
        epilog="Examples:\n"
               "  # Lissajous curve:\n"
               "  python create_dataset_generic.py --shape-class lissajous --seq-name lissajous_5_3 --freq-ratio-a 5 --freq-ratio-b 3\n\n"
               "  # Star8 (backward compatible):\n"
               "  python create_dataset_generic.py --shape-class star8 --seq-name star8\n\n"
               "  # Auto-naming for parallel sweeps:\n"
               "  python create_dataset_generic.py --shape-class lissajous --auto-name --freq-ratio-a 7 --freq-ratio-b 5\n\n"
               "  # With animations (NEW!):\n"
               "  python create_dataset_generic.py --shape-class lissajous --seq-name animated_demo \\\n"
               "      --freq-ratio-a 5 --freq-ratio-b 3 --total-frames 500 \\\n"
               "      --generate-animation --generate-event-animation\n\n"
               "  # Animations saved to: train_optical_flow/{seq_name}/animation/*.gif\n"
               "  # See ANIMATION_GUIDE.md for detailed animation options!\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Shape/movement selection
    parser.add_argument("--shape-class", required=True, 
                       choices=['star8', 'triangle', 'lissajous', 'multi_lissajous'],
                       help="Shape movement class to use")
    parser.add_argument("--seq-name", default="generic_dataset", 
                       help="Sequence name (directory name)")
    parser.add_argument("--auto-name", action="store_true", default=False,
                       help="Auto-append config hash to avoid conflicts")
    
    # Common parameters
    parser.add_argument("--total-frames", type=int, default=DEFAULT_TOTAL_FRAMES)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--save-step", type=int, default=DEFAULT_SAVE_STEP)
    parser.add_argument("--frame-time-us", type=int, default=DEFAULT_FRAME_TIME_US)
    parser.add_argument("--flow-dt-us", type=int, default=DEFAULT_FLOW_DT_US)
    parser.add_argument("--start-ts-us", type=int, default=DEFAULT_START_TS_US)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                       help=f"Root output directory (default: toy_datasets/data/)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sanity-check", action="store_true", default=True)
    parser.add_argument("--no-sanity-check", dest="sanity_check", action="store_false")
    
    # Animation generation
    parser.add_argument("--generate-animation", action="store_true", default=False,
                       help="Generate GIF animation of the shape movement")
    parser.add_argument("--animation-frame-step", type=int, default=5,
                       help="Frame step for animation (skip frames to reduce file size)")
    parser.add_argument("--animation-interval", type=int, default=50,
                       help="Delay between frames in milliseconds for animation")
    parser.add_argument("--animation-fps", type=int, default=20,
                       help="Frames per second for saved animation")
    
    parser.add_argument("--generate-event-animation", action="store_true", default=False,
                       help="Generate GIF animation of event accumulation")
    parser.add_argument("--event-accumulation-ms", type=int, default=10,
                       help="Time window in milliseconds to accumulate events per frame")
    parser.add_argument("--event-animation-fps", type=int, default=20,
                       help="Frames per second for event animation")
    
    # Lissajous-specific parameters
    parser.add_argument("--shape-type", default="circle",
                       choices=['circle', 'square', 'star', 'hexagon'],
                       help="Shape type for Lissajous")
    parser.add_argument("--shape-size", type=int, default=20,
                       help="Shape size for Lissajous")
    parser.add_argument("--freq-ratio-a", type=int, default=3,
                       help="Lissajous frequency ratio numerator")
    parser.add_argument("--freq-ratio-b", type=int, default=2,
                       help="Lissajous frequency ratio denominator")
    parser.add_argument("--phase-shift", type=float, default=np.pi/2,
                       help="Lissajous phase shift (radians)")
    parser.add_argument("--amplitude-scale", type=float, default=0.35,
                       help="Lissajous amplitude scale relative to image size")
    parser.add_argument("--rotation-speed", type=float, default=2.0,
                       help="Rotation speed (rotations per full trajectory)")
    
    # Triangle-specific parameters
    parser.add_argument("--triangle-base", type=float, default=40)
    parser.add_argument("--triangle-height", type=float, default=60)
    parser.add_argument("--added-height", type=float, default=100)
    parser.add_argument("--start-pos-x", type=float, default=128)
    parser.add_argument("--start-pos-y", type=float, default=128)
    parser.add_argument("--movement-direction", type=float, default=0.0)
    
    # Star8-specific parameters
    parser.add_argument("--face-color", default="black")
    parser.add_argument("--num-points", type=int, default=5)
    parser.add_argument("--outer-radius", type=int, default=40)
    parser.add_argument("--inner-radius", type=int, default=20)
    parser.add_argument("--number-of-rotations", type=int, default=2)
    
    # Multi-shape parameters
    parser.add_argument("--num-shapes", type=int, default=3,
                       help="Number of shapes for multi_lissajous")

    args = parser.parse_args()
    image_size = (args.image_height, args.image_width)
    
    # Auto-calculate flow_dt_us if needed
    if args.flow_dt_us == DEFAULT_FLOW_DT_US and args.save_step != DEFAULT_SAVE_STEP:
        args.flow_dt_us = args.save_step * args.frame_time_us
        print(f"ℹ️  Auto-calculated flow_dt_us = {args.flow_dt_us} us")
    
    # Get shape class
    ShapeClass = get_shape_class(args.shape_class)
    
    # Build kwargs based on shape class
    shape_kwargs = {
        'total_frames': args.total_frames,
        'image_size': image_size,
    }
    
    if args.shape_class == 'lissajous':
        shape_kwargs.update({
            'shape_type': args.shape_type,
            'shape_size': args.shape_size,
            'freq_ratio_a': args.freq_ratio_a,
            'freq_ratio_b': args.freq_ratio_b,
            'phase_shift': args.phase_shift,
            'amplitude_scale': args.amplitude_scale,
            'rotation_speed': args.rotation_speed,
            'face_color': args.face_color,
        })
    elif args.shape_class == 'multi_lissajous':
        shape_kwargs.update({
            'num_shapes': args.num_shapes,
        })
    elif args.shape_class == 'triangle':
        shape_kwargs.update({
            'triangle_base': args.triangle_base,
            'triangle_height': args.triangle_height,
            'added_height': args.added_height,
            'start_pos': np.array([args.start_pos_x, args.start_pos_y]),
            'face_color': args.face_color,
            'movement_direction': args.movement_direction,
        })
    elif args.shape_class == 'star8':
        shape_kwargs.update({
            'face_color': args.face_color,
            'num_points': args.num_points,
            'outer_radius': args.outer_radius,
            'inner_radius': args.inner_radius,
            'number_of_rotations': args.number_of_rotations,
        })
    
    # Create shape instance
    shape_instance = ShapeClass(**shape_kwargs)
    print(f"Created {shape_instance.shape_name} with {shape_instance.trajectory_name} trajectory")
    
    # Auto-naming
    if args.auto_name:
        config_hash = generate_dataset_hash(**vars(args))
        outdir = os.path.join(args.outdir, f"variant_{config_hash}")
        print(f"🔸 Auto-naming enabled: Using variant directory '{config_hash}'")
    else:
        outdir = args.outdir
        config_hash = None
    
    seq_name = args.seq_name
    
    # Generate dataset
    result = build_dataset(
        shape_instance=shape_instance,
        seq_name=seq_name,
        total_frames=args.total_frames,
        image_size=image_size,
        save_step=args.save_step,
        frame_time_us=args.frame_time_us,
        outdir=outdir,
        start_ts_us=args.start_ts_us,
        flow_dt_us=args.flow_dt_us,
        test_size=args.test_size,
    )
    
    if config_hash:
        result['config_hash'] = config_hash
        result['dataset_variant'] = f"variant_{config_hash}"
    
    # Generate animation if requested
    if args.generate_animation and not result.get('skipped', False):
        print("\n" + "="*60)
        print("Generating shape movement animation...")
        print("="*60)
        
        # Create animation
        anim = shape_instance.create_animation(
            frame_step=args.animation_frame_step,
            interval=args.animation_interval
        )
        
        # Save animation to the sequence directory
        animation_dir = os.path.join(outdir, "train_optical_flow", seq_name, "animation")
        os.makedirs(animation_dir, exist_ok=True)
        animation_path = os.path.join(animation_dir, f"{seq_name}_movement.gif")
        
        print(f"Saving animation to {animation_path}...")
        anim.save(animation_path, writer='pillow', fps=args.animation_fps)
        print(f"✓ Shape movement animation saved: {animation_path}")
        
        result['animation_path'] = animation_path
    
    # Generate event animation if requested
    if args.generate_event_animation and not result.get('skipped', False):
        print("\n" + "="*60)
        print("Generating event animation...")
        print("="*60)
        
        # Get the events (they were already generated in build_dataset)
        # We need to regenerate them here since they were split
        print("Regenerating events for visualization...")
        events = shape_instance.generate_events()
        events['t'] = events['t'] * args.frame_time_us
        
        # Use only train split events if test_size > 0
        if args.test_size > 0:
            split_start_ts_us = int(np.floor(args.total_frames * (1.0 - args.test_size))) * args.frame_time_us
            train_mask = events['t'] < split_start_ts_us
            events_to_animate = events[train_mask]
        else:
            events_to_animate = events
        
        # Save event animation
        animation_dir = os.path.join(outdir, "train_optical_flow", seq_name, "animation")
        os.makedirs(animation_dir, exist_ok=True)
        event_animation_path = os.path.join(animation_dir, f"{seq_name}_events.gif")
        
        create_event_animation(
            events=events_to_animate,
            image_size=image_size,
            output_path=event_animation_path,
            fps=args.event_animation_fps,
            accumulation_time_ms=args.event_accumulation_ms,
            frame_time_us=args.frame_time_us,
        )
        
        result['event_animation_path'] = event_animation_path
    
    # Sanity checks
    if args.sanity_check and not result.get('skipped', False):
        print("\n" + "="*60)
        print("Running sanity checks...")
        print("="*60)
        
        print(f"\n[SANITY CHECK] Train split: {seq_name}")
        n_train = generate_sanity_check_figures(
            shape_instance=shape_instance,
            seq_name=seq_name,
            outdir=outdir,
            image_size=image_size,
            total_frames=args.total_frames,
            save_step=args.save_step,
            test_size=args.test_size,
            split="train",
        )
        print(f"[SANITY CHECK] Generated {n_train} train sanity figures")
        
        if args.test_size > 0:
            seq_name_test = seq_name + "_test"
            print(f"\n[SANITY CHECK] Test split: {seq_name_test}")
            n_test = generate_sanity_check_figures(
                shape_instance=shape_instance,
                seq_name=seq_name_test,
                outdir=outdir,
                image_size=image_size,
                total_frames=args.total_frames,
                save_step=args.save_step,
                test_size=args.test_size,
                split="test",
            )
            print(f"[SANITY CHECK] Generated {n_test} test sanity figures")
        
        print("\n" + "="*60)
        print("Sanity checks completed!")
        print("="*60)
