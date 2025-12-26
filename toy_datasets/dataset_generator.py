"""
Dataset Generator Class for synthetic optical flow and events.

This module provides a reusable class-based interface for generating
synthetic datasets with various shapes, trajectories, and textures.

Now uses Hydra/OmegaConf DictConfig directly for consistency with idn/train.py
"""

import os
import sys
import importlib
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, Union
from omegaconf import DictConfig, OmegaConf
import imageio.v2 as imageio

from dsec_utils import (
    encode_flow_dsec,
    decode_flow_dsec,
    create_dsec_events_h5,
    create_identity_rectify_map,
    generate_dataset_hash,
)


class DatasetGenerator:
    """Generate synthetic optical flow and event datasets.
    
    This class encapsulates all the functionality for generating datasets
    with various shapes, trajectories, and textures. It can be used both
    as a standalone tool and as a library component.
    
    Now accepts OmegaConf DictConfig directly (Hydra-native approach).
    
    Example:
        >>> from omegaconf import OmegaConf
        >>> from dataset_generator import DatasetGenerator
        >>> 
        >>> # Create config from dict
        >>> config = OmegaConf.create({
        ...     'shape_class': 'lissajous',
        ...     'seq_name': 'my_dataset',
        ...     'freq_ratio_a': 5,
        ...     'freq_ratio_b': 3,
        ...     'outdir': 'toy_datasets/data',
        ... })
        >>> 
        >>> # Generate dataset
        >>> generator = DatasetGenerator(config)
        >>> result = generator.generate()
        >>> print(f"Generated {result['num_flow_pairs_train']} flow pairs")
    """
    
    def __init__(self, config: Union[DictConfig, dict]):
        """Initialize the dataset generator.
        
        Args:
            config: OmegaConf DictConfig or dict with all parameters
        """
        # Convert to DictConfig if dict is provided
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        
        # Disable struct mode to allow dynamic field access
        # This is necessary for backward compatibility with code that adds fields
        OmegaConf.set_struct(config, False)
        
        self.config = config
        self.shape_instance = None
        self.image_size = (config.image_height, config.image_width)
        
    def _get_shape_class(self, shape_class_name: str):
        """Dynamically import and return the shape movement class.
        
        Args:
            shape_class_name: Name of the module/class
            
        Returns:
            The class object
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
    
    def _select_random_dtd_texture(self, dtd_path: str, seed: int) -> str:
        """Select a random texture image from the DTD dataset.
        
        Args:
            dtd_path: Path to the DTD dataset
            seed: Random seed for reproducibility
            
        Returns:
            Path to randomly selected texture image
        """
        if not os.path.exists(dtd_path):
            raise ValueError(f"DTD path does not exist: {dtd_path}")
        
        categories = sorted([d for d in os.listdir(dtd_path) 
                            if os.path.isdir(os.path.join(dtd_path, d))])
        
        if not categories:
            raise ValueError(f"No categories found in DTD path: {dtd_path}")
        
        rng = np.random.RandomState(seed)
        category = str(rng.choice(categories))
        
        category_path = os.path.join(dtd_path, category)
        images = sorted(glob.glob(os.path.join(category_path, "*.jpg")))
        
        if not images:
            raise ValueError(f"No images found in category: {category}")
        
        image_path = str(rng.choice(images))
        
        print(f"Selected DTD texture (seed={seed}): {category}/{os.path.basename(image_path)}")
        
        return image_path
    
    def _preselect_dtd_textures(self):
        """Pre-select DTD textures and update config BEFORE hash generation.
        
        This ensures that the selected texture paths are included in the dataset hash,
        preventing hash collisions for datasets that differ only in DTD texture selection.
        """
        cfg = self.config
        use_dtd = cfg.get('use_random_dtd_texture', False)
        dtd_mode = cfg.get('dtd_texture_mode', 'both')
        
        if use_dtd:
            dtd_root = cfg.get('dtd_root', 'data/dtd/images')
            
            if dtd_mode in ['fg', 'foreground', 'both']:
                # Use dtd_fg_seed if provided, otherwise fall back to random_seed
                fg_seed = cfg.get('dtd_fg_seed', cfg.get('random_seed'))
                dtd_fg_path = str(self._select_random_dtd_texture(dtd_root, fg_seed))
                OmegaConf.update(cfg, 'foreground_texture', 'image', force_add=True)
                OmegaConf.update(cfg, 'fg_image_path', dtd_fg_path, force_add=True)
                print(f"🎨 Pre-selected DTD foreground texture (seed={fg_seed}): {dtd_fg_path}")
            
            if dtd_mode in ['bg', 'background', 'both']:
                # Use dtd_bg_seed if provided, otherwise fall back to random_seed + 1
                bg_seed = cfg.get('dtd_bg_seed')
                if bg_seed is None:
                    bg_seed = (cfg.get('random_seed') + 1) if cfg.get('random_seed') is not None else None
                dtd_bg_path = str(self._select_random_dtd_texture(dtd_root, bg_seed))
                OmegaConf.update(cfg, 'background_texture', 'image', force_add=True)
                OmegaConf.update(cfg, 'bg_image_path', dtd_bg_path, force_add=True)
                print(f"🎨 Pre-selected DTD background texture (seed={bg_seed}): {dtd_bg_path}")
    
    def _build_texture_params(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build texture parameter dictionaries from config.
        
        Note: DTD texture selection is now done in _preselect_dtd_textures() before hash generation.
        This method only builds the parameter dictionaries from the already-configured texture settings.
        
        Returns:
            Tuple of (foreground_params, background_params)
        """
        cfg = self.config
        foreground_params = {}
        background_params = {}
        
        # Build foreground texture params
        if cfg.get('foreground_texture'):
            if cfg.foreground_texture == 'solid':
                foreground_params = {'color': cfg.get('fg_texture_color', 'black')}
            elif cfg.foreground_texture == 'noise':
                foreground_params = {
                    'base_color': cfg.get('fg_texture_color', 'black'),
                    'noise_type': cfg.get('fg_noise_type', 'gaussian'),
                    'scale': cfg.get('fg_noise_scale', 0.3),
                    'seed': cfg.get('fg_noise_seed', 42)
                }
            elif cfg.foreground_texture == 'gradient':
                foreground_params = {
                    'gradient_type': cfg.fg_gradient_type,
                    'color1': cfg.fg_gradient_color1,
                    'color2': cfg.fg_gradient_color2,
                    'angle': cfg.fg_gradient_angle
                }
            elif cfg.foreground_texture == 'checkerboard':
                foreground_params = {
                    'square_size': cfg.fg_checker_size,
                    'color1': cfg.fg_checker_color1,
                    'color2': cfg.fg_checker_color2
                }
            elif cfg.foreground_texture == 'image':
                from matplotlib.colors import to_rgb
                fg_fill_color = cfg.get('fg_image_fill_color', 'black')
                fill_color = to_rgb(fg_fill_color) if isinstance(fg_fill_color, str) else fg_fill_color
                foreground_params = {
                    'image_path': cfg.fg_image_path,
                    'resize_mode': cfg.get('fg_image_resize_mode', 'fill'),
                    'fill_color': fill_color
                }
        
        # Build background texture params
        if cfg.background_texture:
            if cfg.background_texture == 'solid':
                background_params = {'color': cfg.bg_texture_color}
            elif cfg.background_texture == 'noise':
                background_params = {
                    'base_color': cfg.bg_texture_color,
                    'noise_type': cfg.bg_noise_type,
                    'scale': cfg.bg_noise_scale,
                    'seed': cfg.bg_noise_seed
                }
            elif cfg.background_texture == 'gradient':
                background_params = {
                    'gradient_type': cfg.bg_gradient_type,
                    'color1': cfg.bg_gradient_color1,
                    'color2': cfg.bg_gradient_color2,
                    'angle': cfg.bg_gradient_angle
                }
            elif cfg.background_texture == 'checkerboard':
                background_params = {
                    'square_size': cfg.bg_checker_size,
                    'color1': cfg.bg_checker_color1,
                    'color2': cfg.bg_checker_color2
                }
            elif cfg.background_texture == 'image':
                from matplotlib.colors import to_rgb
                bg_fill_color = cfg.get('bg_image_fill_color', 'white')
                fill_color = to_rgb(bg_fill_color) if isinstance(bg_fill_color, str) else bg_fill_color
                background_params = {
                    'image_path': cfg.bg_image_path,
                    'resize_mode': cfg.get('bg_image_resize_mode', 'fill'),
                    'fill_color': fill_color
                }
        
        return foreground_params, background_params
    
    def _create_shape_instance(self):
        """Create the shape instance based on configuration."""
        cfg = self.config
        
        # Set defaults for animation configuration if not provided
        if not cfg.get('animation_fps'):
            OmegaConf.update(cfg, 'animation_fps', 10, force_add=True)
        if not cfg.get('animation_frame_step'):
            OmegaConf.update(cfg, 'animation_frame_step', 10, force_add=True)
        if not cfg.get('event_animation_fps'):
            OmegaConf.update(cfg, 'event_animation_fps', 10, force_add=True)
        if not cfg.get('event_accumulation_ms'):
            OmegaConf.update(cfg, 'event_accumulation_ms', 10, force_add=True)
        
        ShapeClass = self._get_shape_class(cfg.shape_class)
        
        # Build texture params
        fg_params, bg_params = self._build_texture_params()
        
        # Common kwargs for all shapes
        shape_kwargs = {
            'total_frames': cfg.total_frames,
            'image_size': self.image_size,
            'foreground_texture': cfg.foreground_texture,
            'background_texture': cfg.background_texture,
            'foreground_texture_params': fg_params,
            'background_texture_params': bg_params,
            'frame_time_us': cfg.frame_time_us,
        }
        
        # Add shape-specific parameters
        if cfg.shape_class == 'lissajous':
            shape_kwargs.update({
                'shape_type': cfg.shape_type,
                'shape_size': cfg.shape_size,
                'star_num_points': cfg.star_num_points,
                'star_inner_ratio': cfg.star_inner_ratio,
                'polygon_num_sides': cfg.polygon_num_sides,
                'freq_ratio_a': cfg.freq_ratio_a,
                'freq_ratio_b': cfg.freq_ratio_b,
                'phase_shift': cfg.phase_shift,
                'amplitude_scale': cfg.amplitude_scale,
                'rotation_speed': cfg.rotation_speed,
                'face_color': cfg.face_color,
            })
        elif cfg.shape_class == 'multi_lissajous':
            shape_kwargs.update({
                'num_shapes': cfg.num_shapes,
            })
        elif cfg.shape_class == 'triangle':
            shape_kwargs.update({
                'triangle_base': cfg.triangle_base,
                'triangle_height': cfg.triangle_height,
                'added_height': cfg.added_height,
                'start_pos': np.array([cfg.start_pos_x, cfg.start_pos_y]),
                'face_color': cfg.face_color,
                'movement_direction': cfg.movement_direction,
            })
        elif cfg.shape_class == 'star8':
            shape_kwargs.update({
                'face_color': cfg.face_color,
                'num_points': cfg.num_points,
                'outer_radius': cfg.outer_radius,
                'inner_radius': cfg.inner_radius,
                'number_of_rotations': cfg.number_of_rotations,
            })
        
        self.shape_instance = ShapeClass(**shape_kwargs)
        print(f"Created {self.shape_instance.shape_name} with {self.shape_instance.trajectory_name} trajectory")
        
        return self.shape_instance
    
    def _generate_events(self) -> np.ndarray:
        """Generate events using the configured method.
        
        Returns:
            Structured numpy array with event data
        """
        cfg = self.config
        
        # Force intensity-based or v2e if textures are present
        if (cfg.foreground_texture is not None or cfg.background_texture is not None):
            if cfg.event_generation_method not in ['v2e', 'intensity']:
                print("\n" + "="*60)
                print("⚠️  TEXTURE DETECTED: Automatically switching to intensity-based event generation")
                print("="*60)
                print("Reason: Synthetic boundary-based events only work with solid shapes.")
                print("Textured shapes require intensity-based or v2e simulation for proper event generation.")
                print("="*60 + "\n")
                cfg.event_generation_method = 'intensity'
        
        print("Generating events...")
        
        if cfg.event_generation_method == 'intensity':
            # Use intensity-based event generation
            print(f"Using intensity-based event generation:")
            print(f"  pos_threshold: {cfg.get('intensity_pos_threshold', 0.05)}")
            print(f"  neg_threshold: {cfg.get('intensity_neg_threshold', 0.05)}")
            print(f"  fg_gamma: {cfg.get('v2e_fg_gamma', 1.0)} (from v2e settings)")
            print(f"  bg_gamma: {cfg.get('v2e_bg_gamma', 1.0)} (from v2e settings)")
            print(f"  fg_scale: {cfg.get('v2e_fg_brightness', 1.0)} (from v2e settings)")
            print(f"  bg_scale: {cfg.get('v2e_bg_brightness', 1.0)} (from v2e settings)")
            print(f"  shot_noise_rate_hz: {cfg.get('intensity_shot_noise_rate_hz', 0.0)} Hz")
            
            events = self.shape_instance.generate_events_from_intensity(
                pos_threshold=cfg.get('intensity_pos_threshold', 0.05),
                neg_threshold=cfg.get('intensity_neg_threshold', 0.05),
                fg_gamma=cfg.get('v2e_fg_gamma', 1.0),
                bg_gamma=cfg.get('v2e_bg_gamma', 1.0),
                fg_scale=cfg.get('v2e_fg_brightness', 1.0),
                bg_scale=cfg.get('v2e_bg_brightness', 1.0),
                shot_noise_rate_hz=cfg.get('intensity_shot_noise_rate_hz', 0.0),
            )
            # Convert frame indices to microseconds for intensity-based events
            events['t'] = events['t'] * cfg.frame_time_us
            
        elif cfg.event_generation_method == 'v2e':
            try:
                from v2e_event_generator import V2EEventGenerator
                
                print(f"Using v2e realistic DVS simulation:")
                print(f"  pos_threshold: {cfg.v2e_pos_thres}")
                print(f"  neg_threshold: {cfg.v2e_neg_thres}")
                print(f"  sigma_threshold: {cfg.v2e_sigma_thres}")
                print(f"  cutoff_hz: {cfg.v2e_cutoff_hz}")
                
                events = self.shape_instance.generate_events_v2e(
                    pos_threshold=cfg.v2e_pos_thres,
                    neg_threshold=cfg.v2e_neg_thres,
                    sigma_threshold=cfg.v2e_sigma_thres,
                    cutoff_hz=cfg.v2e_cutoff_hz,
                    leak_rate_hz=cfg.v2e_leak_rate_hz,
                    shot_noise_rate_hz=cfg.v2e_shot_noise_rate_hz,
                    refractory_period_s=cfg.v2e_refractory_period_s,
                    frame_time_us=cfg.frame_time_us,
                    seed=cfg.v2e_seed,
                    photoreceptor_noise=cfg.v2e_photoreceptor_noise,
                    leak_jitter_fraction=cfg.v2e_leak_jitter_fraction,
                    noise_rate_cov_decades=cfg.v2e_noise_rate_cov_decades,
                    fg_gamma=cfg.v2e_fg_gamma,
                    bg_gamma=cfg.v2e_bg_gamma,
                    fg_scale=cfg.v2e_fg_brightness,
                    bg_scale=cfg.v2e_bg_brightness,
                    temporal_filter_percent=cfg.get('v2e_temporal_filter_percent', None),
                )
            except ImportError as e:
                print(f"ERROR: v2e not available. {e}")
                print("Install v2e with one of these methods:")
                print("  1. uv pip install -e \".[v2e]\"")
                print("  2. git submodule add https://github.com/SensorsINI/v2e.git external/v2e")
                print("     cd external/v2e && uv pip install -e . && cd ../..")
                print("\nSee toy_datasets/INSTALL_V2E.md for details.")
                sys.exit(1)
        else:
            # Use synthetic boundary-based events
            print("Using synthetic boundary-based events")
            events = self.shape_instance.generate_events()
            # Convert frame indices to microseconds for synthetic events
            events['t'] = events['t'] * cfg.frame_time_us
        
        return events
    
    def _split_events(self, events: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Split events into train and test sets.
        
        Args:
            events: Full event array
            
        Returns:
            Tuple of (train_events, test_events)
        """
        cfg = self.config
        
        if cfg.test_size <= 0:
            return events, None
        
        split_start_frame = int(np.floor(cfg.total_frames * (1.0 - cfg.test_size)))
        split_start_ts_us = split_start_frame * cfg.frame_time_us
        
        train_mask = events['t'] < split_start_ts_us
        test_mask = ~train_mask
        
        events_train = events[train_mask]
        events_test = events[test_mask].copy()
        
        if len(events_test) > 0:
            events_test['t'] -= events_test['t'].min()
        
        return events_train, events_test
    
    def _save_events(self, events_train: np.ndarray, events_test: Optional[np.ndarray],
                    outdir: str, seq_name: str, seq_name_test: Optional[str]) -> Dict[str, Any]:
        """Save events to HDF5 files.
        
        Returns:
            Dict with event file paths
        """
        # Save train events
        event_dir_train = os.path.join(outdir, "train_events", seq_name, "events", "left")
        os.makedirs(event_dir_train, exist_ok=True)
        
        events_h5_train = os.path.join(event_dir_train, "events.h5")
        create_dsec_events_h5(events_train, events_h5_train, t_offset=0)
        
        rectify_map_train = os.path.join(event_dir_train, "rectify_map.h5")
        create_identity_rectify_map(self.image_size, rectify_map_train)
        
        result = {
            'events_h5_train': events_h5_train,
            'rectify_map_train': rectify_map_train,
        }
        
        # Save test events if present
        if events_test is not None and seq_name_test:
            event_dir_test = os.path.join(outdir, "train_events", seq_name_test, "events", "left")
            os.makedirs(event_dir_test, exist_ok=True)
            
            events_h5_test = os.path.join(event_dir_test, "events.h5")
            create_dsec_events_h5(events_test, events_h5_test, t_offset=0)
            
            rectify_map_test = os.path.join(event_dir_test, "rectify_map.h5")
            create_identity_rectify_map(self.image_size, rectify_map_test)
            
            result.update({
                'events_h5_test': events_h5_test,
                'rectify_map_test': rectify_map_test,
            })
        else:
            result.update({
                'events_h5_test': None,
                'rectify_map_test': None,
            })
        
        return result
    
    def _generate_flows(self, outdir: str, seq_name: str, split: str = 'train') -> Tuple[int, str, str]:
        """Generate optical flow images for a split.
        
        Args:
            outdir: Output directory
            seq_name: Sequence name
            split: 'train' or 'test'
            
        Returns:
            Tuple of (num_pairs, flow_dir, timestamps_path)
        """
        cfg = self.config
        H, W = self.image_size
        
        # Determine frame range
        if split == 'train':
            split_start_frame = 0
            split_end_frame = int(np.floor(cfg.total_frames * (1.0 - cfg.test_size)))
        else:  # test
            split_start_frame = int(np.floor(cfg.total_frames * (1.0 - cfg.test_size)))
            split_end_frame = cfg.total_frames
        
        # Setup directories
        flow_dir = os.path.join(outdir, "train_optical_flow", seq_name, "flow", "forward")
        ts_path = os.path.join(outdir, "train_optical_flow", seq_name, "flow", "forward_timestamps.txt")
        os.makedirs(flow_dir, exist_ok=True)
        
        # Prepare coordinate grid
        rows = np.arange(H)
        cols = np.arange(W)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        all_coords = np.vstack([cc.ravel(), rr.ravel()]).T.astype(float)
        
        # Generate flows
        ts_rows = []
        idx = 0
        
        flow_dt_us = int(cfg.save_step * cfg.frame_time_us)

        for frame_from in range(split_start_frame, split_end_frame, cfg.save_step):
            frame_to = frame_from + cfg.save_step
            if frame_to > split_end_frame:
                break
            
            flows = self.shape_instance.compute_optical_flow_between_frames(all_coords, frame_from, frame_to)
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
            png_path = os.path.join(flow_dir, f"{idx:06d}.png")
            imageio.imwrite(png_path, img16, format="PNG-FI")
            
            from_ts = idx * flow_dt_us
            to_ts = from_ts + flow_dt_us
            ts_rows.append((from_ts, to_ts))
            
            print(f"[{split}] Saved flow #{idx} : frame {frame_from} -> {frame_to} (valid: {int(valid.sum())} pixels)")
            idx += 1
        
        # Save timestamps
        os.makedirs(os.path.dirname(ts_path), exist_ok=True)
        with open(ts_path, "w") as f:
            f.write("# from_timestamp_us, to_timestamp_us\n")
            for fr, to in ts_rows:
                f.write(f"{fr}, {to}\n")
        
        return idx, flow_dir, ts_path
    
    def _save_metadata(self, outdir: str, seq_name: str, num_train: int, num_test: int) -> None:
        """Save dataset metadata to JSON file."""
        import json
        cfg = self.config
        
        split_start_frame = int(np.floor(cfg.total_frames * (1.0 - cfg.test_size)))
        
        metadata = {
            'seq_name': seq_name,
            'shape_class': self.shape_instance.__class__.__name__,
            'total_frames': cfg.total_frames,
            'image_size': list(self.image_size),
            'save_step': cfg.save_step,
            'frame_time_us': cfg.frame_time_us,
            'flow_dt_us': int(cfg.save_step * cfg.frame_time_us),
            'start_ts_us': cfg.start_ts_us,
            'test_size': cfg.test_size,
            'split_start_frame': split_start_frame,
            'num_flow_pairs_train': num_train,
            'num_flow_pairs_test': num_test,
        }
        
        metadata_path = os.path.join(outdir, "train_optical_flow", seq_name, "dataset_metadata.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved dataset metadata to {metadata_path}")
    
    def _check_existing_dataset(self, outdir: str, seq_name: str) -> bool:
        """Check if dataset already exists and handle accordingly.
        
        Returns:
            True if should skip generation (dataset exists and no force_regenerate)
        """
        cfg = self.config
        
        seq_flow_root = os.path.join(outdir, "train_optical_flow", seq_name)
        seq_event_root = os.path.join(outdir, "train_events", seq_name)
        
        dataset_exists = os.path.exists(seq_flow_root) or os.path.exists(seq_event_root)
        
        if not dataset_exists:
            return False
        
        metadata_path = os.path.join(outdir, "train_optical_flow", seq_name, "dataset_metadata.json")
        
        if cfg.force_regenerate:
            print(f"[force-regenerate] Removing existing dataset: {seq_name}")
            import shutil
            for path in [seq_flow_root, seq_event_root]:
                if os.path.exists(path):
                    shutil.rmtree(path)
            # Also remove test split if it exists
            if cfg.test_size > 0:
                seq_flow_root_test = os.path.join(outdir, "train_optical_flow", f"{seq_name}_test")
                seq_event_root_test = os.path.join(outdir, "train_events", f"{seq_name}_test")
                for path in [seq_flow_root_test, seq_event_root_test]:
                    if os.path.exists(path):
                        shutil.rmtree(path)
            print(f"[force-regenerate] Removed existing dataset, regenerating...")
            return False
        elif os.path.exists(metadata_path):
            print(f"[skip] Dataset already exists at {outdir}/{seq_name}")
            return True
        else:
            print(f"[cleanup] Removing incomplete dataset: {seq_flow_root}")
            import shutil
            for path in [seq_flow_root, seq_event_root]:
                if os.path.exists(path):
                    shutil.rmtree(path)
            return False
    
    def generate(self) -> Dict[str, Any]:
        """Generate the complete dataset.
        
        Returns:
            Dict with output paths and metadata:
                - flow_dir_train: Path to train flow directory
                - timestamps_train: Path to train timestamps file
                - events_h5_train: Path to train events HDF5
                - rectify_map_train: Path to train rectify map
                - num_flow_pairs_train: Number of train flow pairs
                - flow_dir_test: Path to test flow directory (or None)
                - timestamps_test: Path to test timestamps file (or None)
                - events_h5_test: Path to test events HDF5 (or None)
                - rectify_map_test: Path to test rectify map (or None)
                - num_flow_pairs_test: Number of test flow pairs
                - animation_path: Path to animation GIF (if generated)
                - event_animation_path: Path to event animation GIF (if generated)
                - skipped: True if dataset already existed and was skipped
        """
        cfg = self.config
        
        # Pre-select DTD textures if needed (BEFORE hash generation)
        # This ensures the selected texture paths are included in the dataset hash
        self._preselect_dtd_textures()
        
        # Handle auto-naming
        if cfg.auto_name:
            config_hash = generate_dataset_hash(**OmegaConf.to_container(cfg, resolve=True))
            outdir = os.path.join(cfg.outdir, f"variant_{config_hash}")
            print(f"🔸 Auto-naming enabled: Using variant directory '{config_hash}'")
        else:
            outdir = cfg.outdir
            config_hash = None
        
        seq_name = cfg.seq_name
        seq_name_test = f"{seq_name}_test" if cfg.test_size > 0 else None
        
        # Check if dataset already exists
        if self._check_existing_dataset(outdir, seq_name):
            flow_dir_train = os.path.join(outdir, "train_optical_flow", seq_name, "flow", "forward")
            ts_path_train = os.path.join(outdir, "train_optical_flow", seq_name, "flow", "forward_timestamps.txt")
            event_dir_train = os.path.join(outdir, "train_events", seq_name, "events", "left")
            
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
                "outdir": outdir,
            }
        
        # Create shape instance
        self._create_shape_instance()
        
        # Generate events
        events = self._generate_events()
        
        # Split events
        events_train, events_test = self._split_events(events)
        
        # Save events
        event_result = self._save_events(events_train, events_test, outdir, seq_name, seq_name_test)
        
        # Generate flows for train split
        num_train, flow_dir_train, ts_path_train = self._generate_flows(outdir, seq_name, 'train')
        
        # Generate flows for test split
        num_test = 0
        flow_dir_test = None
        ts_path_test = None
        if cfg.test_size > 0 and seq_name_test:
            num_test, flow_dir_test, ts_path_test = self._generate_flows(outdir, seq_name_test, 'test')
        
        print(f"\nDone. Wrote {num_train} train flow PNGs to '{flow_dir_train}'"
              + (f" and {num_test} test flow PNGs to '{flow_dir_test}'" if cfg.test_size > 0 else ""))
        
        # Save metadata
        self._save_metadata(outdir, seq_name, num_train, num_test)
        if seq_name_test:
            self._save_metadata(outdir, seq_name_test, num_train, num_test)
        
        result = {
            "flow_dir_train": flow_dir_train,
            "timestamps_train": ts_path_train,
            "num_flow_pairs_train": num_train,
            "flow_dir_test": flow_dir_test,
            "timestamps_test": ts_path_test,
            "num_flow_pairs_test": num_test,
            "skipped": False,
            "outdir": outdir,  # Include the actual outdir used (may include variant folder)
        }
        result.update(event_result)
        
        if config_hash:
            result['config_hash'] = config_hash
            result['dataset_variant'] = f"variant_{config_hash}"
        
        return result
    
    def generate_animations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate animations after dataset generation.
        
        Args:
            result: Result dict from generate()
            
        Returns:
            Updated result dict with animation paths
        """
        cfg = self.config
        
        if result.get('skipped', False):
            return result
        
        outdir = result['outdir']
        seq_name = cfg.seq_name
        
        # Generate shape movement animation
        if cfg.generate_animation:
            print("\n" + "="*60)
            print("Generating shape movement animation...")
            print("="*60)
            
            has_textures = cfg.foreground_texture is not None or cfg.background_texture is not None
            
            animation_dir = os.path.join(outdir, "train_optical_flow", seq_name, "animation")
            os.makedirs(animation_dir, exist_ok=True)
            animation_path = os.path.join(animation_dir, f"{seq_name}_movement.gif")
            
            if has_textures:
                print("Using textured rendering for animation...")
                frames = self.shape_instance.create_animation_with_textures(
                    frame_step=cfg.animation_frame_step,
                    fps=cfg.animation_fps
                )
                imageio.mimsave(animation_path, frames, fps=cfg.animation_fps, loop=0)
            else:
                interval_ms = int(1000 / cfg.animation_fps)
                anim = self.shape_instance.create_animation(
                    frame_step=cfg.animation_frame_step,
                    interval=interval_ms
                )
                anim.save(animation_path, writer='pillow', fps=cfg.animation_fps)
            
            print(f"✓ Shape movement animation saved: {animation_path}")
            result['animation_path'] = animation_path
        
        # Generate event animation
        if cfg.generate_event_animation:
            print("\n" + "="*60)
            print("Generating event animation...")
            print("="*60)
            
            from create_dataset_generic import create_event_animation
            import h5py
            
            events_h5_path = result['events_h5_train']
            print(f"Loading events from {events_h5_path}...")
            
            with h5py.File(events_h5_path, 'r') as f:
                x = f['events/x'][:]
                y = f['events/y'][:]
                t = f['events/t'][:]
                p = f['events/p'][:]
            
            events_to_animate = np.zeros(len(x), dtype=[
                ('x', np.int16),
                ('y', np.int16),
                ('t', np.int64),
                ('p', np.bool_)
            ])
            events_to_animate['x'] = x
            events_to_animate['y'] = y
            events_to_animate['t'] = t
            events_to_animate['p'] = p.astype(np.bool_)
            
            print(f"Loaded {len(events_to_animate)} events from train split")
            
            animation_dir = os.path.join(outdir, "train_optical_flow", seq_name, "animation")
            os.makedirs(animation_dir, exist_ok=True)
            event_animation_path = os.path.join(animation_dir, f"{seq_name}_events.gif")
            
            create_event_animation(
                events=events_to_animate,
                image_size=self.image_size,
                output_path=event_animation_path,
                fps=cfg.event_animation_fps,
                accumulation_time_ms=cfg.event_accumulation_ms,
                frame_time_us=cfg.frame_time_us,
            )
            
            result['event_animation_path'] = event_animation_path
        
        return result
    
    def generate_sanity_checks(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sanity check visualizations.
        
        Args:
            result: Result dict from generate()
            
        Returns:
            Updated result dict
        """
        cfg = self.config
        
        if not cfg.sanity_check or result.get('skipped', False):
            return result
        
        from create_dataset_generic import generate_sanity_check_figures
        
        print("\n" + "="*60)
        print("Running sanity checks...")
        print("="*60)
        
        outdir = result['outdir']
        seq_name = cfg.seq_name
        
        print(f"\n[SANITY CHECK] Train split: {seq_name}")
        n_train = generate_sanity_check_figures(
            shape_instance=self.shape_instance,
            seq_name=seq_name,
            outdir=outdir,
            image_size=self.image_size,
            total_frames=cfg.total_frames,
            save_step=cfg.save_step,
            test_size=cfg.test_size,
            split="train",
        )
        print(f"[SANITY CHECK] Generated {n_train} train sanity figures")
        
        if cfg.test_size > 0:
            seq_name_test = seq_name + "_test"
            print(f"\n[SANITY CHECK] Test split: {seq_name_test}")
            n_test = generate_sanity_check_figures(
                shape_instance=self.shape_instance,
                seq_name=seq_name_test,
                outdir=outdir,
                image_size=self.image_size,
                total_frames=cfg.total_frames,
                save_step=cfg.save_step,
                test_size=cfg.test_size,
                split="test",
            )
            print(f"[SANITY CHECK] Generated {n_test} test sanity figures")
        
        print("\n" + "="*60)
        print("Sanity checks completed!")
        print("="*60)
        
        return result
