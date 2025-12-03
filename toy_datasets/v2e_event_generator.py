"""
V2E-based event generator for realistic DVS event simulation.

V2E is an optional dependency. Install it with:
    git submodule update --init --recursive
    cd external/v2e && pip install -e .
"""
import os
import sys
import numpy as np
from pathlib import Path


def check_v2e_available():
    """Check if v2e is installed and importable."""
    try:
        import v2ecore
        return True
    except ImportError:
        return False


def get_v2e_path():
    """Get path to v2e submodule if it exists."""
    # Try to find v2e in submodule location
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    v2e_submodule = repo_root / "external" / "v2e"
    
    if v2e_submodule.exists():
        return str(v2e_submodule)
    
    return None


def ensure_v2e_imported():
    """
    Ensure v2e can be imported, adding submodule path if needed.
    Raises ImportError with helpful message if v2e not available.
    """
    if check_v2e_available():
        return True
    
    # Try to add v2e from submodule
    v2e_path = get_v2e_path()
    if v2e_path and v2e_path not in sys.path:
        sys.path.insert(0, v2e_path)
    
    if check_v2e_available():
        return True
    
    # v2e not available - provide helpful error
    error_msg = (
        "\nv2e is not installed. To use v2e event generation:\n\n"
        "Option 1 - Install v2e from submodule:\n"
        "  git submodule update --init --recursive\n"
        "  cd external/v2e\n"
        "  pip install -e .\n\n"
        "Option 2 - Install v2e directly:\n"
        "  git clone https://github.com/SensorsINI/v2e.git\n"
        "  cd v2e\n"
        "  pip install -e .\n\n"
        "Option 3 - Use synthetic event generation (default):\n"
        "  python create_dataset_generic.py --event-generation-method synthetic\n"
    )
    raise ImportError(error_msg)


class V2EEventGenerator:
    """
    Wrapper for v2e event generation from rendered frames.
    
    This provides a simpler interface to v2e's EventEmulator for
    generating realistic DVS events from synthetic video frames.
    """
    
    def __init__(
        self,
        image_size,
        pos_thres=0.5,
        neg_thres=0.5,
        sigma_thres=0.0,
        cutoff_hz=0,
        leak_rate_hz=0,
        shot_noise_rate_hz=0,
        refractory_period_s=0,
        seed=0,
        photoreceptor_noise=False,
        leak_jitter_fraction=0,
        noise_rate_cov_decades=0,
        **kwargs
    ):
        """
        Initialize v2e event emulator.
        
        Args:
            image_size: (H, W) tuple
            pos_thres: Positive threshold (log_e intensity change to trigger event)
            neg_thres: Negative threshold (log_e intensity change)
            sigma_thres: Threshold variation stddev (pixel-to-pixel variation)
            cutoff_hz: Photoreceptor lowpass filter cutoff frequency
            leak_rate_hz: Leak event rate per pixel (background noise)
            shot_noise_rate_hz: Shot noise rate (temporal noise)
            refractory_period_s: Refractory period in seconds
            seed: Random seed for reproducibility (0=random, >0=fixed)
            photoreceptor_noise: Use photoreceptor noise model (more realistic)
            leak_jitter_fraction: Leak event timing jitter (fraction of interval)
            noise_rate_cov_decades: Spatial variation in noise rates (decades)
            **kwargs: Additional v2e parameters
        """
        # Ensure v2e is available
        ensure_v2e_imported()
        
        # Import v2e after ensuring it's available
        from v2ecore.emulator import EventEmulator
        
        self.image_size = image_size
        self.H, self.W = image_size
        
        # Store parameters
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres
        self.sigma_thres = sigma_thres
        
        # Initialize v2e emulator
        self.emulator = EventEmulator(
            pos_thres=pos_thres,
            neg_thres=neg_thres,
            sigma_thres=sigma_thres,
            cutoff_hz=cutoff_hz,
            leak_rate_hz=leak_rate_hz,
            shot_noise_rate_hz=shot_noise_rate_hz,
            refractory_period_s=refractory_period_s,
            seed=seed,
            photoreceptor_noise=photoreceptor_noise,
            leak_jitter_fraction=leak_jitter_fraction,
            noise_rate_cov_decades=noise_rate_cov_decades,
            output_folder=None,  # We handle output ourselves
            dvs_h5=None,
            dvs_aedat2=None,
            dvs_text=None,
        )
        
        print(f"Initialized V2E event generator:")
        print(f"  Image size: {self.W}x{self.H}")
        print(f"  Thresholds: +{pos_thres}, -{neg_thres} (sigma={sigma_thres})")
        print(f"  Cutoff: {cutoff_hz} Hz")
        print(f"  Leak rate: {leak_rate_hz} Hz/pixel")
        print(f"  Shot noise: {shot_noise_rate_hz} Hz")
        
    def generate_events_from_frames(self, frames, frame_times_us):
        """
        Generate events from a sequence of frames using v2e.
        
        Args:
            frames: Array of frames [T, H, W] with values 0-255 (uint8)
            frame_times_us: Array of frame timestamps in microseconds
            
        Returns:
            events: Structured numpy array with dtype [('x', 'y', 't', 'p')]
                    where t is in microseconds
        """
        if len(frames) < 2:
            print("[warning] Need at least 2 frames to generate events")
            return np.array([], dtype=[('x', np.int16), ('y', np.int16), 
                                      ('t', np.int64), ('p', bool)])
        
        print(f"Generating events from {len(frames)} frames using v2e...")
        print(f"  Time range: {frame_times_us[0]:.0f} - {frame_times_us[-1]:.0f} μs")
        print(f"  Frame interval: {np.mean(np.diff(frame_times_us)):.1f} μs (avg)")
        
        all_events = []
        
        # Process frames sequentially through v2e emulator
        from tqdm import tqdm
        for i in tqdm(range(len(frames)), desc="V2E event generation"):
            # v2e expects frames as float [0, 255] or normalized [0, 1]
            frame = frames[i].astype(np.float32)
            
            # Convert microseconds to seconds
            t_frame_s = frame_times_us[i] / 1e6
            
            # Generate events for this frame
            # Note: v2e's generate_events internally compares with previous frame
            events = self.emulator.generate_events(frame, t_frame_s)
            
            if events is not None and len(events) > 0:
                all_events.append(events)
        
        # Concatenate all events
        if all_events:
            # Events from v2e are in format [t, x, y, p] where:
            # t = timestamp in seconds
            # x, y = pixel coordinates
            # p = polarity (+1 for ON, -1 for OFF)
            events_raw = np.concatenate(all_events)
            
            # Convert to our format: structured array with ('x', 'y', 't', 'p')
            # where t is in microseconds and p is boolean
            events = np.zeros(len(events_raw), dtype=[
                ('x', np.int16),
                ('y', np.int16),
                ('t', np.int64),
                ('p', bool)
            ])
            
            events['t'] = (events_raw[:, 0] * 1e6).astype(np.int64)  # Convert seconds to microseconds
            events['x'] = events_raw[:, 1].astype(np.int16)
            events['y'] = events_raw[:, 2].astype(np.int16)
            events['p'] = events_raw[:, 3] > 0  # Convert +1/-1 to True/False
            
            print(f"Generated {len(events)} events with v2e")
            return events
        else:
            print("[warning] No events generated by v2e")
            return np.array([], dtype=[('x', np.int16), ('y', np.int16), 
                                      ('t', np.int64), ('p', bool)])
    
    def __repr__(self):
        return (f"V2EEventGenerator(size={self.W}x{self.H}, "
                f"thres=±{self.pos_thres}, sigma={self.sigma_thres})")


# Convenience function for checking availability
def is_v2e_available():
    """
    Check if v2e is available for use.
    
    Returns:
        bool: True if v2e can be imported, False otherwise
    """
    return check_v2e_available() or get_v2e_path() is not None
