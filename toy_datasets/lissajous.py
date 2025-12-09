"""
Lissajous curve movement - non-periodic complex trajectories.

A Lissajous curve is defined by parametric equations:
    x(t) = A * sin(a*t + δ)
    y(t) = B * sin(b*t)

When a/b is irrational, the curve never repeats (non-periodic).
Common interesting ratios: 3:2, 5:4, 7:5, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from shape_movement import ShapeMovementBase


class LissajousMovement(ShapeMovementBase):
    """Shape moving along a Lissajous curve with rotation."""
    
    def __init__(
        self, 
        total_frames=2000,
        image_size=(256, 256),
        face_color='cyan',
        # Shape parameters
        shape_type='circle',  # 'circle', 'square', 'star', 'hexagon', 'polygon'
        shape_size=20,
        # Star-specific parameters
        star_num_points=5,     # number of points for star
        star_inner_ratio=0.5,  # ratio of inner to outer radius (0.0-1.0)
        # Polygon-specific parameters
        polygon_num_sides=6,   # number of sides for regular polygon
        # Lissajous parameters
        freq_ratio_a=3,  # frequency ratio numerator
        freq_ratio_b=2,  # frequency ratio denominator
        phase_shift=np.pi/2,  # δ in the equation
        amplitude_scale=0.35,  # scale relative to image size
        # Rotation parameters
        rotation_speed=2.0,  # rotations per full trajectory
        # Texture parameters
        foreground_texture=None,
        background_texture=None,
        foreground_texture_params=None,
        background_texture_params=None,
        frame_time_us=1000,
        **kwargs
    ):
        self.shape_type = shape_type
        self.shape_size = shape_size
        self.star_num_points = star_num_points
        self.star_inner_ratio = star_inner_ratio
        self.polygon_num_sides = polygon_num_sides
        self.freq_ratio_a = freq_ratio_a
        self.freq_ratio_b = freq_ratio_b
        self.phase_shift = phase_shift
        self.amplitude_scale = amplitude_scale
        self.rotation_speed = rotation_speed
        
        super().__init__(total_frames, image_size, face_color,
                        foreground_texture=foreground_texture,
                        background_texture=background_texture,
                        foreground_texture_params=foreground_texture_params,
                        background_texture_params=background_texture_params,
                        frame_time_us=frame_time_us)
    
    @property
    def shape_name(self):
        return f"{self.shape_type.capitalize()}"
    
    @property
    def shape_description(self):
        return f"A {self.shape_type} with size {self.shape_size} moving along a Lissajous curve"
    
    def trajectory_description(self):
        return (
            f"Lissajous curve with frequency ratio {self.freq_ratio_a}:{self.freq_ratio_b}, "
            f"phase shift {self.phase_shift:.2f}, amplitude scale {self.amplitude_scale:.2f}"
        )
    
    @property
    def trajectory_name(self):
        return f"Lissajous {self.freq_ratio_a}:{self.freq_ratio_b}"
    
    def define_shape(self):
        """Create the shape centered at origin."""
        if self.shape_type == 'circle':
            # Circle as polygon approximation
            n_points = 32
            angles = np.linspace(0, 2 * np.pi, n_points + 1)
            x = self.shape_size * np.cos(angles)
            y = self.shape_size * np.sin(angles)
            vertices = np.vstack([x, y]).T
            codes = [Path.MOVETO] + [Path.LINETO] * (n_points - 1) + [Path.CLOSEPOLY]
            
        elif self.shape_type == 'square':
            s = self.shape_size
            vertices = np.array([
                [-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]
            ])
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            
        elif self.shape_type == 'star':
            # Configurable n-pointed star
            n_points = self.star_num_points
            outer_r = self.shape_size
            inner_r = self.shape_size * self.star_inner_ratio
            angles = np.linspace(0, 2 * np.pi, n_points * 2 + 1)
            radii = np.empty_like(angles)
            radii[::2] = outer_r
            radii[1::2] = inner_r
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            vertices = np.vstack([x, y]).T
            codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
            
        elif self.shape_type == 'hexagon':
            # Hexagon (backward compatible - use 6-sided polygon)
            n_points = 6
            angles = np.linspace(0, 2 * np.pi, n_points + 1)
            x = self.shape_size * np.cos(angles)
            y = self.shape_size * np.sin(angles)
            vertices = np.vstack([x, y]).T
            codes = [Path.MOVETO] + [Path.LINETO] * (n_points - 1) + [Path.CLOSEPOLY]
            
        elif self.shape_type == 'polygon':
            # General n-sided regular polygon
            n_points = self.polygon_num_sides
            angles = np.linspace(0, 2 * np.pi, n_points + 1)
            x = self.shape_size * np.cos(angles)
            y = self.shape_size * np.sin(angles)
            vertices = np.vstack([x, y]).T
            codes = [Path.MOVETO] + [Path.LINETO] * (n_points - 1) + [Path.CLOSEPOLY]
            
        else:
            raise ValueError(f"Unknown shape type: {self.shape_type}")
        
        anchor_point = np.array([0, 0])
        return Path(vertices, codes), anchor_point
    
    def trajectory_at(self, frame):
        """
        Lissajous curve trajectory with rotation.
        
        x(t) = A * sin(a*t + δ)
        y(t) = B * sin(b*t)
        """
        # Normalize time to [0, 2π]
        # For non-periodic behavior, use longer time range
        t = 2 * np.pi * frame / self.total_frames
        
        # Center of image
        center = np.array([self.image_size[1] / 2, self.image_size[0] / 2])
        
        # Amplitudes based on image size
        A = self.image_size[1] * self.amplitude_scale
        B = self.image_size[0] * self.amplitude_scale
        
        # Frequency ratios
        a = self.freq_ratio_a
        b = self.freq_ratio_b
        
        # Lissajous parametric equations
        x = A * np.sin(a * t + self.phase_shift)
        y = B * np.sin(b * t)
        position = center + np.array([x, y])
        
        # Rotation
        rotation = self.rotation_speed * t
        
        # Velocities (derivatives)
        dt = 2 * np.pi / self.total_frames
        vx = A * a * np.cos(a * t + self.phase_shift) * dt
        vy = B * b * np.cos(b * t) * dt
        linear_velocity = np.array([vx, vy])
        
        angular_velocity = self.rotation_speed * dt
        
        return position, rotation, linear_velocity, angular_velocity


class MultiShapeLissajous(ShapeMovementBase):
    """Multiple shapes moving along different Lissajous curves."""
    
    def __init__(
        self,
        total_frames=2000,
        image_size=(256, 256),
        num_shapes=3,
        foreground_texture=None,
        background_texture=None,
        foreground_texture_params=None,
        background_texture_params=None,
        frame_time_us=1000,
        **kwargs
    ):
        self.num_shapes = num_shapes
        # Create multiple shape instances
        self.shapes = []
        
        # Define different parameters for each shape
        configs = [
            {'shape_type': 'circle', 'freq_ratio_a': 3, 'freq_ratio_b': 2, 'face_color': 'red'},
            {'shape_type': 'square', 'freq_ratio_a': 5, 'freq_ratio_b': 3, 'face_color': 'blue'},
            {'shape_type': 'star', 'freq_ratio_a': 7, 'freq_ratio_b': 5, 'face_color': 'green'},
            {'shape_type': 'hexagon', 'freq_ratio_a': 4, 'freq_ratio_b': 3, 'face_color': 'yellow'},
        ]
        
        for i in range(num_shapes):
            config = configs[i % len(configs)].copy()
            config.update({
                'total_frames': total_frames,
                'image_size': image_size,
                'shape_size': 15 + i * 5,  # Vary sizes
                'phase_shift': i * np.pi / 4,  # Vary phases
                'amplitude_scale': 0.3 - i * 0.05,  # Vary amplitudes
                'frame_time_us': frame_time_us,
            })
            self.shapes.append(LissajousMovement(**config))
        
        # Use first shape's parameters for base class
        super().__init__(total_frames, image_size, self.shapes[0].face_color,
                        foreground_texture=foreground_texture,
                        background_texture=background_texture,
                        foreground_texture_params=foreground_texture_params,
                        background_texture_params=background_texture_params,
                        frame_time_us=frame_time_us)
    
    @property
    def shape_name(self):
        return f"{self.num_shapes} Shapes"
    
    @property
    def shape_description(self):
        return f"{self.num_shapes} different shapes moving along different Lissajous curves"
    
    def trajectory_description(self):
        return "Multiple independent Lissajous trajectories"
    
    @property
    def trajectory_name(self):
        return "Multi-Lissajous"
    
    def define_shape(self):
        """This won't be used for multi-shape, but required by base class."""
        return self.shapes[0].define_shape()
    
    def trajectory_at(self, frame):
        """This won't be used for multi-shape, but required by base class."""
        return self.shapes[0].trajectory_at(frame)
    
    def update_shape(self, frame):
        """Update all shapes for the current frame."""
        for shape in self.shapes:
            shape.update_shape(frame)
    
    def generate_events(self):
        """Generate events from all shapes combined."""
        all_events = []
        for shape in self.shapes:
            events = shape.generate_events()
            all_events.append(events)
        
        if len(all_events) > 0:
            combined = np.concatenate(all_events)
            # Sort by time
            sorted_idx = np.argsort(combined['t'])
            return combined[sorted_idx]
        return np.array([], dtype=[('x', np.int16), ('y', np.int16), ('t', np.float64), ('p', bool)])
    
    def compute_optical_flow_between_frames(self, pixels, frame_from, frame_to):
        """Compute flow considering all shapes."""
        pixels = np.asarray(pixels, dtype=float)
        flows = np.full((len(pixels), 2), np.nan, dtype=np.float32)
        
        # Check each shape and take the first valid flow for each pixel
        for shape in self.shapes:
            shape_flow = shape.compute_optical_flow_between_frames(pixels, frame_from, frame_to)
            # Fill in NaN values in flows with valid values from shape_flow
            nan_mask = np.isnan(flows[:, 0])
            valid_mask = ~np.isnan(shape_flow[:, 0])
            update_mask = nan_mask & valid_mask
            flows[update_mask] = shape_flow[update_mask]
        
        return flows


if __name__ == "__main__":
    # Example usage
    print("Creating Lissajous movement example...")
    
    # Single shape with Lissajous trajectory
    lissajous = LissajousMovement(
        total_frames=5000,
        image_size=(256, 256),
        shape_type='star',
        freq_ratio_a=5,
        freq_ratio_b=3,
        phase_shift=np.pi/4,
        rotation_speed=2.0
    )
    
    # Create animation
    anim = lissajous.create_animation(frame_step=5, interval=50)
    anim.save('lissajous_movement.gif', writer='pillow', fps=20)
    print("Saved animation to lissajous_movement.gif")
    
    # Generate events
    events = lissajous.generate_events()
    print(f"Generated {len(events)} events")
    
    # Multiple shapes example
    multi = MultiShapeLissajous(total_frames=500, num_shapes=3)
    events_multi = multi.generate_events()
    print(f"Generated {len(events_multi)} events from {multi.num_shapes} shapes")
