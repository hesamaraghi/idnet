import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import json

from shape_movement import ShapeMovementBase

class StarMovement(ShapeMovementBase):
    
    def __init__(
        self, 
        total_frames=200, 
        image_size=(256, 256), 
        face_color='gold', 
        num_points=5, 
        outer_radius=40, 
        inner_radius=20, 
        number_of_rotations=2,
        foreground_texture=None,
        background_texture=None,
        foreground_texture_params=None,
        background_texture_params=None,
        frame_time_us=1000,
        **kwargs
    ):
        params = {
            "total_frames": total_frames,
            "image_size": image_size,
            "face_color": face_color,
            "num_points": num_points,
            "outer_radius": outer_radius,
            "inner_radius": inner_radius,
            "number_of_rotations": number_of_rotations,
            "foreground_texture": foreground_texture,
            "background_texture": background_texture,
            "foreground_texture_params": foreground_texture_params,
            "background_texture_params": background_texture_params,
            "frame_time_us": frame_time_us,
        }
        if kwargs:
            params.update(kwargs)
        print("StarMovement init params:")
        print(json.dumps(params, indent=2, default=str))
        
        
        self.num_points = num_points
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.number_of_rotations = number_of_rotations
        super().__init__(
            total_frames, 
            image_size, 
            face_color,
            foreground_texture=foreground_texture,
            background_texture=background_texture,
            foreground_texture_params=foreground_texture_params,
            background_texture_params=background_texture_params,
            frame_time_us=frame_time_us
        )
        
    @property
    def shape_name(self):
        return f"{self.num_points}-pointed Star"
    @property
    def shape_description(self):
        return (
            f"A {self.num_points}-pointed star that moves in a figure-eight (lemniscate) path "
            "while rotating around its center."
        )
    def trajectory_description(self):
        return (
            "The star moves along a figure-eight (lemniscate) path centered in the image. "
            "It completes a specified number of rotations around its center as it moves."
        )   
    @property
    def trajectory_name(self):
        return "Figure-Eight (Lemniscate) Path"
    
    def define_shape(self):
        """
        Create a 5-pointed star centered at origin.
        """
        angles = np.linspace(0, 2 * np.pi, self.num_points * 2 + 1)
        radii = np.empty_like(angles)
        radii[::2] = self.outer_radius
        radii[1::2] = self.inner_radius

        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        vertices = np.vstack([x, y]).T

        codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
        anchor_point = np.array([0, 0])  # Center of the star

        return Path(vertices, codes), anchor_point

    def trajectory_at(self, frame):
        """
        Movement in a figure-eight (lemniscate) path with rotation.
        """
        t = 2 * np.pi * frame / self.total_frames  # normalized to [0, 2pi]

        A = self.image_size[1] / 3  # horizontal radius
        B = self.image_size[0] / 3  # vertical radius
        center = np.array([self.image_size[1] / 2, self.image_size[0] / 2])

        # Lemniscate of Gerono
        x = A * np.sin(t)
        y = B * np.sin(2 * t) / 2
        position = center + np.array([x, y])

        # Rotation based on time
        rotation = self.number_of_rotations * t  # 2 full rotations over the course

        # Estimate velocities using finite differences
        dt = 2 * np.pi / self.total_frames
        linear_velocity = np.array([
            A * np.cos(t),
            B * np.cos(2 * t)
        ]) * dt

        angular_velocity = self.number_of_rotations * dt  # derivative of 2*t w.r.t t

        return position, rotation, linear_velocity, angular_velocity