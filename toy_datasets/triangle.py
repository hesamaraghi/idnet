import sys
import types

# create fake mkl module
sys.modules['mkl'] = types.ModuleType('mkl')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from shape_movement import ShapeMovementBase


class TriangleMovement(ShapeMovementBase):
    def __init__(
        self,
        total_frames,
        image_size,
        triangle_base,
        triangle_height,
        added_height,
        start_pos,
        face_color="blue",
        speed=1.0, 
        movement_direction=0.0,
        frame_time_us=1000,
    ):
        self.speed = speed
        self.movement_direction = movement_direction
        self.movement_direction = np.deg2rad(self.movement_direction)
        self.triangle_base = triangle_base
        self.triangle_height = triangle_height
        self.added_height = added_height
        self.start_pos = start_pos
        super().__init__(total_frames, image_size, face_color, frame_time_us=frame_time_us)

    def define_shape(self):
        vertices = np.array(
            [
                [0, 0],
                [-self.triangle_base / 2, -self.triangle_height],
                [self.triangle_base / 2, -self.triangle_height],
                [0, 0],
            ]
        )
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        hanger_point = np.array([0, 0])
        return Path(vertices, codes), hanger_point

    def trajectory_at(self, frame):
        num_frames = self.total_frames
        # Reverse direction at midpoint

        if frame >= num_frames / 2:
            partial_movement = (num_frames - frame) / (num_frames / 2)
            direction = -1
        else:
            partial_movement = frame / (num_frames / 2)
            direction = 1

        y_position = partial_movement * self.added_height
        x_position = partial_movement * self.added_height * np.tan(self.movement_direction)
        rotation = 0
        angular_velocity = 0

        v_x = self.added_height * np.tan(self.movement_direction) / num_frames * 2
        v_y = self.added_height / num_frames * 2
        linear_velocity = np.array([v_x, v_y])
        # Apply direction
        linear_velocity *= direction

        position = np.array([x_position, y_position]) + self.start_pos

        return position, rotation, linear_velocity, angular_velocity

    def change_event_speeds(self):
        events = self.generate_events()
        # Add velocity to events
        if len(events) > 0:
            events["v_x"] *= self.speed
            events["v_y"] *= self.speed
            events["t"] = np.float64(events["t"]) / self.speed
        return events
