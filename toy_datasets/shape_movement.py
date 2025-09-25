import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.animation as animation
from abc import ABC, abstractmethod
from tqdm import tqdm

class ShapeMovementBase(ABC):
    def __init__(self, total_frames, image_size, face_color='blue'):
        self.total_frames = total_frames
        self.image_size = image_size  # (height, width)
        self.face_color = face_color

        self.shape_path, self.anchor_point = self.define_shape()
        self.cached_vertices = self.shape_path.vertices.copy()  # store original vertices
        self.transformed_path = None  # will be updated per frame

    @abstractmethod
    def define_shape(self):
        """
        Should return:
            - shape_path: matplotlib.path.Path object
            - anchor_point: np.array([x, y])
        """
        pass

    @abstractmethod
    def trajectory_at(self, frame):
        """
        Should return:
            - position: np.array([x, y])
            - rotation: float (radians)
            - linear_velocity: np.array([vx, vy])
            - angular_velocity: float (radians/frame)
        """
        pass

    def update_shape(self, frame):
        """
        Applies transformation (rotation + translation) to the shape based on trajectory.
        """
        position, rotation, _, _ = self.trajectory_at(frame)
        # Translate shape to origin using anchor point
        vertices = self.cached_vertices - self.anchor_point
        # Rotate
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        rotated = vertices @ rotation_matrix.T
        # Translate to new position
        transformed = rotated + position
        self.transformed_path = Path(transformed, self.shape_path.codes)

    def generate_frame(self, frame):
        """
        Renders the current frame with the transformed shape.
        """
        self.update_shape(frame)
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.image_size[1])
        ax.set_ylim(0, self.image_size[0])
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

        patch = PathPatch(self.transformed_path, facecolor=self.face_color, lw=2)
        ax.add_patch(patch)

        plt.close(fig)
        return fig
    
    def create_animation(self, frame_step=1, interval=50):
        """
        Create a matplotlib animation of the shape movement.
        
        Args:
            frame_step (int): Skip frames by this amount (e.g., 5 means animate every 5th frame)
            interval (int): Delay between frames in milliseconds for animation
        
        Returns:
            matplotlib.animation.FuncAnimation object
        """
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

        ax.set_xlim(0, self.image_size[1])
        ax.set_ylim(0, self.image_size[0])
        ax.set_aspect('equal')
        ax.axis('off')

        # Initial drawing
        self.update_shape(0)
        patch = PathPatch(self.transformed_path, facecolor=self.face_color, lw=2)
        ax.add_patch(patch)

        def update(frame):
            self.update_shape(frame)
            patch.set_path(self.transformed_path)
            return (patch,)

        frames = range(0, self.total_frames, frame_step)
        anim = animation.FuncAnimation(
            fig, update, frames=frames, interval=interval, blit=True
        )
        plt.close(fig)
        return anim


    def compute_optical_flow(self, pixels, frame):
        """
        Compute optical flow for each pixel.
        Args:
            pixels: Nx2 array of (x, y) coordinates
            frame: frame number
        Returns:
            optical_flow: Nx2 array of (vx, vy)
        """
        position, rotation, linear_velocity, angular_velocity = self.trajectory_at(frame)

        self.update_shape(frame)  # ensure transformed_path is updated

        pixels = np.asarray(pixels)

        relative_pos = pixels - position

        # Angular component: v = ω × r = angular_velocity * [-dy, dx]
        angular_component = angular_velocity * np.stack([-relative_pos[:, 1], relative_pos[:, 0]], axis=-1)

        flow = linear_velocity + angular_component

        return flow

    def generate_events(self):
        """
        Generate events with optical flow at each activated pixel.

        Returns:
            np.ndarray with dtype [('x', int16), ('y', int16), ('t', int64), 
                                ('p', bool), ('v_x', float32), ('v_y', float32)]
        """
        img_height, img_width = self.image_size
        yy, xx = np.meshgrid(np.arange(img_height), np.arange(img_width), indexing='ij')
        xy_flatten = np.vstack([xx.ravel(), yy.ravel()])
        all_coords = xy_flatten.T  # shape: (H*W, 2)

        xs, ys, ts, ps, vxs, vys = [], [], [], [], [], []
        prev_indices = np.zeros(img_width * img_height, dtype=bool)

        for frame in tqdm(range(self.total_frames), desc="Generating Events"):
            self.update_shape(frame)
            path = self.transformed_path
            new_indices = path.contains_points(all_coords)

            # Detect positive (on) and negative (off) transitions
            pos_idx = np.where((~prev_indices) & new_indices)[0]
            neg_idx = np.where(prev_indices & (~new_indices))[0]

            for indices, polarity in [(pos_idx, True), (neg_idx, False)]:
                if len(indices) == 0:
                    continue
                coords = all_coords[indices]
                flows = self.compute_optical_flow(coords, frame)

                xs.append(coords[:, 0])
                ys.append(coords[:, 1])
                ts.append(np.full(len(indices), frame, dtype=np.int64))
                ps.append(np.full(len(indices), polarity, dtype=bool))
                vxs.append(flows[:, 0])
                vys.append(flows[:, 1])

            prev_indices = new_indices.copy()
            
        if xs:
            xs = np.concatenate(xs).astype(np.int16)
            ys = np.concatenate(ys).astype(np.int16)
            ts = np.concatenate(ts)
            ps = np.concatenate(ps)
            vxs = np.concatenate(vxs).astype(np.float32)
            vys = np.concatenate(vys).astype(np.float32)

            events = np.zeros(len(xs), dtype=[
                ('x', np.int16),
                ('y', np.int16),
                ('t', np.int64),
                ('p', bool),
                ('v_x', np.float32),
                ('v_y', np.float32),
            ])
            events['x'] = xs
            events['y'] = ys
            events['t'] = ts
            events['p'] = ps
            events['v_x'] = vxs
            events['v_y'] = vys
            return events
        else:
            return np.zeros(0, dtype=[
                ('x', np.int16),
                ('y', np.int16),
                ('t', np.int64),
                ('p', bool),
                ('v_x', np.float32),
                ('v_y', np.float32),
            ])
