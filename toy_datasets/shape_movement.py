import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.animation as animation
from abc import ABC, abstractmethod
from tqdm import tqdm

class ShapeMovementBase(ABC):
    def __init__(self, total_frames, image_size, face_color='blue', 
                 foreground_texture=None, background_texture=None,
                 foreground_texture_params=None, background_texture_params=None):
        """
        Initialize shape movement with optional textures.
        
        Args:
            total_frames: Number of frames in the sequence
            image_size: (height, width) tuple
            face_color: Default color for the shape (used if no foreground texture)
            foreground_texture: Type of texture for the shape ('solid', 'noise', 'gradient', 'checkerboard', None)
            background_texture: Type of texture for the background ('solid', 'noise', 'gradient', 'checkerboard', None)
            foreground_texture_params: Dict with texture-specific parameters
            background_texture_params: Dict with texture-specific parameters
        """
        self.total_frames = total_frames
        self.image_size = image_size  # (height, width)
        self.face_color = face_color
        
        # Texture settings
        self.foreground_texture = foreground_texture
        self.background_texture = background_texture
        self.foreground_texture_params = foreground_texture_params or {}
        self.background_texture_params = background_texture_params or {}
        
        # Texture caching (fix issue #5)
        self._fg_texture_cache = None
        self._bg_texture_cache = None
        self._fg_texture_cache_size = None
        self._bg_texture_cache_size = None

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

    def create_animation_with_textures(self, frame_step=1, fps=20):
        """
        Create animation using render_frame() which supports textures.
        
        Args:
            frame_step (int): Skip frames by this amount (e.g., 5 means animate every 5th frame)
            fps (int): Frames per second for the output animation
        
        Returns:
            List of frames as numpy arrays (H, W, 3) RGB
        """
        frames = []
        frame_indices = range(0, self.total_frames, frame_step)
        
        print(f"Rendering {len(list(frame_indices))} frames...")
        for frame_num in tqdm(frame_indices):
            # Render grayscale frame
            gray_frame = self.render_frame(frame_num)
            
            # Convert to RGB for animation
            rgb_frame = np.stack([gray_frame, gray_frame, gray_frame], axis=-1)
            frames.append(rgb_frame)
        
        return frames


    def compute_optical_flow(self, pixels, frame):
        """
        Compute optical flow for each pixel.
        Args:
            pixels: Nx2 array of (x, y) coordinates
            frame: frame number
        Returns:
            optical_flow: Nx2 array of (vx, vy) in units of pixels/frame
        """
        position, rotation, linear_velocity, angular_velocity = self.trajectory_at(frame)

        self.update_shape(frame)  # ensure transformed_path is updated

        pixels = np.asarray(pixels)

        relative_pos = pixels - position

        # Angular component: v = ω × r = angular_velocity * [-dy, dx]
        angular_component = angular_velocity * np.stack(
            [-relative_pos[:, 1], relative_pos[:, 0]], axis=-1
        )

        flow = linear_velocity + angular_component

        return flow

    def compute_optical_flow_between_frames(self, pixels, frame_from, frame_to):
        """
        Compute the displacement (optical flow) of shape points between two frames
        by mapping rigid-body coordinates from frame_from to frame_to.

        Only defined for points that lie inside the shape at frame_from. For
        points outside the shape at frame_from the returned flow will be (np.nan, np.nan).

        Args:
            pixels: (N,2) array-like of (x, y) image coordinates
            frame_from: int, source frame index
            frame_to: int, target frame index

        Returns:
            Nx2 numpy array of displacements (dx, dy) as float32. Points outside
            the shape at frame_from will have np.nan values.
        """
        pixels = np.asarray(pixels, dtype=float)

        # Get rigid transform parameters for both frames
        pos_from, rot_from, _, _ = self.trajectory_at(frame_from)
        pos_to, rot_to, _, _ = self.trajectory_at(frame_to)

        # Update path at source frame to test membership
        self.update_shape(frame_from)
        inside_mask = self.transformed_path.contains_points(pixels)

        flows = np.full((len(pixels), 2), np.nan, dtype=np.float32)
        if not np.any(inside_mask):
            return flows

        # Convert points inside the shape to local (object) coordinates at frame_from
        # local = R(-rot_from) @ (p - pos_from)
        cos_f, sin_f = np.cos(rot_from), np.sin(rot_from)
        R_from_inv = np.array([[cos_f, sin_f], [-sin_f, cos_f]])

        pts_inside = pixels[inside_mask]
        rel = pts_inside - pos_from
        local = rel @ R_from_inv.T

        # Map local coordinates to frame_to: p_to = R(rot_to) @ local + pos_to
        cos_t, sin_t = np.cos(rot_to), np.sin(rot_to)
        R_to = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        pts_to = local @ R_to.T + pos_to

        displacement = pts_to - pts_inside
        flows[inside_mask] = displacement.astype(np.float32)

        return flows

    def compute_optical_flow_every_n(self, N, pair_step=1):
        """
        Compute optical-flow (displacement) between frames separated by N for
        pairs of frames across the full image grid.

        The method computes the rigid mapping for pixels that belong to the
        shape at the source frame. For such pixels the displacement is returned;
        pixels outside the shape at the source frame are omitted.

        Args:
            N: int, number of frames between source and target (target = source + N)
            pair_step: int, step between source frames to evaluate (default 1).

        Returns:
            A list of tuples (frame_from, frame_to, events) where events is a
            structured numpy array with dtype:
                [('x', int16), ('y', int16), ('t_from', int64), ('t_to', int64),
                 ('v_x', float32), ('v_y', float32)]
        """
        img_height, img_width = self.image_size
        yy, xx = np.meshgrid(np.arange(img_height), np.arange(img_width), indexing='ij')
        all_coords = np.vstack([xx.ravel(), yy.ravel()]).T  # (H*W, 2)

        results = []
        last_source = self.total_frames - N
        for frame_from in range(0, last_source + 1, pair_step):
            frame_to = frame_from + N
            if frame_to >= self.total_frames:
                break

            flows = self.compute_optical_flow_between_frames(all_coords, frame_from, frame_to)
            valid = ~np.isnan(flows[:, 0])
            if not np.any(valid):
                continue

            coords = all_coords[valid]
            disp = flows[valid]

            xs = coords[:, 0].astype(np.int16)
            ys = coords[:, 1].astype(np.int16)
            tfs = np.full(len(xs), frame_from, dtype=np.int64)
            tts = np.full(len(xs), frame_to, dtype=np.int64)
            vxs = disp[:, 0].astype(np.float32)
            vys = disp[:, 1].astype(np.float32)

            events = np.zeros(len(xs), dtype=[
                ('x', np.int16), ('y', np.int16), ('t_from', np.int64), ('t_to', np.int64),
                ('v_x', np.float32), ('v_y', np.float32)
            ])
            events['x'] = xs
            events['y'] = ys
            events['t_from'] = tfs
            events['t_to'] = tts
            events['v_x'] = vxs
            events['v_y'] = vys

            results.append((frame_from, frame_to, events))

        return results

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

    def _generate_texture(self, texture_type, params, shape='full'):
        """
        Generate texture pattern as an image.
        
        Args:
            texture_type: 'solid', 'noise', 'gradient', 'checkerboard', 'image', or None
            params: Dictionary with texture parameters
            shape: 'full' for full image size, or tuple (H, W) for specific size
            
        Returns:
            RGB image array (H, W, 3) with values in [0, 1]
        """
        if shape == 'full':
            H, W = self.image_size
        else:
            H, W = shape
            
        if texture_type is None or texture_type == 'solid':
            # Solid color
            color = params.get('color', [1.0, 1.0, 1.0])  # Default white
            if isinstance(color, str):
                # Convert named color to RGB
                from matplotlib.colors import to_rgb
                color = to_rgb(color)
            texture = np.tile(np.array(color).reshape(1, 1, 3), (H, W, 1))
            
        elif texture_type == 'noise':
            # Random noise texture (fix issue #3: seed for consistency)
            noise_type = params.get('noise_type', 'gaussian')  # 'gaussian' or 'uniform'
            scale = params.get('scale', 0.3)  # Noise strength [0, 1]
            base_color = params.get('base_color', [0.5, 0.5, 0.5])
            seed = params.get('seed', None)  # Allow seeding for reproducibility
            
            if isinstance(base_color, str):
                from matplotlib.colors import to_rgb
                base_color = to_rgb(base_color)
            
            # Use local random state for consistency
            rng = np.random.RandomState(seed)
            
            if noise_type == 'gaussian':
                noise = rng.randn(H, W, 3) * scale
            else:  # uniform
                noise = (rng.rand(H, W, 3) - 0.5) * 2 * scale
                
            texture = np.array(base_color).reshape(1, 1, 3) + noise
            texture = np.clip(texture, 0, 1)
            
        elif texture_type == 'gradient':
            # Linear or radial gradient
            gradient_type = params.get('gradient_type', 'linear')  # 'linear' or 'radial'
            color1 = params.get('color1', [0.0, 0.0, 0.0])
            color2 = params.get('color2', [1.0, 1.0, 1.0])
            angle = params.get('angle', 0)  # For linear gradient, angle in degrees
            
            if isinstance(color1, str):
                from matplotlib.colors import to_rgb
                color1 = to_rgb(color1)
            if isinstance(color2, str):
                from matplotlib.colors import to_rgb
                color2 = to_rgb(color2)
                
            if gradient_type == 'linear':
                # Linear gradient
                angle_rad = np.deg2rad(angle)
                y_coords, x_coords = np.mgrid[0:H, 0:W]
                
                # Rotate coordinates
                x_rot = x_coords * np.cos(angle_rad) + y_coords * np.sin(angle_rad)
                
                # Normalize to [0, 1]
                grad = (x_rot - x_rot.min()) / (x_rot.max() - x_rot.min() + 1e-8)
                
            else:  # radial
                # Radial gradient from center
                center_y, center_x = params.get('center', (H/2, W/2))
                y_coords, x_coords = np.mgrid[0:H, 0:W]
                distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                grad = distances / max_dist
                grad = np.clip(grad, 0, 1)
            
            # Interpolate between colors
            color1 = np.array(color1).reshape(1, 1, 3)
            color2 = np.array(color2).reshape(1, 1, 3)
            grad = grad[:, :, np.newaxis]
            texture = color1 * (1 - grad) + color2 * grad
            
        elif texture_type == 'checkerboard':
            # Checkerboard pattern
            square_size = params.get('square_size', 16)
            color1 = params.get('color1', [0.0, 0.0, 0.0])
            color2 = params.get('color2', [1.0, 1.0, 1.0])
            
            if isinstance(color1, str):
                from matplotlib.colors import to_rgb
                color1 = to_rgb(color1)
            if isinstance(color2, str):
                from matplotlib.colors import to_rgb
                color2 = to_rgb(color2)
            
            y_coords, x_coords = np.mgrid[0:H, 0:W]
            checker = ((x_coords // square_size) + (y_coords // square_size)) % 2
            
            color1 = np.array(color1).reshape(1, 1, 3)
            color2 = np.array(color2).reshape(1, 1, 3)
            checker = checker[:, :, np.newaxis]
            texture = color1 * (1 - checker) + color2 * checker
            
        elif texture_type == 'image':
            # Load image from file
            image_path = params.get('image_path')
            if image_path is None:
                print("[WARNING] Image texture requested but no image_path provided, using white")
                texture = np.ones((H, W, 3))
            else:
                try:
                    from PIL import Image
                    # Load image
                    img = Image.open(image_path)
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to match canvas size
                    resize_mode = params.get('resize_mode', 'fill')  # 'fill', 'fit', 'tile'
                    
                    if resize_mode == 'fill':
                        # Stretch to fill entire canvas
                        img = img.resize((W, H), Image.Resampling.LANCZOS)
                        texture = np.array(img) / 255.0
                        
                    elif resize_mode == 'fit':
                        # Scale to fit inside canvas while maintaining aspect ratio
                        img.thumbnail((W, H), Image.Resampling.LANCZOS)
                        img_array = np.array(img) / 255.0
                        
                        # Center the image on the canvas
                        texture = np.ones((H, W, 3)) * params.get('fill_color', [1.0, 1.0, 1.0])
                        img_h, img_w = img_array.shape[:2]
                        y_offset = (H - img_h) // 2
                        x_offset = (W - img_w) // 2
                        texture[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = img_array
                        
                    elif resize_mode == 'tile':
                        # Tile the image to fill canvas
                        img_array = np.array(img) / 255.0
                        img_h, img_w = img_array.shape[:2]
                        
                        texture = np.zeros((H, W, 3))
                        for y in range(0, H, img_h):
                            for x in range(0, W, img_w):
                                y_end = min(y + img_h, H)
                                x_end = min(x + img_w, W)
                                h_crop = y_end - y
                                w_crop = x_end - x
                                texture[y:y_end, x:x_end] = img_array[:h_crop, :w_crop]
                    else:
                        # Default to fill
                        img = img.resize((W, H), Image.Resampling.LANCZOS)
                        texture = np.array(img) / 255.0
                        
                except Exception as e:
                    print(f"[ERROR] Failed to load image from {image_path}: {e}")
                    print("[WARNING] Using white background instead")
                    texture = np.ones((H, W, 3))
        
        else:
            # Unknown texture type, default to white
            texture = np.ones((H, W, 3))
            
        return texture

    def render_frame(self, frame: int, fg_gamma: float = 1.0, bg_gamma: float = 1.0, fg_scale: float = 1.0, bg_scale: float = 1.0) -> np.ndarray:
        """
        Render a grayscale image of the shape at the given frame with optional textures.
        
        Args:
            frame: Frame number
            fg_gamma: Gamma correction for foreground (>1 darkens, <1 brightens)
            bg_gamma: Gamma correction for background (>1 darkens, <1 brightens)
            fg_scale: Multiplier for foreground brightness (0-1, <1 darkens foreground)
            bg_scale: Multiplier for background brightness (0-1, <1 darkens background)
            
        Returns:
            Grayscale image as uint8 numpy array (H, W)
        """
        self.update_shape(frame)
        
        H, W = self.image_size
        
        # If textures are enabled, use pixel-level rendering
        if self.foreground_texture or self.background_texture:
            # Create mask for the shape
            y_coords, x_coords = np.mgrid[0:H, 0:W]
            points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
            
            # Check which points are inside the shape
            mask = self.transformed_path.contains_points(points).reshape(H, W)
            
            # Generate background texture (fixed in world space)
            if self.background_texture:
                # Cache background texture (fix issue #5)
                if self._bg_texture_cache is None or self._bg_texture_cache_size != (H, W):
                    self._bg_texture_cache = self._generate_texture(
                        self.background_texture, 
                        self.background_texture_params
                    )
                    self._bg_texture_cache_size = (H, W)
                bg_texture = self._bg_texture_cache
            else:
                # Default white background
                bg_texture = np.ones((H, W, 3))
            
            # Generate foreground texture (moves with shape)
            if self.foreground_texture:
                from matplotlib.transforms import Affine2D
                from scipy.ndimage import map_coordinates
                
                # Get current position and rotation from trajectory
                position, rotation, _, _ = self.trajectory_at(frame)
                
                # Step 1: Find bounding box of the original shape (in local coordinates)
                original_vertices = self.cached_vertices  # Shape vertices before any transformation
                min_x, min_y = original_vertices.min(axis=0)
                max_x, max_y = original_vertices.max(axis=0)
                shape_width = max_x - min_x
                shape_height = max_y - min_y
                
                # Step 2: Generate or cache texture scaled to shape size
                tex_h = int(np.ceil(shape_height)) + 1
                tex_w = int(np.ceil(shape_width)) + 1
                
                # Cache the base texture sized to the shape bounds
                cache_key = (tex_h, tex_w)
                if self._fg_texture_cache is None or self._fg_texture_cache_size != cache_key:
                    self._fg_texture_cache = self._generate_texture(
                        self.foreground_texture,
                        self.foreground_texture_params,
                        shape=(tex_h, tex_w)
                    )
                    self._fg_texture_cache_size = cache_key
                fg_texture_base = self._fg_texture_cache
                
                # Step 3: Get coordinates of pixels inside the transformed shape
                mask_indices = np.where(mask)
                if len(mask_indices[0]) > 0:
                    y_masked = mask_indices[0]
                    x_masked = mask_indices[1]
                    
                    # Create coordinate pairs for pixels in the shape
                    screen_coords = np.column_stack([x_masked, y_masked])
                    
                    # Step 4: Apply inverse transformation to map screen coords to shape-local coords
                    # The forward transformation is:
                    # 1. Translate to origin using anchor point
                    # 2. Rotate around origin
                    # 3. Translate to final position
                    transform = Affine2D()
                    transform.translate(-self.anchor_point[0], -self.anchor_point[1])
                    transform.rotate(rotation)
                    transform.translate(position[0], position[1])
                    
                    inv_transform = transform.inverted()
                    local_coords = inv_transform.transform(screen_coords)
                    
                    # Step 5: Map local coordinates to texture coordinates
                    # The texture is sized to the shape bounding box
                    # Local coords are relative to anchor point, need to add anchor back to get true local position
                    # Then map to [0, tex_w) x [0, tex_h) relative to bounding box
                    local_coords_true = local_coords + self.anchor_point
                    tex_x = local_coords_true[:, 0] - min_x
                    tex_y = local_coords_true[:, 1] - min_y
                    
                    # Step 6: Use bilinear interpolation to sample texture
                    # map_coordinates expects (row, col) order and uses spline interpolation
                    # We use order=1 for bilinear interpolation
                    coords_for_interp = np.array([tex_y, tex_x])  # (2, N) array for map_coordinates
                    
                    # Interpolate each color channel separately
                    fg_texture = bg_texture.copy()
                    for c in range(3):  # RGB channels
                        # Extract channel and interpolate
                        channel_values = map_coordinates(
                            fg_texture_base[:, :, c],
                            coords_for_interp,
                            order=1,  # Bilinear interpolation
                            mode='nearest',  # Use nearest value for out-of-bounds
                            prefilter=False  # Don't apply prefilter for order=1
                        )
                        fg_texture[y_masked, x_masked, c] = channel_values
                else:
                    fg_texture = bg_texture
            else:
                # Use face_color
                from matplotlib.colors import to_rgb
                if isinstance(self.face_color, str):
                    color = to_rgb(self.face_color)
                else:
                    color = self.face_color
                fg_texture = np.tile(np.array(color).reshape(1, 1, 3), (H, W, 1))
            
            # Combine foreground and background using mask
            mask_3d = mask[:, :, np.newaxis]
            
            # Convert to grayscale FIRST before applying brightness scaling
            fg_gray = (fg_texture[:, :, 0] * 0.299 + fg_texture[:, :, 1] * 0.587 + fg_texture[:, :, 2] * 0.114)
            bg_gray = (bg_texture[:, :, 0] * 0.299 + bg_texture[:, :, 1] * 0.587 + bg_texture[:, :, 2] * 0.114)
            
            # Apply gamma correction separately to foreground and background
            if fg_gamma != 1.0:
                fg_gray = np.clip(fg_gray, 0, 1)
                fg_gray = np.power(fg_gray, fg_gamma)
            
            if bg_gamma != 1.0:
                bg_gray = np.clip(bg_gray, 0, 1)
                bg_gray = np.power(bg_gray, bg_gamma)
            
            # Apply brightness scaling
            fg_gray = fg_gray * fg_scale
            bg_gray = bg_gray * bg_scale
            
            # Combine using mask
            gray = np.where(mask, fg_gray, bg_gray)
            
            # Convert to uint8
            gray = np.clip(gray * 255, 0, 255).astype(np.uint8)
            
        else:
            # Use matplotlib rendering (original method)
            fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.invert_yaxis()  # Match image coordinates
            
            # Draw the shape
            patch = PathPatch(self.transformed_path, facecolor=self.face_color, edgecolor='none')
            ax.add_patch(patch)
            ax.set_facecolor('white')
            
            # Render to array
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            
            # Convert to grayscale numpy array
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            
            # Convert ARGB to grayscale (skip alpha channel)
            gray = np.mean(buf[:, :, 1:], axis=2).astype(np.uint8)
            
            plt.close(fig)
        
        return gray

    def generate_events_v2e(
        self,
        pos_threshold: float = 0.5,
        neg_threshold: float = 0.5,
        sigma_threshold: float = 0.0,
        cutoff_hz: float = 0,
        leak_rate_hz: float = 0,
        shot_noise_rate_hz: float = 0,
        refractory_period_s: float = 0,
        frame_time_us: int = 1000,
        seed: int = 0,
        photoreceptor_noise: bool = False,
        leak_jitter_fraction: float = 0,
        noise_rate_cov_decades: float = 0,
        skip_optical_flow: bool = False,
        fg_gamma: float = 2.0,
        bg_gamma: float = 0.6,
        fg_brightness_scale: float = 1.0,
        bg_brightness_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Generate realistic DVS events using v2e simulator.
        
        Args:
            pos_threshold: Positive threshold (contrast sensitivity)
            neg_threshold: Negative threshold (contrast sensitivity)
            sigma_threshold: Threshold mismatch (variance in thresholds)
            cutoff_hz: Photoreceptor cutoff frequency (Hz)
            leak_rate_hz: Leak event rate (Hz)
            shot_noise_rate_hz: Shot noise rate (Hz)
            refractory_period_s: Refractory period (seconds)
            frame_time_us: Microseconds per frame
            seed: Random seed (0=random, >0=fixed for reproducibility)
            photoreceptor_noise: Use photoreceptor noise model (more realistic)
            leak_jitter_fraction: Leak event timing jitter (fraction of interval)
            noise_rate_cov_decades: Spatial variation in noise rates (decades)
            skip_optical_flow: If True, skip optical flow computation (faster)
            fg_gamma: Gamma correction for foreground (>1 darkens, <1 brightens, default=1.5)
            bg_gamma: Gamma correction for background (>1 darkens, <1 brightens, default=0.8)
            fg_brightness_scale: Foreground brightness multiplier (0-1, lower=darker)
            bg_brightness_scale: Background brightness multiplier (0-1, lower=darker)
            
        Returns:
            np.ndarray with dtype [('x', int16), ('y', int16), ('t', int64), 
                                ('p', bool), ('v_x', float32), ('v_y', float32)]
            If skip_optical_flow=True, v_x and v_y will be zero.
        """
        from v2e_event_generator import V2EEventGenerator, is_v2e_available
        
        if not is_v2e_available():
            raise ImportError(
                "v2e is not available. Install with:\n"
                "  uv pip install -e \".[v2e]\"\n"
                "Or see toy_datasets/INSTALL_V2E.md"
            )
        
        print("Rendering frames for v2e...")
        frames = []
        for frame in tqdm(range(self.total_frames), desc="Rendering frames"):
            gray_frame = self.render_frame(frame, fg_gamma, bg_gamma, fg_brightness_scale, bg_brightness_scale)
            frames.append(gray_frame)
        
        frames = np.array(frames)  # Shape: (T, H, W)
        
        # Generate frame times in microseconds
        frame_times_us = np.arange(self.total_frames) * frame_time_us
        
        # Initialize v2e generator
        v2e_gen = V2EEventGenerator(
            image_size=self.image_size,
            pos_thres=pos_threshold,
            neg_thres=neg_threshold,
            sigma_thres=sigma_threshold,
            cutoff_hz=cutoff_hz,
            leak_rate_hz=leak_rate_hz,
            shot_noise_rate_hz=shot_noise_rate_hz,
            refractory_period_s=refractory_period_s,
            seed=seed,
            photoreceptor_noise=photoreceptor_noise,
            leak_jitter_fraction=leak_jitter_fraction,
            noise_rate_cov_decades=noise_rate_cov_decades,
        )
        
        # Generate events
        print("Generating v2e events...")
        v2e_events = v2e_gen.generate_events_from_frames(frames, frame_times_us)
        
        print(f"Generated {len(v2e_events)} v2e events")
        
        # Add optical flow to events (optional)
        if skip_optical_flow:
            print("Skipping optical flow computation")
            # Just add zero flow fields
            events_with_flow = np.zeros(
                len(v2e_events),
                dtype=[('x', 'i2'), ('y', 'i2'), ('t', 'i8'), ('p', '?'), ('v_x', 'f4'), ('v_y', 'f4')]
            )
            events_with_flow['x'] = v2e_events['x']
            events_with_flow['y'] = v2e_events['y']
            events_with_flow['t'] = v2e_events['t']
            events_with_flow['p'] = v2e_events['p']
            events_with_flow['v_x'] = 0.0
            events_with_flow['v_y'] = 0.0
        else:
            print("Computing optical flow for events...")
            events_with_flow = self.add_optical_flow_to_events(v2e_events, frame_time_us)
        
        return events_with_flow

    def add_optical_flow_to_events(self, events: np.ndarray, frame_time_us: int = 1000) -> np.ndarray:
        """
        Add optical flow information to events.
        
        Args:
            events: Structured array with ('x', 'y', 't', 'p')
            frame_time_us: Microseconds per frame (used to determine frame number from timestamp)
            
        Returns:
            Structured array with ('x', 'y', 't', 'p', 'v_x', 'v_y')
            Note: v_x and v_y are in units of pixels/millisecond for consistency with common velocity units
        """
        if len(events) == 0:
            return np.zeros(0, dtype=[
                ('x', np.int16),
                ('y', np.int16),
                ('t', np.int64),
                ('p', bool),
                ('v_x', np.float32),
                ('v_y', np.float32),
            ])
        
        # Create output array
        events_with_flow = np.zeros(len(events), dtype=[
            ('x', np.int16),
            ('y', np.int16),
            ('t', np.int64),
            ('p', bool),
            ('v_x', np.float32),
            ('v_y', np.float32),
        ])
        
        # Copy existing fields
        events_with_flow['x'] = events['x']
        events_with_flow['y'] = events['y']
        events_with_flow['t'] = events['t']
        events_with_flow['p'] = events['p']
        
        # Compute optical flow for each event
        # Group events by approximate frame for efficiency
        # Fix issue #2: Use known frame_time_us from simulation, not inferred from events
        # V2E uses absolute timestamps based on frame_time_us parameter
        
        # Velocity scale factor: convert from pixels/frame to pixels/millisecond
        # frame_time_us is in microseconds, so divide by 1000 to get milliseconds
        # velocity_ms = pixels/frame * (1 frame / frame_time_ms)
        frame_time_ms = frame_time_us / 1000.0
        velocity_scale = 1.0 / frame_time_ms
        
        for i in tqdm(range(len(events)), desc="Adding optical flow", leave=False):
            event = events[i]
            # Use the known frame timing from v2e simulation
            frame = int(event['t'] / frame_time_us)
            frame = min(frame, self.total_frames - 1)
            
            # Compute flow at this pixel (returns pixels/frame)
            coord = np.array([[event['x'], event['y']]], dtype=float)
            flow_per_frame = self.compute_optical_flow(coord, frame)
            
            # Convert to pixels/millisecond for standard velocity units
            flow_per_ms = flow_per_frame * velocity_scale
            
            events_with_flow['v_x'][i] = flow_per_ms[0, 0]
            events_with_flow['v_y'][i] = flow_per_ms[0, 1]
        
        return events_with_flow

