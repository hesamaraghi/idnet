import sys

sys.path.append(".")
sys.path.append("..")  # to import from parent dir

import argparse
import numpy as np

from IPython.display import HTML
import tonic
import matplotlib.pyplot as plt
import os
import os.path as osp
from scipy.ndimage import gaussian_filter, sobel

from triangle import TriangleMovement
from utils.visualize_utils import animate_events
from utils.data_utils import *
from idn.loader.loader_dsec import HarrisRecursive


def spatiotemporal_image_at(data_array, harris_rec, image_size, portion = 0.25):
    
    (img_height, img_width) = image_size
    
    t_max = data_array['t'][-1].item()
    t_min = data_array['t'][0].item()
    

    print(f'max. time: {t_max}')
    t_obs = (t_max - t_min) * portion + t_min
    print(f'observed time: {t_obs}')
    idx = data_array['t'] < t_obs
    data_truncated = data_array[idx].copy()
    print(f'num of events: {data_truncated.shape[0]}')
    
    harris_rec(data_truncated)
    # Compute the temporal lag
    temporal_lag = np.exp(- (t_obs - harris_rec.last_time_tensor)/harris_rec.tau)

    # update the temporal accumulation tensor

    spatiotemporal_image = harris_rec.temporal_accumulation_tensor * temporal_lag
    # crop the image to the center image_size = (img_height, img_width)
    center_y = spatiotemporal_image.shape[1] // 2
    center_x = spatiotemporal_image.shape[2] // 2
    spatiotemporal_image = spatiotemporal_image[:,center_y - img_height // 2:center_y + img_height // 2,
                          center_x - img_width // 2:center_x + img_width // 2]

    return spatiotemporal_image, data_truncated

def transform_structures(Sxx, Sxy, Syy, V):
    """
    Compute C = V A V^T for each pixel where A = [[Sxx, Sxy],[Sxy, Syy]].
    Inputs:
      Sxx, Sxy, Syy : np.ndarray of shape (H, W)
      V             : np.ndarray of shape (2,2)
    Returns:
      Cxx, Cxy, Cyy : np.ndarray of shape (H, W)
    """
    # validate shapes
    if Sxx.shape != Sxy.shape or Sxx.shape != Syy.shape:
        raise ValueError("Sxx, Sxy, Syy must have the same shape (H,W).")
    if V.shape != (2,2):
        raise ValueError("V must be shape (2,2).")
    
    a, b = V[0,0], V[0,1]
    c, d = V[1,0], V[1,1]

    # compute outputs (vectorized)
    Cxx = (a*a) * Sxx + 2*a*b * Sxy + (b*b) * Syy
    Cxy = (a*c) * Sxx + (a*d + b*c) * Sxy + (b*d) * Syy
    Cyy = (c*c) * Sxx + 2*c*d * Sxy + (d*d) * Syy

    return Cxx, Cxy, Cyy

def find_values(sum_val, prod_val):
    """
    Given the sum and product of two numbers,
    return the two numbers in ascending order.
    """
    # Discriminant check
    discriminant = sum_val**2 - 4 * prod_val
    if np.any(discriminant < 0):
        raise ValueError("No real solutions exist for the given sum and product.")
    
    # Quadratic formula
    x1 = (sum_val + np.sqrt(discriminant)) / 2
    x2 = (sum_val - np.sqrt(discriminant)) / 2
    
    return np.maximum(x1, x2), np.minimum(x1, x2)

def plot_structures(Cxx, Cxy, Cyy, title_prefix=None):
    fig, axes = plt.subplots(3, 1, figsize=(6,20))

    images = [Cxx, Cxy, Cyy]
    if title_prefix is None:
        titles = ["Cxx", "Cxy", "Cyy"]
    else:
        titles = [f"{title_prefix}xx", f"{title_prefix}xy", f"{title_prefix}yy"]

    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, origin='upper', cmap='viridis')  # 'origin=upper' inverts y-axis
        ax.set_title(title)
        ax.axis('off')
        ax.invert_yaxis()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # add colorbar per image

    plt.tight_layout()


def multiply_image_matrix(image, M):
    """
    Multiply each pixel [I_x, I_y] by matrix M.
    Inputs:
      image : np.ndarray of shape (H, W, 2)
      M     : np.ndarray of shape (2, 2)
    Returns:
      transformed_image : np.ndarray of shape (H, W, 2)
    """
    if image.shape[2] != 2:
        raise ValueError("Image must have shape (H, W, 2).")
    if M.shape != (2, 2):
        raise ValueError("Matrix M must have shape (2, 2).")
    
    # Reshape image to (H*W, 2) for matrix multiplication
    H, W, _ = image.shape
    reshaped_image = image.reshape(-1, 2)
    
    # Perform matrix multiplication
    transformed_reshaped = reshaped_image @ M.T
    
    # Reshape back to original image shape
    transformed_image = transformed_reshaped.reshape(H, W, 2)
    
    return transformed_image

def flow_to_color(flow, max_flow=None):
    """
    Convert flow to RGB image.
    flow: [H,W,2] numpy array
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    if max_flow is None:
        max_flow = np.max(mag)
        print(f"max flow: {max_flow}")
        if max_flow == 0:
            max_flow = 1e-6

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,0] = ang / 2                  # Hue (0-180 in OpenCV)
    hsv[...,1] = 255                      # Saturation
    hsv[...,2] = np.clip((mag / max_flow) * 255, 0, 255)  # Value (brightness)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb, ang, mag

def plot_flow_with_angles(flow):
    """
    Plot flow as color-coded image with angle values (in degrees) overlayed.
    """
    rgb, ang, mag = flow_to_color(flow)

    plt.figure(figsize=(4,4))
    plt.imshow(rgb)
    plt.axis('off')

    h, w = flow.shape[:2]
    for i in range(h):
        for j in range(w):
            # get angle text
            angle = ang[i, j]
            text_color = 'white' if mag[i,j] > 0.3 * mag.max() else 'black'
            plt.text(j, i, f"{angle:.0f}°", ha='center', va='center', 
                     fontsize=8, color=text_color, weight='bold')

    plt.title("Optical Flow Directions (°)")
    plt.show()

def make_color_wheel(size=200):
    """
    Create a flow color wheel legend (direction -> hue, magnitude -> radius).
    """
    # coordinate grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(y,-x)  # flip y for display

    flow = np.stack((xx, yy), axis=-1)  # [H,W,2]
    return flow_to_color(flow)

def main(args):
    img_height = args.im_size[0]
    img_width = args.im_size[1]
    image_size = (img_height, img_width)

    center = np.array([img_width // 2, img_height // 2])
    num_frames = args.total_frames
    alpha = np.deg2rad(args.alpha)
    triangle_base = args.triangle_base
    triangle_height = args.triangle_height if args.triangle_height is not None else triangle_base / (2 * np.tan(alpha / 2))
    added_height = args.added_height
    start_pos = np.array(args.start_pos) if args.start_pos is not None else np.array([center[0], 0])
    movement_direction = args.movement_direction
    face_color = args.face_color
    speed = args.speed
    tau = args.tau
    filter_size = args.filter_size
    portion = args.portion
    gaussian_sigma = args.gaussian_sigma

    output_dir = osp.join(
        osp.dirname(osp.abspath(__file__)),
        args.output_dir,
    )

    harris_rec = HarrisRecursive(tau, filter_size, image_size)
    print(f"direction of movement (degrees): {movement_direction}")
    
    # Create the triangle movement object
    triangle = TriangleMovement(
        total_frames=num_frames,
        image_size=image_size,
        triangle_base=triangle_base,
        triangle_height=triangle_height,
        added_height=added_height,
        start_pos=start_pos,
        movement_direction=movement_direction,
        face_color=face_color,
        speed=speed
    )

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the image sequence
    if args.save_gif:
        anim = triangle.create_animation(frame_step=args.frame_step)
        anim.save(osp.join(output_dir, "triangle_movement.gif"), fps=args.gif_fps)
        print(f"Saved animation to {osp.join(output_dir, 'triangle_movement.gif')}")

    # Generate event data
    data_array = triangle.generate_events()

    if args.save_gif:
        transform = tonic.transforms.ToFrame(
            sensor_size=(triangle.image_size[1], triangle.image_size[0], 2),
            time_window=10,
            # event_count=500,
            overlap=0.5,
        )
        anim = animate_events(data_array, transform, fig_size=(5,5), invert_yaxis=True)
        anim.save(osp.join(output_dir, "triangle_events.gif"), fps=args.gif_fps)
        print(f"Saved event animation to {osp.join(output_dir, 'triangle_events.gif')}")

    triangle.speed = speed
    data_array = triangle.change_event_speeds()

    spatiotemporal_image, data_truncated = spatiotemporal_image_at(data_array, harris_rec, image_size, portion)
    frame = int(triangle.total_frames * portion)
    points = triangle.trajectory_at(frame)[0].reshape(1,2)
    flows = triangle.compute_optical_flow(points,frame)
    flows_norm = flows / np.linalg.norm(flows, axis=1, keepdims=True)
    points = np.round(points).astype(int).squeeze()
    v_left = np.array([-triangle_height,triangle_base / 2])
    v_left /= np.linalg.norm(v_left)
    v_right = np.array([triangle_height,triangle_base / 2])
    v_right /= np.linalg.norm(v_right)
    V = np.stack([v_left,v_right], axis=1)
    if args.save_gif:
        assert portion < 0.5, "portion should be less than 0.5 to save the spatiotemporal image gif"
        print(f'frame: {frame}')
        print(f'points: {points}')
        print(f'flows: {flows}')
        
        # Determine common vmin and vmax
        vmin = min(data.min() for data in spatiotemporal_image)
        vmax = max(data.max() for data in spatiotemporal_image)
        plt.figure(figsize=(5, 5))
        plt.imshow(spatiotemporal_image[0], vmin=vmin, vmax=vmax, cmap='viridis')
        plt.gca().invert_yaxis()    
            
        plt.quiver(points[0], points[1],
            40*flows_norm[:,0], 40*flows_norm[:,1],
            angles='xy', scale_units='xy', scale=1, color='red')
        plt.quiver(points[0], points[1],
            40*v_left[0], 40*v_left[1],
            angles='xy', scale_units='xy', scale=1, color='blue')
        plt.quiver(points[0], points[1],
            40*v_right[0], 40*v_right[1],
            angles='xy', scale_units='xy', scale=1, color='blue')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"frame {frame} @ speed = {speed} frame/mS", fontsize=12)
        plt.savefig(osp.join(output_dir, f"spatiotemporal_image_frame{frame}.png"))
        plt.close()
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a toy dataset based on a triangular movement.")
    
    # Shape and movement parameters
    parser.add_argument("--im_size", type=int, nargs=2, default=[200, 200], help="Image size (height width)")
    parser.add_argument("--total_frames", type=int, default=2_000, help="Number of frames in the sequence")
    parser.add_argument("--alpha", type=float, default=45.0, help="Angle of the triangle in degrees")
    parser.add_argument("--triangle_base", type=float, default=400.0, help="Base length of the triangle")
    parser.add_argument("--triangle_height", type=float, default=None, help="Height of the triangle (if None, computed from base and alpha)")
    parser.add_argument("--start_pos", type=float, nargs=2, default=None, help="Starting position (x y) of the triangle's centroid. If None, starts at bottom center of image")
    parser.add_argument("--movement_direction", type=float, default=0.0, help="Angle of movement direction and y-axis in degrees")
    parser.add_argument("--added_height", type=float, default=200.0, help="Additional height to add to the triangle")
    parser.add_argument("--face_color", type=str, default="black", help="Color of the shape")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of movement")
    parser.add_argument("--portion", type=float, default=0.25, help="Portion of the event data to use for spatiotemporal image")
    
    # Augmented input (eig + filter) parameters
    parser.add_argument("--tau", type=float, default=30.0, help="Time constant for spatiotemporal filter")
    parser.add_argument("--filter_size", type=int, default=7, help="Size of the spatiotemporal filter")
    parser.add_argument("--gaussian_sigma", type=float, default=1.0, help="Sigma for Gaussian smoothing")

    # results parameters
    parser.add_argument("--output_dir", type=str, default="theory_triangle", help="Directory to save results")
    parser. add_argument("--save_gif", action="store_true", help="Whether to save the output as a GIF")
    parser.add_argument("--gif_fps", type=int, default=30, help="FPS for the output GIF")
    parser.add_argument("--frame_step", type=int, default=20, help="Step between frames to save in the GIF")
    
    args = parser.parse_args()
    
    main(args)
