#!/usr/bin/env python3
"""
Wrapper script to generate multiple theoretical shapes with different configurations.
This keeps track of all created shapes and their parameters.

Usage:
    python toy_datasets/generate_theory_shapes.py --config <config_name>
    python toy_datasets/generate_theory_shapes.py --list
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Predefined shape configurations
SHAPE_CONFIGS = {
    "default_triangle": {
        "im_size": [200, 200],
        "total_frames": 2000,
        "alpha": 75.0,
        "triangle_base": 400.0,
        "triangle_height": None,
        "added_height": 200.0,
        "start_pos": None,
        "movement_direction": 0.0,
        "face_color": "black",
        "speed": 4.0,
        "tau": 30.0,
        "filter_size": 7,
        "gaussian_sigma": 1.0,
        "portion": 0.25,
        "output_dir": "theory_triangle/default",
        "save_gif": True,
        "gif_fps": 15,
        "frame_step": 20,
        "invert_polarity": True,
        "viz_start_point": None,
        "viz_colormap": "bwr",
        "viz_show_colorbar": True,
        "viz_show_flow_arrow": False,
        "viz_show_normal_arrows": False,
    },
    
    "minus30": {
        "im_size": [200, 200],
        "total_frames": 2000,
        "alpha": 75.0,
        "triangle_base": 400.0,
        "triangle_height": None,
        "added_height": 200.0,
        "start_pos": [157.7,0.0],
        "movement_direction": -30.0,
        "face_color": "black",
        "speed": 4.0,
        "tau": 30.0,
        "filter_size": 7,
        "gaussian_sigma": 1.0,
        "portion": 0.25,
        "output_dir": "theory_triangle/minus30",
        "save_gif": True,
        "gif_fps": 15,
        "frame_step": 20,
        "invert_polarity": True,
        "viz_start_point": None,
        "viz_colormap": "bwr",  # Blue-white-red
        "viz_show_colorbar": False,
        "viz_show_flow_arrow": True,
        "viz_show_normal_arrows": True,
    },
    
    "minus11": {
        "im_size": [200, 200],
        "total_frames": 2000,
        "alpha": 75.0,
        "triangle_base": 400.0,
        "triangle_height": None,
        "added_height": 200.0,
        "start_pos": [119.4,0.0],
        "movement_direction": -11.0,
        "face_color": "black",
        "speed": 4.0,
        "tau": 30.0,
        "filter_size": 7,
        "gaussian_sigma": 1.0,
        "portion": 0.25,
        "output_dir": "theory_triangle/minus11",
        "save_gif": True,
        "gif_fps": 15,
        "frame_step": 20,
        "invert_polarity": True,
        "viz_start_point": None,
        "viz_colormap": "bwr",  # Blue-white-red
        "viz_show_colorbar": False,
        "viz_show_flow_arrow": True,
        "viz_show_normal_arrows": True,  # No normal arrows
    },
    
    "pos5": {
        "im_size": [200, 200],
        "total_frames": 2000,
        "alpha": 75.0,
        "triangle_base": 400.0,
        "triangle_height": None,
        "added_height": 200.0,
        "start_pos": [91.25,0.0],
        "movement_direction": 5.0,
        "face_color": "black",
        "speed": 4.0,
        "tau": 30.0,
        "filter_size": 7,
        "gaussian_sigma": 1.0,
        "portion": 0.25,
        "output_dir": "theory_triangle/pos5",
        "save_gif": True,
        "gif_fps": 15,
        "frame_step": 20,
        "invert_polarity": False,  # Don't invert polarity
        "viz_start_point": None,
        "viz_colormap": "bwr", # Blue-white-red
        "viz_show_colorbar": False,  # No colorbar
        "viz_show_flow_arrow": True,  # No flow arrow
        "viz_show_normal_arrows": True,
    },
    
    "pos20": {
        "im_size": [200, 200],
        "total_frames": 2000,
        "alpha": 75.0,
        "triangle_base": 400.0,
        "triangle_height": None,
        "added_height": 200.0,
        "start_pos": [63.6,0.0],
        "movement_direction": 20.0,
        "face_color": "black",
        "speed": 4.0,
        "tau": 30.0,
        "filter_size": 7,
        "gaussian_sigma": 1.0,
        "portion": 0.25,
        "output_dir": "theory_triangle/pos20",
        "save_gif": True,
        "gif_fps": 15,
        "frame_step": 20,
        "invert_polarity": True,
        "viz_start_point": None,
        "viz_colormap": "bwr",
        "viz_show_colorbar": False,
        "viz_show_flow_arrow": True,
        "viz_show_normal_arrows": True,
    },
    
    "small_filter": {
        "im_size": [200, 200],
        "total_frames": 2000,
        "alpha": 45.0,
        "triangle_base": 400.0,
        "triangle_height": None,
        "added_height": 200.0,
        "start_pos": None,
        "movement_direction": 0.0,
        "face_color": "black",
        "speed": 1.0,
        "tau": 30.0,
        "filter_size": 3,
        "gaussian_sigma": 1.0,
        "portion": 0.25,
        "output_dir": "theory_triangle/small_filter",
        "save_gif": True,
        "gif_fps": 30,
        "frame_step": 20,
        "invert_polarity": True,
        "viz_start_point": None,
        "viz_colormap": "viridis",
        "viz_show_colorbar": True,
        "viz_show_flow_arrow": True,
        "viz_show_normal_arrows": True,
    },
    
    "large_filter": {
        "im_size": [200, 200],
        "total_frames": 2000,
        "alpha": 45.0,
        "triangle_base": 400.0,
        "triangle_height": None,
        "added_height": 200.0,
        "start_pos": None,
        "movement_direction": 0.0,
        "face_color": "black",
        "speed": 1.0,
        "tau": 30.0,
        "filter_size": 11,
        "gaussian_sigma": 1.0,
        "portion": 0.25,
        "output_dir": "theory_triangle/large_filter",
        "save_gif": True,
        "gif_fps": 30,
        "frame_step": 20,
        "invert_polarity": True,
        "viz_start_point": None,
        "viz_colormap": "viridis",
        "viz_show_colorbar": True,
        "viz_show_flow_arrow": True,
        "viz_show_normal_arrows": True,
    },
}


def build_command(config_name, config):
    """Build the command to run theory_triangle.py with the given config."""
    script_path = Path(__file__).parent / "theory_triangle.py"
    cmd = [sys.executable, str(script_path)]
    
    # Add all parameters from config
    for key, value in config.items():
        if key == "save_gif":
            if value:
                cmd.append("--save_gif")
        elif key == "viz_show_colorbar":
            if value:
                cmd.append("--viz_show_colorbar")
        elif key == "viz_show_flow_arrow":
            if value:
                cmd.append("--viz_show_flow_arrow")
        elif key == "viz_show_normal_arrows":
            if value:
                cmd.append("--viz_show_normal_arrows")
        elif key == "invert_polarity":
            # Default is True (invert), so only add flag if True, or --no_invert_polarity if False
            if value:
                cmd.append("--invert_polarity")
            else:
                cmd.append("--no_invert_polarity")
        elif isinstance(value, list):
            cmd.append(f"--{key}")
            cmd.extend([str(v) for v in value])
        elif value is not None:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    return cmd


def list_configs():
    """List all available shape configurations."""
    print("\n" + "="*60)
    print("Available Shape Configurations:")
    print("="*60 + "\n")
    
    for name, config in SHAPE_CONFIGS.items():
        print(f"📐 {name}:")
        print(f"   Output: {config['output_dir']}")
        print(f"   Speed: {config['speed']}")
        print(f"   Alpha: {config['alpha']}°")
        print(f"   Movement: {config['movement_direction']}°")
        print(f"   Filter size: {config['filter_size']}")
        print(f"   Invert polarity: {config.get('invert_polarity', True)}")
        print(f"   Colormap: {config.get('viz_colormap', 'viridis')}")
        print(f"   Show colorbar: {config.get('viz_show_colorbar', True)}")
        print(f"   Show flow arrow: {config.get('viz_show_flow_arrow', True)}")
        print(f"   Show normal arrows: {config.get('viz_show_normal_arrows', True)}")
        if config.get('viz_start_point'):
            print(f"   Viz start point: {config['viz_start_point']}")
        print()


def list_generated_shapes():
    """List all previously generated shapes by reading metadata files."""
    base_dir = Path(__file__).parent / "theory_triangle"
    
    if not base_dir.exists():
        print("\nNo shapes have been generated yet.")
        return
    
    print("\n" + "="*60)
    print("Previously Generated Shapes:")
    print("="*60 + "\n")
    
    metadata_files = list(base_dir.glob("**/metadata.json"))
    
    if not metadata_files:
        print("No shapes found with metadata.")
        return
    
    for metadata_path in sorted(metadata_files):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            rel_path = metadata_path.parent.relative_to(base_dir.parent)
            print(f"📁 {rel_path}")
            print(f"   Config: {metadata.get('config_name', 'custom')}")
            print(f"   Created: {metadata.get('timestamp', 'unknown')}")
            
            args = metadata.get('args', {})
            print(f"   Speed: {args.get('speed', 'N/A')}")
            print(f"   Alpha: {args.get('alpha', 'N/A')}°")
            print(f"   Movement: {args.get('movement_direction', 'N/A')}°")
            print(f"   Filter size: {args.get('filter_size', 'N/A')}")
            print(f"   Invert polarity: {args.get('invert_polarity', 'N/A')}")
            print(f"   Colormap: {args.get('viz_colormap', 'N/A')}")
            print(f"   Show colorbar: {args.get('viz_show_colorbar', 'N/A')}")
            print(f"   Show flow arrow: {args.get('viz_show_flow_arrow', 'N/A')}")
            print(f"   Show normal arrows: {args.get('viz_show_normal_arrows', 'N/A')}")
            if args.get('viz_start_point'):
                print(f"   Viz start point: {args['viz_start_point']}")
            print()
        except Exception as e:
            print(f"   ⚠️  Error reading metadata: {e}\n")


def generate_shape(config_name, custom_overrides=None):
    """Generate a shape with the given configuration."""
    if config_name not in SHAPE_CONFIGS:
        print(f"❌ Error: Configuration '{config_name}' not found.")
        print(f"Available configs: {', '.join(SHAPE_CONFIGS.keys())}")
        return False
    
    config = SHAPE_CONFIGS[config_name].copy()
    
    # Apply custom overrides if provided
    if custom_overrides:
        config.update(custom_overrides)
    
    print(f"\n{'='*60}")
    print(f"Generating shape: {config_name}")
    print(f"{'='*60}\n")
    
    # Build and run command
    cmd = build_command(config_name, config)
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Successfully generated shape: {config_name}")
        print(f"   Output directory: {config['output_dir']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error generating shape: {e}")
        return False


def generate_multiple(config_names):
    """Generate multiple shapes in sequence."""
    print(f"\n{'='*60}")
    print(f"Generating {len(config_names)} shapes")
    print(f"{'='*60}\n")
    
    results = {}
    for config_name in config_names:
        success = generate_shape(config_name)
        results[config_name] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("Generation Summary:")
    print(f"{'='*60}\n")
    
    for config_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {config_name}")
    
    total = len(results)
    successful = sum(results.values())
    print(f"\nTotal: {successful}/{total} successful")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple theoretical shapes with different configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available configurations
  python generate_theory_shapes.py --list
  
  # List previously generated shapes
  python generate_theory_shapes.py --list-generated
  
  # Generate a single shape
  python generate_theory_shapes.py --config default_triangle
  
  # Generate multiple shapes
  python generate_theory_shapes.py --config default_triangle fast_triangle wide_triangle
  
  # Generate all available shapes
  python generate_theory_shapes.py --all
        """
    )
    
    parser.add_argument(
        "--config", 
        nargs="+", 
        help="Name(s) of the shape configuration(s) to generate"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Generate all available configurations"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List all available shape configurations"
    )
    parser.add_argument(
        "--list-generated", 
        action="store_true", 
        help="List all previously generated shapes"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_configs()
        return
    
    if args.list_generated:
        list_generated_shapes()
        return
    
    if args.all:
        config_names = list(SHAPE_CONFIGS.keys())
        generate_multiple(config_names)
        return
    
    if args.config:
        if len(args.config) == 1:
            generate_shape(args.config[0])
        else:
            generate_multiple(args.config)
        return
    
    # No arguments provided
    parser.print_help()


if __name__ == "__main__":
    main()
