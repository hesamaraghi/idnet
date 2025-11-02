#!/usr/bin/env bash

# ----------------------------------------------------------------------
# Default arguments (edit these as you wish)
# ----------------------------------------------------------------------
DEFAULT_IM_SIZE="200 200"
DEFAULT_TOTAL_FRAMES=2000
DEFAULT_ALPHA=70.0
DEFAULT_TRIANGLE_BASE=400.0
DEFAULT_TRIANGLE_HEIGHT="None"
DEFAULT_START_POS="None"
DEFAULT_MOVEMENT_DIRECTION=-25.0
DEFAULT_ADDED_HEIGHT=200.0
DEFAULT_FACE_COLOR="black"
DEFAULT_SPEED=10.0
DEFAULT_PORTION=0.25
DEFAULT_TAU=30.0
DEFAULT_FILTER_SIZE=7
DEFAULT_GAUSSIAN_SIGMA=1.0
DEFAULT_OUTPUT_DIR="theory_triangle"
DEFAULT_GIF_FPS=30
DEFAULT_FRAME_STEP=20

# ----------------------------------------------------------------------
# Build argument list
# ----------------------------------------------------------------------
python theory_triangle.py \
  --im_size ${DEFAULT_IM_SIZE} \
  --total_frames ${DEFAULT_TOTAL_FRAMES} \
  --alpha ${DEFAULT_ALPHA} \
  --triangle_base ${DEFAULT_TRIANGLE_BASE} \
  --movement_direction ${DEFAULT_MOVEMENT_DIRECTION} \
  --added_height ${DEFAULT_ADDED_HEIGHT} \
  --face_color ${DEFAULT_FACE_COLOR} \
  --speed ${DEFAULT_SPEED} \
  --portion ${DEFAULT_PORTION} \
  --tau ${DEFAULT_TAU} \
  --filter_size ${DEFAULT_FILTER_SIZE} \
  --gaussian_sigma ${DEFAULT_GAUSSIAN_SIGMA} \
  --output_dir ${DEFAULT_OUTPUT_DIR} \
  --gif_fps ${DEFAULT_GIF_FPS} \
  --frame_step ${DEFAULT_FRAME_STEP} \
  --save_gif