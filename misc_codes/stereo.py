#!/usr/bin/env python3


# I abuse <username>/<username> config repo for hosting miscellanous codes.

# Stereo Image from one single image and a bunch of camera parameters
# prints the stuff, should be thrown into "Writing Spatial Photos" example
# on Apple

# Generates all the parameters using a bunch of formulas I found in spare
# time

# Untested as I do not actually have an Apple Vision Pro. Looked up a bunch
# of formulas, ChatGPT did the inpainting, forward_warp_with_disparity
# converted and randomly adjusted from questions on Stack Overflow, but the
# algorithm should be without copyright

import cv2
import torch
import numpy as np
import argparse
import sys
from depth_anything_v2.dpt import DepthAnythingV2  # Adjust import path as needed
import math

DEVICE = 'mps'  # Change if needed

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
model = DepthAnythingV2(**model_configs['vitl'])
model.load_state_dict(torch.load("depth_anything_v2_metric_hypersim_vitl.pth", map_location='cpu'))
model = model.to(DEVICE).eval()

def forward_warp_with_disparity(left_rgb: np.ndarray, disparity: np.ndarray) -> np.ndarray:
    H, W = disparity.shape
    right_img = np.zeros_like(left_rgb)
    z_buf     = np.full((H, W), np.inf, dtype=np.float32)

    for y in range(H):
        for x in range(W):
            d = disparity[y, x]
            if not np.isfinite(d) or d <= 0:
                continue
            x_f = x - d
            x_i = int(round(x_f))
            if 0 <= x_i < W:
                z = 1.0 / (d + 1e-6)
                if z < z_buf[y, x_i]:
                    z_buf[y, x_i] = z
                    right_img[y, x_i, :] = left_rgb[y, x, :]
    return right_img

def inpaint_holes(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray == 0).astype(np.uint8) * 255
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, help="Left-view image path")
    parser.add_argument("--output", required=True, help="Output path for synthesized right image")
    parser.add_argument("--f_mm", type=float, required=True, help="Camera focal length in millimeters")
    parser.add_argument("--sensor_width_mm", type=float, required=True, help="Camera sensor width in millimeters")
    parser.add_argument("--baseline_m", type=float, required=True, help="Stereo baseline in meters")
    parser.add_argument("--disparityAdjustment", type=float, default=0.0,
                        help="Disparity adjustment as fraction of image width (positive moves objects closer on Vision Pro). Printed only.")
    return parser.parse_args()

def compute_focal_length_px(f_mm: float, sensor_width_mm: float, image_width_px: int) -> float:
    return f_mm * (image_width_px / sensor_width_mm)

def compute_horizontal_fov(f_mm: float, sensor_width_mm: float) -> float:
    # FOV in degrees: 2 * atan(sensor_width / (2 * focal_length))
    fov_rad = 2 * math.atan(sensor_width_mm / (2 * f_mm))
    return math.degrees(fov_rad)

def main():
    args = parse_args()

    left_bgr = cv2.imread(args.left)
    if left_bgr is None:
        print(f"Error loading image: {args.left}", file=sys.stderr)
        sys.exit(1)
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

    image_height, image_width = left_rgb.shape[:2]

    focal_length_px = compute_focal_length_px(args.f_mm, args.sensor_width_mm, image_width)
    fov_deg = compute_horizontal_fov(args.f_mm, args.sensor_width_mm)

    depth_meters = model.infer_image(left_rgb)  # Metric depth in meters
    depth_clipped = np.clip(depth_meters, 1e-3, None)

    disparity = (focal_length_px * args.baseline_m / depth_clipped).astype(np.float32)

    # Note: disparityAdjustment is printed but NOT applied to disparity
    print(f"Disparity adjustment parameter received: {args.disparityAdjustment} (no effect applied)")

    right_warped = forward_warp_with_disparity(left_rgb, disparity)
    right_filled = inpaint_holes(right_warped)

    right_bgr = cv2.cvtColor(right_filled, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, right_bgr)

    print(f"Synthesized right image saved to {args.output}")
    print(f"Horizontal FOV (degrees): {fov_deg:.2f}")
    print(f"Disparity Adjustment (fraction of image width): {args.disparityAdjustment} (positive moves objects closer to the screen on Vision Pro)")
    print(f"Stereo baseline (meters): {args.baseline_m:.3f}")

if __name__ == "__main__":
    main()

