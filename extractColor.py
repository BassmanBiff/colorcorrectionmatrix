#!/usr/bin/env python3

# TODO: Use open ("incomplete") contours when reconstructing missing chips
# TODO: Don't require size estimate from user
# TODO: Try template matching to find grid?

import argparse
import csv
import cv2                              # Find chips
import numpy as np
import rawpy
import sys


def to_uint8(img):
    return (img >> 8).astype(np.uint8)


def check_length(array):
    length = len(array)
    if length < 12 or length > 24:
        print("Error: Found {} chips, can't construct color grid"
              ".".format(length))
        sys.exit(2)


# Parse input
parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str)
parser.add_argument('output_csv', type=argparse.FileType('w'),
                    default='colorchart.csv')
parser.add_argument('-g', '--gamma', type=float, default=2.2)
parser.add_argument(
    '-x', type=int, default=35,
    help='expected width of color chips, in pixels')
parser.add_argument(
    '-y', type=int, default=25,     # Should be -h and -w, but -h is taken :(
    help='expected height of color chips, in pixels')
args = parser.parse_args()

# Load image
if args.input_image[-4:] == '.png':     # png
    img = cv2.imread(args.input_image)
elif args.input_image[-4:] == '.dng':   # dng with sane defaults
    raw = rawpy.imread(args.input_image)
    img = raw.postprocess(
        demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
        half_size=False,
        four_color_rgb=False,
        fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
        noise_thr=None,
        median_filter_passes=0,
        use_camera_wb=False,
        use_auto_wb=False,
        user_wb=None,
        output_color=rawpy.ColorSpace.raw,
        output_bps=16,
        user_flip=None,
        user_black=None,
        user_sat=None,
        no_auto_bright=True,
        auto_bright_thr=None,
        adjust_maximum_thr=0.75,
        bright=1.0,
        highlight_mode=rawpy.HighlightMode.Clip,
        exp_shift=None,
        exp_preserve_highlights=0.0,
        no_auto_scale=True,
        gamma=(1, 1),     # (2.222, 4.5) or (1, 1)
        chromatic_aberration=None,
        bad_pixels_path=None)

# Calculate rescaling factor for display only
f_scale = min(1, 1024 / np.max(img.shape))

# Handle 16-bit format (keep precision from original image)
if img.dtype == np.uint16:
    img = img << 6  # 10-bit to 16-bit
    img_display = to_uint8(cv2.resize(img, (0, 0), fx=f_scale, fy=f_scale))
    img_edges = to_uint8(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
else:
    img_display = cv2.resize(img, (0, 0), fx=f_scale, fy=f_scale)
    img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# cv2.imshow('Input', img_display)
# cv2.waitKey(0)

# Find gradients
median, sigma = np.median(img_edges), 0.33      # Tune sigma if needed
img_edges = cv2.Canny(
    img_edges,
    int(max(0, (1.0 - sigma) * median)),
    int(min(255, (1.0 + sigma) * median)))
img_edges, contours, hierarchy = cv2.findContours(
    img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.imshow('Edges', cv2.resize(img_edges, (0, 0), fx=f_scale, fy=f_scale))
# cv2.waitKey(0)

# Find color chips
color_chips = []    # Format: [top-left x, top-left y, width, height]
pad = int((args.x + args.y) / 12)
pad2 = pad * 2
for i in range(len(contours)):
    # If contour is closed with no others inside:
    if hierarchy[0][i][3] > -1:
        # Fit bounding rectangle
        r = cv2.boundingRect(contours[i])
        x, y, = r[2:]
        # If rectangle is near expected size, add to color_chips
        if x > args.x * 0.8 and x < args.x * 1.2 and \
           y > args.y * 0.8 and y < args.y * 1.2:
            color_chips.append(
                [r[0] + pad, r[1] + pad, r[2] - pad2, r[3] - pad2])
check_length(color_chips)

# Remove false positives
color_array = np.array(color_chips)
h = int(np.median(color_array[:, 3]))
w = int(np.median(color_array[:, 2]))
for i, chip in enumerate(color_chips):
    x, y, = chip[2:]
    if x < w * 0.9 or x > w * 1.1 or \
       y < h * 0.9 or y > h * 1.1:
        color_chips[i] = 0
color_chips = [chip for chip in color_chips if chip != 0]
check_length(color_chips)
print("Color chips found:\t\t", len(color_chips))

# Find leftmost and top chips to define first column and row
left_i = top_i = 0
left_x = top_y = 9E9    # Arbitrary large number
for i in range(len(color_chips)):
    x, y = color_chips[i][:2]
    if x < left_x:
        left_i, left_x = i, x
    if y < top_y:
        top_i, top_y = i, y

# Assemble color grid
color_grid = np.full((4, 6), 255, np.uint8)   # Known shape


def add_chip(chip_i, grid_i, grid_j):
    ''' Add a color chip to both the color grid data and display image '''
    if color_grid[grid_j, grid_i] != 255:
        print("Error: Attempted to reassign color_grid[{}, {}] from {} to {}"
              ".".format(grid_j, grid_i, color_grid[grid_j, grid_i], chip_i))
        sys.exit(2)
    color_grid[grid_j, grid_i] = chip_i                     # add to grid
    # print(chip_i, grid_i, grid_j)
    chip = [int(x * f_scale) for x in color_chips[chip_i]]  # scale for display
    cv2.rectangle(                                          # add to display
        img_display,
        (chip[0], chip[1]),
        (chip[0] + chip[2], chip[1] + chip[3]),
        (0, 0, 255),
        1)


# Assign column and row to detected chips
for i in range(len(color_chips)):
    x, y, = color_chips[i][:2]
    # Assign column based x distance from leftmost detected chip
    # 0.8 and 1.2 factors compensate for blank space between adjacent chips
    dist = x - left_x
    for col in range(6):
        # If
        if dist > (w + pad2) * (-0.5 + col * 0.8) \
           and dist <= (w + pad2) * (0.5 + col * 1.2):
            chip_col = col
            break
    else:
        print("Error: Unable to assign column to chip {}, dist={}"
              ".".format(i, dist))
        sys.exit(2)
    # Assign row based on y distance from topmost detected chip
    dist = y - top_y
    for row in range(4):
        if dist > (h + pad2) * (-0.5 + row * 0.8) \
           and dist <= (h + pad2) * (0.5 + row * 1.2):
            chip_row = row
            break
    else:
        print("Error: Unable to assign row to chip {}, dist={}"
              ".".format(i, dist))
        sys.exit(2)
    add_chip(i, chip_col, chip_row)

# Find average column/row spacing
x_spacing = x_n = y_spacing = y_n = 0   # theta = t_n = 0
for i in range(6):
    for j in range(4):
        chip_index = color_grid[j, i]
        if chip_index < 255:
            chip = color_chips[chip_index]
            x, y, = chip[:2]
            if i < 5:
                next_index = color_grid[j, i + 1]
                if next_index < 255:
                    next_chip = color_chips[next_index]
                    next_x, next_y, = next_chip[:2]
                    x_spacing += next_x - x
                    # theta += np.arctan((next_y - y) / (next_x - x))
                    x_n += 1
                    # t_n += 1
            if j < 3:
                next_index = color_grid[j + 1, i]
                if next_index < 255:
                    next_chip = color_chips[next_index]
                    next_x, next_y, = next_chip[:2]
                    y_spacing += next_chip[1] - chip[1]
                    # theta += np.arctan((next_y - y) / (next_x - x))
                    y_n += 1
                    # t_n += 1
x_spacing = int(x_spacing / x_n)
y_spacing = int(y_spacing / y_n)
# theta /= t_n

# Fill in chips that weren't detected
h_pad, w_pad, n = int(h * 0.3), int(w * 0.3), 0
while (color_grid == 255).any():
    for i in range(6):
        for j in range(4):
            if color_grid[j, i] == 255:
                # Try to use the...
                # ...chip above the missing one
                found_neighbor = False
                if j > 0 and color_grid[j - 1, i] != 255:
                    found_neighbor = True
                    near_chip = color_chips[color_grid[j - 1, i]]
                    near_x, near_y, = near_chip[:2]
                    x, y = near_x, near_y + y_spacing
                # ...chip to the left
                elif i > 0 and color_grid[j, i - 1] != 255:
                    found_neighbor = True
                    near_chip = color_chips[color_grid[j, i - 1]]
                    near_x, near_y, = near_chip[:2]
                    x, y = near_x + x_spacing, near_y
                # ...chip below
                elif j < 3 and color_grid[j + 1, i] != 255:
                    found_neighbor = True
                    near_chip = color_chips[color_grid[j + 1, i]]
                    near_x, near_y, = near_chip[:2]
                    x, y = near_x, near_y - y_spacing
                # ...chip to the right
                elif i < 5 and color_grid[j, i + 1] != 255:
                    found_neighbor = True
                    near_chip = color_chips[color_grid[j, i + 1]]
                    near_x, near_y, = near_chip[:2]
                    x, y = near_x - x_spacing, near_y
                # If a good neighbor exists, add new chip to the grid
                if found_neighbor:
                    if near_chip[2] > w * 0.8 or near_chip[3] > h * 0.8:
                        x += w_pad
                        y += h_pad
                    chip_w, chip_h = w - 2 * w_pad, h - 2 * h_pad
                    color_chips.append([x, y, chip_w, chip_h])
                    add_chip(len(color_chips) - 1, i, j)
                    n += 1
print("Color chips reconstructed:\t", n)
print("Median size of detected chips:\t x={}, y={}".format(w + pad2, h + pad2))
check_length(color_chips)
cv2.imshow('Confirm sample areas', img_display)
cv2.waitKey(0)

# Get color info
color_info = np.zeros((4, 6, 3))
if img.dtype == np.uint8:
    f = 255.0
elif img.dtype == np.uint16:
    f = 65535.0
for i in range(6):
    for j in range(4):
        x, y, w, h, = color_chips[color_grid[j, i]]
        for k in range(3):
            # print(x, y, w, h, j, i, k)
            color_info[j, i, k] = img[y:y + h, x:x + w, 2 - k].mean() / f

# Linearize (degamma)
if args.input_image[-4:] == '.png':
    color_info = np.power(color_info, args.gamma)

# Write results
print("Normalized color values:\ni  r        g        b")
writer = csv.writer(args.output_csv, lineterminator='\n')
writer.writerow(' rgb')
i = 0
for row in color_info:
    for col in row:
        print('{}, {:.5}, {:.5}, {:.5}'.format(i, *col))
        writer.writerow([i, *col])
        i += 1
