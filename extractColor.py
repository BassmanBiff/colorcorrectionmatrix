#!/usr/bin/env python3

import argparse
import csv
import cv2                              # Find chips
import numpy as np
import rawpy
import sys
from scipy.optimize import curve_fit    # Find missing chips


def to_uint8(img):
    return (img >> 8).astype(np.uint8)


# Parse input
parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str)
parser.add_argument('output_csv', type=argparse.FileType('w'),
                    default='colorchart.csv')
parser.add_argument('-g', '--gamma', type=float, default=2.2)
parser.add_argument(
    '-s', type=float, default=1.0,
    help='scale factor for display')
parser.add_argument(
    '-x', type=int, default=35,
    help='expected width of color chips, in pixels')
parser.add_argument(
    '-y', type=int, default=25,     # Should be -h and -w, but -h is taken :(
    help='expected height of color chips, in pixels')
args = parser.parse_args()

# Load image, find contours
if args.input_image[-4:] == '.png':
    img = cv2.imread(args.input_image)                  # Load
    img_display = img.copy()
elif args.input_image[-4:] == '.dng':
    raw = rawpy.imread(args.input_image)                # Load
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
        gamma=(1, 1),     # (2.222, 4.5), (1, 1)
        chromatic_aberration=None,
        bad_pixels_path=None)
    img = img << 6  # 10-bit
    img_display = cv2.resize(img, (0, 0), fx=args.s, fy=args.s)
    img_display = to_uint8(img_display)

# cv2.imshow('img', img_display)
# cv2.waitKey(0)

# Find gradients
edges = to_uint8(img)
if img.size > 307200:
    edges = cv2.Canny(edges, 200, 600, apertureSize=5)
else:
    edges = cv2.Canny(edges, 30, 40, apertureSize=3)
cv2.imshow('img', cv2.resize(edges, (0, 0), fx=args.s, fy=args.s))
cv2.waitKey(0)
edges, contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find color chips
color_chips = []
pad = int((args.x + args.y) / 12)
pad2 = pad * 2
for i in range(len(contours)):
    if hierarchy[0][i][3] > -1:     # Contour is closed, contains no others
        r = cv2.boundingRect(contours[i])
        x, y = r[2], r[3]
        if x > args.x * 0.8 and x < args.x * 1.2 and \
           y > args.y * 0.8 and y < args.y * 1.2:
            color_chips.append(
                [r[0] + pad, r[1] + pad, r[2] - pad2, r[3] - pad2])

# Sort color chips, report average h and w for refining input args
color_array = np.array(color_chips)
h = np.median(color_array[:, 3])
w = np.median(color_array[:, 2])
print('Color chip median size: x = {}, y = {}\n'.format(w + pad2, h + pad2))

# Remove false positives
for i, chip in enumerate(color_chips):
    x, y = chip[2], chip[3]
    if x < w * 0.9 or x > w * 1.1 or \
       y < h * 0.9 or y > h * 1.1:
        color_chips[i] = 0
color_chips = [x for x in color_chips if x != 0]

print("Color chips found: ", len(color_chips))
if len(color_chips) < 12:
    print("Can't construct color grid, exiting.")
    sys.exit(2)

# Assemble color grid

# color_grid = np.array((6, 4))
#
#
# def add_chip(chip_i, grid_i, grid_j):
#     ''' Add a color chip to both the color grid data and display image '''
#     color_grid[grid_i, grid_j] = chip_i                           # add to grid
#     chip = [x * args.s for x in color_chips[chip_i]]    # scale for display
#     cv2.rectangle(                                      # add to display
#         img_display,
#         (chip[0], chip[1]),
#         (chip[0] + chip[2], chip[1] + chip[3]),
#         (0, 0, 255),
#         1)
#
#
# # Find top-left chip
# color_chips.sort()
# # ...
# ref_x, ref_y, = color_chips[i][:2]
# add_chip(i, 0, 0)
#
#
# grid_i = grid_j = 0
# for i, chip in enumerate(color_chips):
#     x, y, = chip[:2]
#
#     # Determine column
#     col_results = [0] * 6
#     for i, col in enumerate(col_results):
#         if abs(x - )
#
#
#     # Determine row
#
#     # Add to grid if there isn't already a chip there
#

color_grid = [[]]


def add_chip(chip, j):
    ''' Add a color chip to both the color grid data and display image '''
    color_grid[j].append(chip)              # add to grid
    chip = [int(x * args.s) for x in chip]  # scale for display
    cv2.rectangle(                          # add to display
        img_display,
        (chip[0], chip[1]),
        (chip[0] + chip[2], chip[1] + chip[3]),
        (0, 0, 255),
        1)


j = 0
for i in range(len(color_chips)-1):
    chip = color_chips[i]
    add_chip(chip, j)
    if color_chips[i+1][0] - chip[0] > w/2:
        color_grid.append([])
        j += 1
add_chip(color_chips[-1], j)
for i in range(len(color_grid)):
    color_grid[i] = sorted(color_grid[i], key=lambda x: x[1])

# Reconstruct exactly one missing chip if needed (this sucks)
if len(color_chips) == 23:
    print("Attempting to reconstruct missing color chip...\n")

    # The missing chip's column is the short one
    for i in range(len(color_grid)):
        if len(color_grid[i]) < 4:
            missing_col = i
            break

    # The missing chip's row is more complicated
    # Find y value that defines each row
    rows = [0, 0, 0, 0]
    for i in range(len(color_grid)):
        if i == missing_col:    # Skip the short column to not throw things off
            continue
        j = 0
        for chip in color_grid[i]:
            rows[j] += chip[1] / (len(color_grid) - 1)
            j += 1
    # Remove each row that best matches a y value in the short column
    for chip in color_grid[missing_col]:
        min_dif, i_closest = 9999999, 0
        for i in range(len(rows)):
            dif = abs(chip[1] - rows[i])
            if dif < min_dif:
                min_dif, i_closest = dif, i
        rows[i_closest] = 0
    # Remaining row is the one we want
    for i in range(len(rows)):
        if rows[i] > 0:
            missing_row = i
            break

    # Find x, y from the intersection of the missing chip's row and col
    def line(x, m, b):
        return m * x + b

    x, y = [], []
    for chip in color_grid[missing_col]:
        x.append(chip[0])
        y.append(chip[1])
    col_line = curve_fit(line, x, y)[0]     # [m, b]

    x, y = [], []
    for i, col in enumerate(color_grid):
        if i == missing_col:
            continue
        chip = col[missing_row]
        x.append(chip[0])
        y.append(chip[1])
    row_line = curve_fit(line, x, y)[0]     # [m, b]

    x = -(int((row_line[1] - col_line[1]) / (row_line[0] - col_line[0])))
    y = int(line(x, *row_line))
    add_chip((x, y, w, h), missing_col)
    color_grid[missing_col] = sorted(
        color_grid[missing_col], key=lambda x: x[1])

cv2.imshow('Confirm sample areas', img_display)
cv2.waitKey(0)

# Get color info
color_info = np.zeros((4, 6, 3))
if img.dtype == np.uint8:
    f = 255.0
elif img.dtype == np.uint16:
    f = 65535.0
for i, col in enumerate(color_grid):
    for j, chip in enumerate(col):
        x, y, w, h, = chip
        for k in range(3):
            color_info[j, i, k] = img[y:y+h, x:x+w, 2-k].mean() / f

# Linearize (degamma)
if args.input_image[-4:] == '.png':
    color_info = np.power(color_info, args.gamma)

# Write results
print('Normalized color chip values:\ni  r        g        b')
writer = csv.writer(args.output_csv, lineterminator='\n')
writer.writerow(' rgb')
i = 0
for row in color_info:
    for col in row:
        print('{}, {:.5}, {:.5}, {:.5}'.format(i, *col))
        writer.writerow([i, *col])
        i += 1
