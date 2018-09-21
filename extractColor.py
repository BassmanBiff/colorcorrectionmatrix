#!/usr/bin/env python3

# TODO: Use all good neighbors when reconstructing chips
# TODO: Use open ("incomplete") contours when reconstructing missing chips
# TODO: Don't require size estimate from user
# TODO: Try template matching to find grid?

import argparse
import colorutils as utils
import csv
import cv2
import numpy as np
import sys


# Color grid (6 x 4): maps color chip indices to grid positions
color_grid = np.full((4, 6), 255, np.uint8)


def add_chip(chip_i, grid_i, grid_j):
    ''' Add color chip to color_grid and outline it for display '''
    # Catch attempt to assign the same spot on the grid twice
    if color_grid[grid_j, grid_i] != 255:
        print("Error: Attempted to reassign color_grid[{}, {}] from {} to {}"
              ".".format(grid_j, grid_i, color_grid[grid_j, grid_i], chip_i))
        sys.exit(2)

    color_grid[grid_j, grid_i] = chip_i                     # add to grid
    chip = [int(v * scale) for v in color_chips[chip_i]]    # scale for display
    cv2.rectangle(                                          # add to display
        img_display,
        (chip[0], chip[1]),
        (chip[0] + chip[2], chip[1] + chip[3]),
        (0, 0, 255),
        1)


def check_length(array):
    '''Quit if too few or too many color chips detected'''
    length = len(array)
    if length < 12 or length > 24:
        print("Error: Found {} chips, can't construct color grid"
              ".".format(length))
        sys.exit(2)


# Parse input
parser = argparse.ArgumentParser()
parser.add_argument(
    'input_image', type=str)
parser.add_argument(
    'output_csv', type=argparse.FileType('w'), default='colorchart.csv')
parser.add_argument(
    '-g', '--gamma', type=float, default=1.0,
    help="gamma value of input image")
parser.add_argument(
    '-x', type=int,
    help="expected width of color chips, in pixels")
parser.add_argument(
    '-y', type=int,     # Should be -h and -w, but -h is taken by "help" :(
    help="expected height of color chips, in pixels")
parser.add_argument(
    '-v', '--verbose', action="store_true", default=False,
    help="verbose output")
args = parser.parse_args()

# Load image (16-bit RGB)
img = utils.imread(args.input_image)

# Make copy for display (8-bit BGR)
scale = utils.display_scale(img)
img_display = cv2.resize(img, (0, 0), fx=scale, fy=scale)
img_display = cv2.cvtColor(np.uint8(img_display >> 8), cv2.COLOR_RGB2BGR)
utils.imshow('Input', img_display)

# Degamma
img = np.power(img, args.gamma)

# Find edges
img_edges = np.uint8(np.uint16(img) >> 8)                   # 8-bit (for Canny)
img_edges = cv2.cvtColor(img_edges, cv2.COLOR_RGB2GRAY)     # Grayscale
median, sigma = np.median(img_edges), 0.33                  # Thresholds
img_edges = cv2.Canny(                                      # Gradients
    img_edges,
    int(max(0, (1.0 - sigma) * median)),
    int(min(255, (1.0 + sigma) * median)))
img_edges, contours, hierarchy = cv2.findContours(          # Contours
    img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# utils.imshow('Edges', img_edges, scale)

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

# Remove false positive color chips
color_array = np.array(color_chips)
h, w = int(np.median(color_array[:, 3])), int(np.median(color_array[:, 2]))
for i, chip in enumerate(color_chips):
    x, y, = chip[2:]
    if x < w * 0.9 or x > w * 1.1 or \
       y < h * 0.9 or y > h * 1.1:
        color_chips[i] = 0
color_chips = [chip for chip in color_chips if chip != 0]
check_length(color_chips)
print("\nColor chips found:\t\t", len(color_chips))

# Find leftmost and top chips to define first column and row
# HACK: Assumes we've found a chip in both the top row and first column
left_i = top_i = 0
left_x, top_y = color_chips[0][:2]
for i in range(1, len(color_chips)):
    x, y = color_chips[i][:2]
    if x < left_x:
        left_i, left_x = i, x
    if y < top_y:
        top_i, top_y = i, y

# Assign column and row to detected chips
# TODO: This could be smarter
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

# Reconstruct chips that weren't detected
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
utils.imshow("Confirm sample areas", img_display)

# Get normalized (range 0 - 1) color info from each chip
color_info = np.zeros((4, 6, 3))
for i in range(6):
    for j in range(4):
        x, y, w, h, = color_chips[color_grid[j, i]]
        for k in range(3):
            color_info[j, i, k] = img[y:y + h, x:x + w, k].mean() / 65535

# Write results
if args.verbose:
    print("Normalized color values:\ni  r        g        b")
writer = csv.writer(args.output_csv, lineterminator='\n')
writer.writerow(' rgb')
i = 0
for row in color_info:
    for col in row:
        if args.verbose:
            print('{}, {:.5}, {:.5}, {:.5}'.format(i, *col))
        writer.writerow([i, *col])
        i += 1
print("\nSaved color chart as " + args.output_csv.name)
