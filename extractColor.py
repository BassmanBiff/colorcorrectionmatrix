#!/usr/bin/env python3

import argparse
import csv
import cv2                              # Find chips
import numpy as np
from scipy.optimize import curve_fit    # Find missing chips


# Parse inputhip_area
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

# Load image, find contours
img = cv2.imread(args.input_image)                  # Load
img_display = img.copy()                            # Copy for drawing on
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)        # Convert to grayscale
edges = cv2.Canny(gray, 30, 40)                     # Find gradients
edges, contours, hierarchy = cv2.findContours(      # Find contours and info
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find color chips
color_chips = []
pad = 5
pad2 = pad * 2
for i in range(len(contours)):
    if hierarchy[0][i][3] > -1:     # Contour is closed, contains no others
        r = cv2.boundingRect(contours[i])
        h, w = r[2], r[3]
        if h > args.x * 0.7 and h < args.x * 1.2 and \
           w > args.y * 0.7 and w < args.y * 1.2:
            color_chips.append(
                (r[0] + pad, r[1] + pad, r[2] - pad2, r[3] - pad2))
print("Color chips found: ", len(color_chips))

# Sort color chips, report average h and w for refining input args
color_chips.sort()
h = w = 0
for chip in color_chips:
    h += chip[3]
    w += chip[2]
h = int(h/len(color_chips))
w = int(w/len(color_chips))
print('x, y: {}, {}\n'.format(w + pad2, h + pad2))

# Assemble color grid
color_grid = [[]]


def add_chip(chip, j):
    ''' Add a color chip to both the color grid data and image '''
    color_grid[j].append(chip)
    cv2.rectangle(
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
for i, col in enumerate(color_grid):
    for j, chip in enumerate(col):
        x, y, w, h, = chip
        for k in range(3):
            color_info[j, i, k] = img[y:y+h, x:x+w, 2-k].mean() / 255.

# Linearize (degamma)
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
