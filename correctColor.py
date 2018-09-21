#!/usr/bin/env python3

# TODO: Verify renormalization; use original max/min?

import argparse
import colorutils as utils
import numpy as np


def load_ccm(ccm_csv, illuminant):
    csvData = ccm_csv.read()
    lines = csvData.replace(' ', '').split('\n')
    del lines[len(lines) - 1]

    data, cells = list(), list()
    for i in range(len(lines)):
        if lines[i] == illuminant:
            j = 1
            while i + j < len(lines) and ',' in lines[i + j]:
                cells.append(lines[j].split(','))
                j += 1
            break
    else:
        raise ValueError('Illuminant not found in ccm: ' + illuminant)

    i = 0
    for line in cells:
        data.append(list())
        for j in range(len(line)):
            data[i].append(float(line[j]))
        i += 1

    return np.asarray(data)


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'ccm', action='store', type=argparse.FileType('r'))
    parser.add_argument(
        'input', action='store', type=str)
    parser.add_argument(
        'output', action='store', type=str, default=None)
    parser.add_argument(
        '-g', '--gamma',  action='store', type=float, default=1.0,
        help="Gamma value of source img, default=1")
    parser.add_argument(
        '-i', '--illuminant', action='store', type=str, default='D65',
        help="Illuminant, D50 or D65 (default D65)")
    args = parser.parse_args()
    args.output = args.output or args.input[:-4] + "_corrected.png"

    # Load image (16-bit RGB)
    ccm = load_ccm(args.ccm, args.illuminant)
    img = utils.imread(args.input)
    scale = min(1024 / max(img.shape), 1)
    utils.imshow("Input", utils.RGB2BGR(img), scale)  # Need BGR for display

    # Process image
    print("\nProcessing step  Max px\t\tMin px")
    img = np.divide(img, 65535, dtype=np.float64)   # Normalize (range 0 - 1)
    print("Original\t", img.max(), img.min())
    if args.gamma != 1.0:
        img = np.power(img, args.gamma)             # Linearize (degamma)
        print("Degamma\t\t", img.max(), img.min())
    else:
        print("Degamma\t\t skipped (gamma == 1)")
    img = utils.RGB2XYZ(img, args.illuminant)       # Convert to XYZ
    print("XYZ\t\t", img.max(), img.min())
    img = img.dot(ccm)                              # Apply CCM
    print("Corrected\t", img.max(), img.min())
    img = utils.XYZ2RGB(img, args.illuminant)       # Convert to RGB
    print("RGB\t\t", img.max(), img.min())
    if img.min() < 0:                               # Fix values < 0
        # img -= img.min()
        img[img < 0] = 0
    gamma = 2.2 if args.gamma == 1 else args.gamma  # Choose gamma correction
    img = np.power(img, 1/gamma)                    # Apply gamma correction
    print("Gamma\t\t", img.max(), img.min())
    white_balance = np.empty(3, dtype=np.float64)   # White balance + black lvl
    black_balance = np.empty(3, dtype=np.float64)
    for i in range(3):
        white_balance[i] = img[:, :, i].max()
        black_balance[i] = img[:, :, i].min()
    white_level = white_balance.min()
    img[img > white_level] = white_level            # Apply white balance
    print("White balance\t", img.max(), img.min())
    black_level = black_balance.max()
    img -= black_level                              # Apply black level
    print("Black level\t", img.max(), img.min())
    img[img < 0] = 0                                # Renormalize (0 - 1)
    img /= img.max()
    print("Renormed\t", img.max(), img.min())

    # Save and display
    white_balance /= white_balance.max()
    print("\nWhite balance (R, G, B): {}".format(white_balance))
    print("Black level: {}".format(black_level))
    img = np.uint16(utils.BGR2RGB(img) * 65535)     # 16-bit BGR for OpenCV
    utils.imwrite(args.output, img)                 # Save
    print("\nSaved corrected image as " + args.output)
    utils.imshow("Corrected", img, scale)           # Display
