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


def update(msg, img):
    if args.verbose:
        msg = "{:<18}".format(msg)
        if isinstance(img, np.ndarray):
            print(msg + "{:<24}{}".format(img.min(), img.max()))
        else:
            print(msg + "{}".format(img))


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'ccm', action='store', type=argparse.FileType('r'))
    parser.add_argument(
        'input', action='store', type=str)
    parser.add_argument(
        'output', action='store', type=str, nargs='?', default=None)
    parser.add_argument(
        '-g', '--gamma',  action='store', type=float, default=1.0,
        help="Gamma value of source img, default=1")
    parser.add_argument(
        '-i', '--illuminant', action='store', type=str, default='D65',
        help="Illuminant, D50 or D65 (default D65)")
    parser.add_argument(
        '-v', '--verbose', action="store_true", default=False,
        help="verbose output")
    args = parser.parse_args()

    # Load image (16-bit RGB)
    ccm = load_ccm(args.ccm, args.illuminant)
    img = utils.imread(args.input)
    scale = utils.display_scale(img)
    utils.imshow("Input", utils.rgb2bgr(img), scale)  # Need BGR for display

    # Color and gamma correction
    if args.verbose:
        print("\n{:<18}{:<24}{}".format("Processing step", "Min px", "Max px"))
    img = np.divide(img, 65535, dtype=np.float64)   # Normalize (range 0 - 1)
    update("original", img)
    if args.gamma != 1.0:                           # Linearize (degamma)
        img = np.power(img, args.gamma)
        update("degamma", img)
    else:
        update("degamma", "skipped (gamma == 1)")
    img = utils.rgb2xyz(img, args.illuminant)       # XYZ
    update("xyz", img)
    img = img.dot(ccm)                              # Color correction
    update("color correction", img)
    img = utils.xyz2rgb(img, args.illuminant)       # RGB
    update("rgb", img)
    gamma = 2.2 if args.gamma == 1 else args.gamma  # Gamma correction
    img = np.where(img < 0, 0, img ** 1/gamma)
    update("gamma", img)

    # White balance and black level
    n_channels = 3
    white_balance = np.empty(n_channels, dtype=np.float64)
    black_balance = np.empty(n_channels, dtype=np.float64)
    for i in range(n_channels):
        channel = img[..., i]
        white_balance[i], black_balance[i] = channel.max(), channel.min()
    white_level = white_balance.min()               # White balance
    img[img > white_level] = white_level
    update("white balance", img)
    black_level = black_balance.max()               # Black level
    img -= black_level
    update("black level", img)
    img = np.where(img < 0, 0, img / img.max())     # Brightness (renormalize)
    # img[img < 0] = 0
    update("brightness", img)

    # Save and display
    if args.verbose:
        white_balance /= white_balance.max()
        print("\nWhite balance (R, G, B): {}".format(white_balance))
        print("Black level: {}".format(black_level))
    img = np.float32(utils.rgb2bgr(img))            # float32 BGR for OpenCV
    if args.output:                                 # Save
        utils.imwrite(args.output, img)
        print("\nSaved corrected image as " + args.output)
    utils.imshow("Corrected", img, scale)           # Display
