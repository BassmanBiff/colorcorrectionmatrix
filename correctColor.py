#!/usr/bin/env python3

import argparse
import colorutils as utils
import numpy as np
import cv2


def loadCCM(ccmCsvFile, illuminant):
    csvData = ccmCsvFile.read()
    lines = csvData.replace(' ', '').split('\n')
    del lines[len(lines) - 1]

    data = list()
    cells = list()

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
    parser.add_argument('ccm', action='store', type=argparse.FileType('r'))
    parser.add_argument('input', action='store')
    parser.add_argument('output', action='store', default=None)
    parser.add_argument(
        '-g', '--gamma',  action='store', type=float, default=1.0,
        help='Gamma value of source img, default=1')
    parser.add_argument(
        '-i', '--illuminant', action='store', type=str, default='D65',
        help='Illuminant, D50 or D65 (default D65)')
    args = parser.parse_args()
    args.output = args.output or args.input[:-4] + "_corrected.png"

    # Load image (16-bit RGB)
    ccm = loadCCM(args.ccm, args.illuminant)
    img = utils.load_image(args.input)
    cv2.imshow("Input", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    # Normalize (range 0 - 1)
    img = np.divide(img, 65535, dtype=np.float64)
    # print("input", img.max(), img.min())

    # Linearize (degamma)
    img = np.power(img, args.gamma)
    # print("degamma", img.max(), img.min())

    # Convert to XYZ
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]])
    img = utils.applyMatrix(img, M)
    # print("xyz", img.max(), img.min())

    # Apply CCM
    img = utils.applyMatrix(img, ccm)
    # print("corrected", img.max(), img.min())

    # Convert to sRGB
    img = utils.applyMatrix(img, np.linalg.inv(M))
    # print("sRGB", img.max(), img.min())

    # Reapply gamma
    img[img < 0] = 0
    img = np.power(img, 1/args.gamma)
    # print("gamma", img.max(), img.min())

    # Convert to 16-bit
    img = np.uint16(img * 65535)

    # # TESTING
    # img = cv2.cvtColor(img, cv2.COLOR_XYZ2RGB)
    # print("rgb", img.max()/65535, img.min()/65535)

    # Save and display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, img)
    cv2.imshow("Corrected", img)
    cv2.waitKey(0)
