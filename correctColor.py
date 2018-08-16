#!/usr/bin/env python3

import argparse
import numpy as np
from PIL import Image


def loadCCM(ccmCsvFile, illuminant):
    csvData = ccmCsvFile.read()
    lines = csvData.replace(' ', '').split('\n')
    del lines[len(lines) - 1]

    data = list()
    cells = list()

    for i in range(len(lines)):
        if lines[i] == illuminant:
            cells.append(lines[i + 1].split(','))
            cells.append(lines[i + 2].split(','))
            cells.append(lines[i + 3].split(','))
            cells.append(lines[i + 4].split(','))
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


def gamma_table(gamma_r, gamma_g, gamma_b, gain_r=1.0, gain_g=1.0, gain_b=1.0):
    rng = range(256)
    r_tbl = [min(255, int((x / 255.) ** gamma_r * gain_r * 255.)) for x in rng]
    g_tbl = [min(255, int((x / 255.) ** gamma_g * gain_g * 255.)) for x in rng]
    b_tbl = [min(255, int((x / 255.) ** gamma_b * gain_b * 255.)) for x in rng]
    return r_tbl + g_tbl + b_tbl


def applyGamma(img, gamma=2.2):
    inv_gamma = 1. / gamma
    return img.point(gamma_table(inv_gamma, inv_gamma, inv_gamma))


def deGamma(img, gamma=2.2):
    return img.point(gamma_table(gamma, gamma, gamma))


def sRGB2XYZ(img, illuminant):
    if illuminant == 'D50':
        rgb2xyz = (0.4360747, 0.3850649, 0.1430804, 0,
                   0.2225045, 0.7168786, 0.0606169, 0,
                   0.0139322, 0.0971045, 0.7141733, 0)
    elif illuminant == 'D65':
        rgb2xyz = (0.412391, 0.357584, 0.180481, 0,
                   0.212639, 0.715169, 0.072192, 0,
                   0.019331, 0.119195, 0.950532, 0)
    else:
        raise ValueError('Invalid illuminant: ' + illuminant)
    return img.convert("RGB", rgb2xyz)


def XYZ2sRGB(img, illuminant):
    if illuminant == 'D50':
        xyz2rgb = (3.1338561, -1.6168667, -0.4906146, 0,
                   -0.9787684, 1.9161415,  0.0334540, 0,
                   0.0719453, -0.2289914,  1.4052427, 0)
    elif illuminant == 'D65':
        xyz2rgb = (3.240970, -1.537383, -0.498611, 0,
                   -0.969244, 1.875968,  0.041555, 0,
                   0.055630, -0.203977,  1.056972, 0)
    else:
        raise ValueError('Invalid illuminant: ' + illuminant)
    return img.convert("RGB", xyz2rgb)


def correctColor(img, ccm):
    return img.convert("RGB", tuple(ccm.transpose().flatten()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ccm', action='store', type=argparse.FileType('r'))
    parser.add_argument('input', action='store')
    parser.add_argument('output', action='store')
    parser.add_argument(
        '-g', '--gamma',  action='store', type=float, default=2.2,
        help='Gamma value of reference and source data. (Default=2.2)')
    parser.add_argument(
        '-i', '--illuminant', action='store', type=str, default='D65',
        help='Illuminant, D50 or D65 (default D65)')
    args = parser.parse_args()
    gamma = args.gamma

    ccm = loadCCM(args.ccm, args.illuminant)
    input_img = Image.open(args.input, 'r').convert("RGB")
    input_img = deGamma(input_img, gamma=gamma)
    input_img = sRGB2XYZ(input_img, args.illuminant)
    input_img = correctColor(input_img, ccm)
    input_img = XYZ2sRGB(input_img, args.illuminant)
    input_img = applyGamma(input_img, gamma=gamma)
    input_img.save(args.output)
