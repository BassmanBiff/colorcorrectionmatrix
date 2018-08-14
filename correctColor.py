#!/usr/bin/env python3

import argparse
import colorspace
from PIL import Image


def gamma_table(gamma_r, gamma_g, gamma_b, gain_r=1.0, gain_g=1.0, gain_b=1.0):
    r_tbl = [min(255, int((x / 255.) ** (gamma_r) * gain_r * 255.))
             for x in range(256)]
    g_tbl = [min(255, int((x / 255.) ** (gamma_g) * gain_g * 255.))
             for x in range(256)]
    b_tbl = [min(255, int((x / 255.) ** (gamma_b) * gain_b * 255.))
             for x in range(256)]
    return r_tbl + g_tbl + b_tbl


def applyGamma(img, gamma=2.2):
    inv_gamma = 1. / gamma
    return img.point(gamma_table(inv_gamma, inv_gamma, inv_gamma))


def deGamma(img, gamma=2.2):
    return img.point(gamma_table(gamma, gamma, gamma))


def correctColor(img, ccm):
    return img.convert("RGB", tuple(ccm.transpose().flatten()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ccm', action='store', type=argparse.FileType('r'))
    parser.add_argument('input', action='store')
    parser.add_argument('output', action='store')
    parser.add_argument(
        '-g', '--gamma', action='store', type=float, default=2.2,
        help='Gamma value of reference and source data. (Default=2.2)')
    parser.add_argument(
        '-i', '--illuminant', action='store', type=str, default='D65',
        help='Illuminant value (D50 or D65, default=D65)')
    args = parser.parse_args()
    gamma = args.gamma

    ccm = colorspace.loadCCM(args.ccm, args.illuminant)
    input_img = Image.open(args.input, 'r').convert("RGB")
    input_img = deGamma(input_img, gamma=gamma)
    input_img = colorspace.sRGB2XYZ(input_img, args.illuminant)
    input_img = correctColor(input_img, ccm)
    input_img = colorspace.XYZ2sRGB(input_img, args.illuminant)
    input_img = applyGamma(input_img, gamma=gamma)
    input_img.save(args.output)
