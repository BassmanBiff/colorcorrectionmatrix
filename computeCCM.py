#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import colorutils as utils
import csv
import numpy as np
# from scipy.optimize import fmin


def load_colorchart_csv(f):
    '''Load color chart data
        Input CSV's shape is (25, 4), which contains one extra row and column.
    '''
    data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(1, 2, 3))
    assert data.shape == (24, 3)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference_csv', type=argparse.FileType('r'))
    parser.add_argument('source_csv', type=argparse.FileType('r'))
    parser.add_argument(
        'output_csv', type=argparse.FileType('w'), default='ccm.csv')
    parser.add_argument(
        '-g', '--gamma', type=float, default=1.0,
        help='Gamma value of reference and source data.')
    parser.add_argument(
        '-i', '--illuminant', type=str, default='D65',
        help='Illuminant of source and reference images.')
    args = parser.parse_args()
    gamma = args.gamma

    # Load color charts
    reference_raw = load_colorchart_csv(args.reference_csv)
    source_raw = load_colorchart_csv(args.source_csv)

    # Degamma
    reference_linear = np.power(reference_raw, gamma)
    source_linear = np.power(source_raw, gamma)

    # XYZ
    reference_xyz = utils.sRGB2XYZ(reference_linear, args.illuminant)
    source_xyz = utils.sRGB2XYZ(source_linear, args.illuminant)

    # Original method, gave 4x3 matrix
    # source_xyz * ccm == reference_xyz
    # (24, 3 + 1) * (4, 3) = (24 * 3)
    # source_xyz = np.append(source_xyz, np.ones((24, 1)), axis=1)
    # ccm = np.linalg.pinv(source_xyz).dot(reference_xyz)

    ccm, res, rank, s, = np.linalg.lstsq(source_xyz, reference_xyz, rcond=None)

    # Test
    before = ((reference_xyz - source_xyz) ** 2).sum()
    print("Residuals --- before: {}, after: {}".format(before, res.sum()))

    # Write out
    print('CCM at {}:\n'.format(args.illuminant), ccm)
    writer = csv.writer(args.output_csv, lineterminator='\n')
    writer.writerow([args.illuminant])
    writer.writerows(ccm)
