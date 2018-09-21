#!/usr/bin/env python3

import argparse
import colorutils as utils
import numpy as np
from csv import writer as csvwriter


def load_colorchart(filename):
    '''Load color chart as linearized XYZ'''
    # Load data
    colors = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(1, 2, 3))
    assert colors.shape == (24, 3)

    # Process and return data
    colors = np.power(colors, args.gamma)               # Degamma
    colors = utils.RGB2XYZ(colors, args.illuminant)     # Convert to XYZ
    return colors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'reference_csv', type=argparse.FileType('r'))
    parser.add_argument(
        'source_csv', type=argparse.FileType('r'))
    parser.add_argument(
        'output_csv', type=argparse.FileType('w'), default='ccm.csv')
    parser.add_argument(
        '-g', '--gamma', type=float, default=1.0,
        help="Gamma value of reference and source data.")
    parser.add_argument(
        '-i', '--illuminant', type=str, default='D65',
        help="lluminant of source and reference images.")
    parser.add_argument(
        '-v', '--verbose', action="store_true", default=False,
        help="verbose output")
    args = parser.parse_args()

    # Load color charts as XYZ
    ref = load_colorchart(args.reference_csv)
    src = load_colorchart(args.source_csv)

    # # Solve for ccm (4x3 matrix, original method)
    # # src * ccm == ref; (24, 3 + 1) * (4, 3) = (24 * 3)
    # src = np.append(src, np.ones((24, 1)), axis=1)
    # ccm = np.linalg.pinv(src).dot(ref)    # same result as np.linalg.lstsq()

    # Solve for ccm (3x3 matrix)
    ccm, res, = np.linalg.lstsq(src, ref, rcond=None)[:2]

    # Report residuals
    before = ((ref - src) ** 2).sum()
    print("\nResiduals\nBefore: {}\nAfter: {}".format(before, res.sum()))

    # Save result
    if args.verbose:
        print("CCM for {} illuminant:\n".format(args.illuminant), ccm)
    writer = csvwriter(args.output_csv, lineterminator='\n')
    writer.writerow([args.illuminant])
    writer.writerows(ccm)
    print("\nSaved color correction matrix as " + args.output_csv.name)
