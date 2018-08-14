#!/usr/bin/env python3

import numpy as np
import argparse
from PIL import Image, ImageDraw
from colorspace import XYZ2sRGB, sRGB2XYZ
import matplotlib.pyplot as plt


def csvfile2nparray(f):
    str_data = f.read()
    lines = str_data.replace(' ', '').split('\n')
    del lines[len(lines) - 1]

    data = list()
    cells = list()

    for i in range(len(lines)):
        cells.append(lines[i].split(','))

    start_row = 0
    if not cells[0][0].replace(".", "", 1).isdigit():
        del cells[0]
        start_row = 1

    i = 0
    for line in cells:
        data.append(list())
        for j in range(start_row, len(line)):
            data[i].append(float(line[j]))
        i += 1
    # print(data)

    return np.asarray(data, dtype=np.float32)


def loadCCM(ccmCsvFile):
    csvData = ccmCsvFile.read()
    lines = csvData.replace(' ', '').split('\n')
    del lines[len(lines) - 1]

    data = list()
    cells = list()

    for i in range(len(lines)):
        cells.append(lines[i].split(','))

    i = 0
    for line in cells:
        data.append(list())
        for j in range(len(line)):
            data[i].append(float(line[j]))
        i += 1

    return np.asarray(data)


def drawChartComparison(reference, corrected, matchRatio):
    offset = 15
    patch = 100
    patchHalf = patch / 2
    width = offset + (patch + offset) * 6
    height = offset + (patch + offset) * 4
    im = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    for i in range(len(reference)):
        ix = i % 6
        iy = int(i / 6)
        rx = offset + (patch + offset) * ix
        ry = offset + (patch + offset) * iy
        draw.rectangle((rx, ry, rx + patch, ry + patchHalf),
                       fill=(int(reference[i][0] * 255),
                             int(reference[i][1] * 255),
                             int(reference[i][2] * 255)))
        draw.rectangle((rx, ry + patchHalf, rx + patch, ry + patch),
                       fill=(int(corrected[i][0] * 255),
                             int(corrected[i][1] * 255),
                             int(corrected[i][2] * 255)))
        draw.multiline_text((rx + patchHalf - 10, ry + 2 + patch),
                            '{0:3.1f}%'.format(matchRatio[i]),
                            fill=(0, 0, 0))
    return im


def saveResultImg(chart, graph, filename):
    offset = 0
    dst = Image.new('RGB', (max(chart.width, graph.width) + offset,
                            chart.height + graph.height + offset),
                    (255, 255, 255))
    dst.paste(chart, (0, 0))
    dst.paste(graph, (0, chart.height + offset))
    dst.save('{}.png'.format(filename))


def correctChart(source, ccm):
    sourceXYZ = sRGB2XYZ(source)
    correctedSource = []

    sourceXYZ = np.append(sourceXYZ, np.ones((24, 1)), axis=1)
    correctedSource = np.dot(sourceXYZ, ccm)

    return XYZ2sRGB(correctedSource)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ccm', action='store',
                        type=argparse.FileType('r'))
    parser.add_argument('referenceCsv', action='store',
                        type=argparse.FileType('r'))
    parser.add_argument('sourceCsv', action='store',
                        type=argparse.FileType('r'))
    parser.add_argument('outputbasename', action='store',
                        type=str)
    # parser.add_argument(
    #     '-g', '--gamma', action='store', type=float, default=1.0,
    #     help='Gamma value of reference and source data. (Default=1.0)')
    args = parser.parse_args()
    # gamma = args.gamma

    ccm = loadCCM(args.ccm)
    reference = csvfile2nparray(args.referenceCsv)
    source = csvfile2nparray(args.sourceCsv)

    correctedSource = correctChart(source, ccm)
    diff = np.absolute(np.subtract(reference, correctedSource))
    matchRatio = np.multiply(
        np.add(
            np.divide(
                np.subtract(correctedSource, reference).sum(axis=1),
                3),
            0),
        100)
    diffIm = drawChartComparison(reference, correctedSource, matchRatio)

    plt.ylim([-15, 15])
    plt.axes().yaxis.grid(True)
    plt.xlim([-1, 24])
    plt.hlines([0], -1, 25, "red")
    plt.bar(np.arange(len(matchRatio)), matchRatio, align="center", width=0.7)
    plt.vlines([5.5, 11.5, 17.5, 23.5], -20, 20, "red")
    plt.xlabel("Patch")
    plt.ylabel("Match %")
    plt.savefig('graph.png')

    graphIm = Image.open('graph.png', 'r')
    saveResultImg(diffIm, graphIm, args.outputbasename)
