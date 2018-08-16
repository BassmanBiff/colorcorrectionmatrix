# Disclaimer
This is a fork of [lighttransport/colorcorrectionmatrix](https://github.com/lighttransport/colorcorrectionmatrix). I have added a color extraction tool (extractColor.py) and enabled some of the basic illuminant stuff that was incomplete in the original version. All readme text outside of the "Extract color" section and this disclaimer is from the original project.

# Compute Color Correction Matrix (CCM)

We compute Color Correction Matrix A.
In other words, we calculate a 4x3 matrix A which approximate the following equation.  

Let P be a reference color checker matrix (24 x 3) and O be a color checker 
matrix to correct (24 x 3).  
`P = [O 1] A`

![reference](./img/referenceStrobo.png)
![image to correct](./img/renderedStrobo.png)

## Data
We have to prepare color checker patch data as csv format.
There are example data in `data` directory.
- `data/colorchart_photo_strobo_linear.csv`
- `data/colorchart_rendered_strobo_linear.csv`  

They are 24x3 matrix. The data are made by reading pixel values using [Natron2](https://natron.fr/)

## Dependency
- Python
    - numpy
    - matplotlib
    - Pillow 
    - OpenEXR
- C++
    - args.hxx(included in this repo)
    - Eigen3

## Build c++ version of computeCCM

``` shell
$ cd cpp
$ mkdir build
$ cmake ../
$ make
```

## Usage
``` shell 
# computeCCM.py [-h] [-g GAMMA] reference_csv source_csv output_csv
$ computeCCM.py data/colorchart_photo_strobo_linear.csv data/colorchart_rendered_strobo_linear.csv ccm.csv
```
This command generates optimal Color Correction Matrix as csv file (`ccm.csv`)

## Test
We can compare reference data and corrected data using `plotChart.py`

``` shell
$ plotChart.py ccm.csv data/colorchart_photo_strobo_linear.csv data/colorchart_rendered_strobo_linear.csv ccm.csv chart
```
![plot chart](img/result_strobo.png)

Each patch shows reference color and corrected color.
Upper one is reference and lower one is corrected color.
The numbers mean relative error.

# Color Correction
Correct given image using CCM.
`correctColor.py` reads jpg or png images, and
`correctColorExr.py` reads exr images.

## Usage
``` shell
$ correctColor.py ccm.csv reference.png corrected
```

![result](img/stroboCorrected.png)  
corrected image

# Image Diff

Generate diff image between a reference image and a corrected image. We compute a difference between two images and take average of rgb for each pixels.

## Usage
``` shell
$ imageDiff.py photo_reference.png corrected.png
```
![diff](img/diffImg.png)

The difference is small as the color approaches blue and 
the difference is big as the color approaches red.

# Extract colors

Experimental tool to extract average color values from an image containing a standard colorchecker x-lite grid.

## Usage
``` shell
$ extractColor.py source_img.png output_colors.csv -x=30 -y=20
```

source_img.png should be an image of a standard x-lite colorchecker grid to extract color info from.
output_colors.csv will contain a list of average r, g, and b for each color chip. It can be fed directly into computeCCM.
-x and -y can be used to specify the approximate size of color chips to look for, in pixels.

# License

CCM is licensed under MIT license.

## Third party licenses
- [args.hxx](https://github.com/Taywee/args) is licensed under MIT License
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) is licensed under Mozilla Public License


# References

* RGB coordinates of the Macbeth ColorChecker, Danny Pascale. June 1st, 2006 version. http://www.babelcolor.com/index_htm_files/RGB%20Coordinates%20of%20the%20Macbeth%20ColorChecker.pdf
* Color Correction Matrix http://www.imatest.com/docs/colormatrix/
* Raw-to-raw: Mapping between image sensor color responses. CVPR 2014. https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Nguyen_Raw-to-Raw_Mapping_between_2014_CVPR_paper.pdf
