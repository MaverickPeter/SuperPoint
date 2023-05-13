# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
                help="path to the images dir")
ap.add_argument("-t", "--type", required=True,
                help="data type")

args = vars(ap.parse_args())

imagePaths = sorted(list(paths.list_images(args["dir"])))

images = []

# load the images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    if args["type"] == 'nclt':
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif 'cam4' in imagePath:
        print('cam')
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    images.append(image)

# stitch the images together to create a panorama
stitcher = Stitcher()
result = stitcher.stitch(images, args["type"])

result = cv2.rotate(result, cv2.ROTATE_180)
print(result.shape)
# result = result[100:600,...]

# show the images
cv2.imwrite("/mnt/workspace/workgroup/xxc/code/panorama_pyimagesearch/output.jpg", result)
