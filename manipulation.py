import cv2
import numpy as np
import math
import os


# Gray scaling an image
def gray_image(input_img):
    img_height, img_width = input_img.shape[:2]
    grayImage = np.empty([img_height, img_width], dtype=np.uint8)
    for i in range(img_height):
        for j in range(img_width):
            grayImage[i][j] = int(
                input_img[i][j][0] * 0.2126 + input_img[i][j][1] * 0.7152 + input_img[i][j][2] * 0.0722)

    return grayImage


# Scaling an image
def scale_image(input_img, width_scale, height_scale):
    w, h = input_img.shape[:2]

    # newWidth and newHeight are new width and height of image required after scaling
    newWidth = int(w * width_scale)
    newHeight = int(h * height_scale)

    # calculating the scaling factor
    widthScale = newWidth / (w - 1)
    heightScale = newHeight / (h - 1)

    scaledImage = np.zeros([newWidth, newHeight, 3])

    for i in range(newWidth - 1):
        for j in range(newHeight - 1):
            scaledImage[i + 1, j + 1] = input_img[1 + int(i / widthScale),
                                                  1 + int(j / heightScale)]

    return scaledImage


# Translating an image
def translate_image(input_img, shift_distance):
    h, w = input_img.shape[:2]
    x_distance = shift_distance[0]
    y_distance = shift_distance[1]
    ts_mat = np.array([[1, 0, x_distance], [0, 1, y_distance]])

    translatedImage = np.zeros(input_img.shape, dtype='u1')

    for i in range(h):
        for j in range(w):
            origin_x = j
            origin_y = i
            origin_xy = np.array([origin_x, origin_y, 1])

            new_xy = np.dot(ts_mat, origin_xy)
            new_x = new_xy[0]
            new_y = new_xy[1]

            if 0 < new_x < w and 0 < new_y < h:
                translatedImage[new_y, new_x] = input_img[i, j]
    return translatedImage


# Flipping across horizontal axis
def flip_horizontal(input_img):
    horizontalFlipImage = input_img[:, ::-1]
    return horizontalFlipImage


# Flipping across vertical axis
def flip_vertical(input_img):
    verticalFlipImage = input_img[::-1]
    return verticalFlipImage


# Image inversion
def invert_image(input_img):
    invertedImage = (255 - input_img)
    return invertedImage


# Image Rotation
def naive_image_rotate(input_img, degree):
    rads = math.radians(degree)

    rotatedImage = np.uint8(np.zeros(input_img.shape))

    # Finding the center point of rotated (or original) image.
    height = rotatedImage.shape[0]
    width = rotatedImage.shape[1]

    midx, midy = (width // 2, height // 2)

    for i in range(rotatedImage.shape[0]):
        for j in range(rotatedImage.shape[1]):
            x = (i - midx) * math.cos(rads) + (j - midy) * math.sin(rads)
            y = -(i - midx) * math.sin(rads) + (j - midy) * math.cos(rads)

            x += midx
            y += midy

            x = round(x)
            y = round(y)

            if x >= 0 and y >= 0 and x < input_img.shape[0] and y < input_img.shape[1]:
                rotatedImage[i, j] = input_img[x, y]

    return rotatedImage


def main():
    print('Please enter the path where input image.png is stored:')
    input_image_path = input()
    if os.name == 'nt':
        directory_path = os.path.dirname(input_image_path)
    else:
        directory_path = os.path.dirname(input_image_path)
    print('Reading input image from path : {}'.format(input_image_path))

    # Reading input color image of size 512 * 256
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    img_height, img_width, color = img.shape
    print('Input image size: height = {}, width = {}'.format(img_height, img_width))

    # Saving gray scale image under same directory with name gray_image.png
    grayImage = gray_image(img)
    cv2.imwrite(os.path.join(directory_path, "gray_image.png"), grayImage)
    print('Saved grayscale image at path : {}'.format(directory_path + "/gray_image.png"))

    # Scaling the gray-scale image
    scaledGrayImage = scale_image(grayImage, 1 / 2, 1 / 2)
    cv2.imwrite(os.path.join(directory_path, "gray_image_scaled.png"), scaledGrayImage)
    print('Saved scaled grayscale image at path : {}'.format(directory_path + "/gray_image_scaled.png"))

    # Translating the gray-scale image
    translatedGrayImage = translate_image(grayImage, (50, 50))
    cv2.imwrite(os.path.join(directory_path, "gray_image_translated.png"), translatedGrayImage)
    print('Saved translated grayscale image at path : {}'.format(directory_path + "/gray_image_translated.png"))

    # Flipping the gray-scale image along the horizontal axis
    horizontalFlipGrayImage = flip_horizontal(grayImage)
    cv2.imwrite(os.path.join(directory_path, "gray_image_flip_horizontal.png"), horizontalFlipGrayImage)
    print('Saved horizontally flipped grayscale image at path : {}'.format(
        directory_path + "/gray_image_flip_horizontal.png"))

    # Flipping the gray-scale image along the vertical axis
    verticalFlipGrayImage = flip_vertical(grayImage)
    cv2.imwrite(os.path.join(directory_path, "gray_image_flip_vertical.png"), verticalFlipGrayImage)
    print('Saved vertically flipped grayscale image at path : {}'.format(
        directory_path + "/gray_image_flip_vertical.png"))

    # Inverting the gray-scale image
    invertedGrayImage = invert_image(grayImage)
    cv2.imwrite(os.path.join(directory_path, "gray_image_inversion.png"), invertedGrayImage)
    print('Saved inverted grayscale image at path : {}'.format(directory_path + "/gray_image_inversion.png"))

    rotatedGrayImage = naive_image_rotate(grayImage, -45)
    cv2.imwrite(os.path.join(directory_path, "gray_image_rotated.png"), rotatedGrayImage)
    print('Saved rotated grayscale image at path : {}'.format(directory_path + "/gray_image_rotated.png"))

    # Scaling the input image
    scaledImage = scale_image(img, 1 / 2, 1 / 2)
    cv2.imwrite(os.path.join(directory_path, "image_scaled.png"), scaledImage)
    print('Saved scaled image at path : {}'.format(directory_path + "/image_scaled.png"))

    # Translating the input image
    translatedImage = translate_image(img, (50, 50))
    cv2.imwrite(os.path.join(directory_path, "image_translated.png"), translatedImage)
    print('Saved translated image at path : {}'.format(directory_path + "/image_translated.png"))

    # Flipping the input image along the horizontal axis
    horizontalFlipImage = flip_horizontal(img)
    cv2.imwrite(os.path.join(directory_path, "image_flip_horizontal.png"), horizontalFlipImage)
    print('Saved horizontally flipped image at path : {}'.format(directory_path + "/image_flip_horizontal.png"))

    # Flipping the gray-scale image along the vertical axis
    verticalFlipImage = flip_vertical(img)
    cv2.imwrite(os.path.join(directory_path, "image_flip_vertical.png"), verticalFlipImage)
    print('Saved vertically flipped image at path : {}'.format(directory_path + "/image_flip_vertical.png"))


main()
