import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from coords import Map_Point, DMS_Coord

START_OF_Y_COORDINATE = 1000

LETTER_N_X_THRESHOLD = 500
LETTER_N_Y_THRESHOLD = 1000

LETTER_E_X_THRESHOLD = 800
LETTER_E_Y_THRESHOLD = 1000


def show_image(image, video=False):
    cv.imshow("image", image)
    if video:
        cv.waitKey(1)
    else:
        cv.waitKey(0)
        cv.destroyAllWindows()


# https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
def locate_template(searchIn: cv.Mat, searchFor: cv.Mat, show=False):
    w = searchFor.shape[1]
    h = searchFor.shape[0]

    result = cv.matchTemplate(searchIn, searchFor, cv.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        if show:
            cv.rectangle(searchIn, pt, bottom_right, 255, 2)

    if show:
        plt.subplot(121), plt.imshow(result, cmap="gray")
        plt.title("Matching Result"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(searchIn, cmap="gray")
        plt.title("Detected Point"), plt.xticks([]), plt.yticks([])

        plt.show()

    return loc


def read_numbers(img: cv.Mat, template_digits):
    digits = []
    for i in range(10):
        template = template_digits[i]
        # TODO resize template if the size of the image is smaller than the template

        result = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(result >= 0.9)
        for pt in zip(*loc[::-1]):
            digits.append((pt[0], i))
            # If we would want to show the rectangles around the digits
            # bottom_right = (pt[0] + template.shape[1], pt[1] + template.shape[0])
            # cv.rectangle(img, pt, bottom_right, 255, 2)

    digits.sort(key=lambda x: x[0])
    number = ""
    for digit in digits:
        number += str(digit[1])
    return number


# Function to split the numbers in the image to get the degrees, minutes and seconds, i think this is a better way to do it than the split_numbers function
def split_numbers_with_marks(
    img: cv.Mat, template_marks: dict[str, cv.Mat], template_digits: dict[str, cv.Mat]
):
    image = img.copy()
    gray = image
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    deg_locs = locate_template(gray, template_marks["deg"])
    deg_locs__filtered = [pt for pt in zip(*deg_locs[::-1]) if pt[0] > 0]
    if len(deg_locs__filtered) == 0:
        return None
    deg_pos = deg_locs__filtered[::-1][0]
    min_locs = locate_template(gray, template_marks["min"])
    min_locs__filtered = [pt for pt in zip(*min_locs[::-1]) if pt[0] < 200]
    if len(min_locs__filtered) == 0:
        return None
    min_pos = min_locs__filtered[::-1][0]
    sec_locs = locate_template(gray, template_marks["sec"])
    sec_locs__filtered = [pt for pt in zip(*sec_locs[::-1]) if pt[0] > 200]
    if len(sec_locs__filtered) == 0:
        return None
    sec_pos = sec_locs__filtered[::-1][0]

    deg = read_numbers(gray[0 : gray.shape[1], 0 : deg_pos[0] + 5], template_digits)
    min = read_numbers(
        gray[0 : gray.shape[1], deg_pos[0] : min_pos[0] + 5], template_digits
    )
    sec = read_numbers(
        gray[0 : gray.shape[1], min_pos[0] : sec_pos[0] + 5], template_digits
    )
    sec = sec[:-2] + "." + sec[-2:]

    return DMS_Coord(deg, min, sec)


# Could be used instead of the filtering in the split_numbers_with_marks function
# https://pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/ based on this article
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


# Load the templates for the digits
def load_templates():
    templates = []
    for i in range(10):
        img = cv.imread(f"templates/digit_{i}.jpg", cv.IMREAD_UNCHANGED)
        # If the image is resized we have to resize the template as well
        # width = int(img.shape[1] / 1.77777777778)
        # img = imutils.resize(img, width=width)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        templates.append(img)

    return templates


# Load the templates for the marks
def load_mark_templates():
    marks = ["deg", "min", "sec"]
    templates = {}
    for mark in marks:
        img = cv.imread(f"templates/{mark}_mark.jpg", cv.IMREAD_UNCHANGED)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        templates[mark] = img

    for letter in ["n", "e"]:
        img = cv.imread(f"templates/letter_{letter}.jpg", cv.IMREAD_UNCHANGED)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        templates[letter] = img

    return templates


# Function to get the images of the lat and long from the image
# First we locate the significat points in the image and then we split the input image to get the lat and long
# The input image is the bottom strip of the original image
def get_coords_images_from_image(image: cv.Mat, template_marks: dict[str, cv.Mat]):
    letter_n_locs = locate_template(image, template_marks["n"])
    letter_n_locs_filtered = [
        pt for pt in zip(*letter_n_locs[::-1]) if pt[0] > LETTER_N_X_THRESHOLD
    ]
    if len(letter_n_locs_filtered) == 0:
        return None, None
    letter_n_pos = letter_n_locs_filtered[::-1][0]
    letter_e_locs = locate_template(image, template_marks["e"])
    letter_e_locs_filtered = [
        pt for pt in zip(*letter_e_locs[::-1]) if pt[0] > LETTER_E_X_THRESHOLD
    ]
    if len(letter_e_locs_filtered) == 0:
        return None, None
    letter_e_pos = letter_e_locs_filtered[::-1][0]
    seconds_mark_locs = locate_template(image, template_marks["sec"])
    seconds_mark_locs_filtered = [
        pt for pt in zip(*seconds_mark_locs[::-1]) if pt[0] > letter_e_pos[0]
    ]
    if len(seconds_mark_locs_filtered) == 0:
        return None, None
    seconds_mark_pos = seconds_mark_locs_filtered[::-1][0]

    lat_left_x = letter_n_pos[0]
    lat_left_y = letter_n_pos[1] - 10
    lat_right_x = letter_e_pos[0]
    lat_right_y = letter_n_pos[1] + template_marks["n"].shape[1] + 10
    lat = image[lat_left_y:lat_right_y, lat_left_x:lat_right_x]

    long_left_x = letter_e_pos[0]
    long_left_y = letter_e_pos[1] - 10
    long_right_x = seconds_mark_pos[0] + template_marks["sec"].shape[1] + 5
    long_right_y = letter_e_pos[1] + template_marks["n"].shape[0] + 10
    long = image[long_left_y:long_right_y, long_left_x:long_right_x]

    return lat, long
