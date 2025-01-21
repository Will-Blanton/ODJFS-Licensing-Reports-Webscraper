"""
Author: Will Blanton

This file contains functions used for preprocessing the pdf images and extracting text from them.
"""
import time

import cv2
import numpy as np
from PIL import Image

DPI = 300

# 500 dpi
# MIN_FIELD_WIDTH = 250
# MIN_FIELD_HEIGHT = 10
# KERNEL = (5, 5)

# 300 dpi
MIN_FIELD_WIDTH = DPI // 2
MIN_FIELD_HEIGHT = DPI // 50
KERNEL = (3, 3)


def display_opencv_image(image):
    """
    Convert an OpenCV image (NumPy array) to a PIL image and display it.

    :param image: (numpy array) OpenCV image in BGR format.
    """

    # convert image to PIL for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # display the image
    pil_image.show()


def preprocess_image(image, display=False):
    if display:
        display_opencv_image(np.array(image))

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, KERNEL, 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
    processed = thresh

    if display:
        display_opencv_image(processed)

    return processed


def get_field_rectangles(image, display=None, verbose=False):
    """
    Detect and draw hierarchical rectangles on a preprocessed binary image.

    :param image: (PIL.Image.Image) The preprocessed binary image (PIL format).
    :param display: Path or boolean to save or display the annotated image. If path is provided, the image is saved. If boolean is provided, the image is displayed.
    :param verbose: (bool) Whether to print debug information.
    :return: (list) List of detected sub-rectangles (fields) in the image.
    """

    # find all rectangles (fields) in the image
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    sub_rectangles = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Parent-child relationship: Sub-rectangles have a parent in the hierarchy
        if hierarchy[0][i][3] != -1:
            # filter by size to only keep document-field rectangles
            if w > MIN_FIELD_WIDTH and h > MIN_FIELD_HEIGHT:
                sub_rectangles.append((x, y, w, h))
                if display:
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # convert to PIL image for display
    if display:
        annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        if isinstance(display, str):
            annotated_image_pil.save(display)

        annotated_image_pil.show()

    if verbose:
        print(f"Detected {len(sub_rectangles)} hierarchical sub-rectangles.")

    sub_rectangles = sorted(sub_rectangles, key=lambda x: x[0])
    sub_rectangles = sorted(sub_rectangles, key=lambda x: x[1])
    return sub_rectangles


def extract_text_from_subrectangles(ocr, image, sub_rectangles, ocr_kwargs=None, verbose=False):
    """
    Extract text from sub-rectangles of an image using EasyOCR.

    :param ocr: (easyocr.Reader) EasyOCR reader object.
    :param image: The preprocessed binary image (OpenCV format).
    :param sub_rectangles: (list) Sorted list of sub-rectangles to extract text from.
    :param ocr_kwargs: (dict) Keyword arguments for the EasyOCR reader.
    :param verbose: (bool) Whether to print debug information.
    :return: (dict) Extracted text from each sub-rectangle. Keys are positions from top-left to bottom-right.
    """

    if ocr_kwargs is None:
        ocr_kwargs = {}

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    extracted_text = [None] * len(sub_rectangles)

    total = 0
    fields = 0
    total_confidence = 0

    for i, (x, y, w, h) in enumerate(sub_rectangles):
        sub_image = image[y:y + h, x:x + w]

        extracted_text[i] = ((x, y, w, h), [])

        if len(sub_image.shape) == 2:
            sub_image = cv2.cvtColor(sub_image, cv2.COLOR_GRAY2RGB)

        # skip empty sub-rectangles (e.g., background)
        if np.sum(sub_image == 0) / sub_image.size < 0.02:
            fields += 1
            continue

        results = ocr.readtext(sub_image, **ocr_kwargs)

        # try to preprocess the image to improve OCR performance (a bit arbitrary as of now)
        # if not results or any([conf < .7 for _, _, conf in results]):
        if not results:
            # og_sub_image = sub_image.copy()
            sub_image = cv2.GaussianBlur(sub_image, KERNEL, 0)
            sub_image = cv2.morphologyEx(sub_image, cv2.MORPH_CLOSE, KERNEL, iterations=2)
            sub_image = cv2.resize(sub_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            results = ocr.readtext(sub_image)

        # if not results:
        #     display_opencv_image(og_sub_image)
        #     display_opencv_image(sub_image)

        if results:
            fields += 1
        elif verbose:
            # display_opencv_image(sub_image)
            print(f"No text detected in sub-rectangle {i}.")

        total += max(1, len(results))

        # save the extracted text and confidence
        for result in results:
            bbox, text, confidence = result
            extracted_text[i][1].append((text, round(confidence, 2)))
            total_confidence += confidence

    if verbose:
        print(
            f"Extracted {fields} / {len(sub_rectangles)} text fields with an average confidence of {total_confidence / total:.2f}.")

    return extracted_text


def process_image(image, ocr, display=False, ocr_kwargs=None, verbose=False):
    """
    Process an image to extract text from sub-rectangles.

    :param image: (PIL.Image.Image) The image to process.
    :param ocr: (easyocr.Reader) EasyOCR reader object.
    :param display: (bool) Whether to display the processed image.
    :param ocr_kwargs: (dict) Keyword arguments for the EasyOCR reader.
    :param verbose: (bool) Whether to print debug information.
    :return: (list) Extracted text from sub-rectangles.
    """
    if verbose:
        start = time.time()
    preprocessed = preprocess_image(image, display=False)
    if verbose:
        print(f"Preprocessing took {time.time() - start:.2f} seconds.")

    if verbose:
        start_rect = time.time()
    sub_rectangles = get_field_rectangles(preprocessed, display=display, verbose=verbose)
    if verbose:
        print(f"Getting field rectangles took {time.time() - start_rect:.2f} seconds.")

    # invert the image for better OCR performance
    final_image = cv2.bitwise_not(preprocessed)

    if verbose:
        start_ocr = time.time()
    extracted_text = extract_text_from_subrectangles(ocr,
                                                     final_image,
                                                     sub_rectangles,
                                                     ocr_kwargs=ocr_kwargs,
                                                     verbose=verbose)
    if verbose:
        print(f"Extracting text took {time.time() - start_ocr:.2f} seconds.")
    return extracted_text
