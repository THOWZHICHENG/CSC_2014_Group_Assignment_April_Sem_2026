# -*- coding: utf-8 -*-
"""
Paragraph Detection using Projection Profile

Improved Version:
- Table detection
- Image detection
- Paragraph detection
- Reading order sorting
- Remove false paragraph from images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Image List
# ==========================================================

images = [
    "001.png",
    "002.png",
    "003.png",
    "004.png",
    "005.png",
    "006.png",
    "007.png",
    "008.png"
]

folder = "Converted Paper (8)/"


# ==========================================================
# Loop Images
# ==========================================================

for img_name in images:

    # ======================================================
    # Read Image
    # ======================================================

    image = cv2.imread(folder + img_name)

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # ======================================================
    # Threshold
    # ======================================================

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    clean_binary = binary.copy()

    # ==========================================================
    # TABLE DETECTION
    # ==========================================================

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(30, width // 15), 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(20, height // 30))
    )

    vertical_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1
    )

    table_mask = cv2.add(horizontal_lines, vertical_lines)

    table_mask = cv2.dilate(
        table_mask,
        np.ones((35, 35), np.uint8),
        iterations=2
    )

    table_mask = cv2.morphologyEx(
        table_mask,
        cv2.MORPH_CLOSE,
        np.ones((50, 50), np.uint8),
        iterations=2
    )

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        table_mask,
        cv2.CV_32S
    )

    table_regions = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area > 2000:
            table_regions.append((x, y, w, h))
            clean_binary[
                max(0, y - 10):min(height, y + h + 10),
                max(0, x - 10):min(width, x + w + 10)
            ] = 0

    # ==========================================================
    # IMAGE DETECTION (Improved)
    # ==========================================================

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        clean_binary,
        cv2.CV_32S
    )

    image_regions = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # Detect large image objects
        if (
            area > 3000
            and w > width * 0.20
            and h > height * 0.05
        ):
            image_regions.append((x, y, w, h))
            # remove image + margin
            clean_binary[
                max(0, y - 20):min(height, y + h + 20),
                max(0, x - 20):min(width, x + w + 20)
            ] = 0

    binary = clean_binary

    # ==========================================================
    # Vertical Projection
    # ==========================================================

    vertical_hist = np.sum(binary == 255, axis=0)

    # ==========================================================
    # Column Detection
    # ==========================================================

    text_columns = []
    start = None

    for i, value in enumerate(vertical_hist):
        if value > 0 and start is None:
            start = i
        elif value == 0 and start is not None:
            if i - start > 30:
                text_columns.append((start, i))
            start = None

    if start is not None:
        text_columns.append((start, width))

    # ==========================================================
    # Paragraph Detection
    # ==========================================================

    paragraphs = []

    for left, right in text_columns:
        column = binary[:, left:right]
        horizontal_hist = np.sum(column == 255, axis=1)

        lines = []
        start = None

        for y, value in enumerate(horizontal_hist):
            if value > 0 and start is None:
                start = y
            elif value == 0 and start is not None:
                if y - start > 5:
                    lines.append((start, y))
                start = None

        if start is not None:
            lines.append((start, height))

        if len(lines) == 0:
            continue

        # ======================================================
        # Group lines into paragraphs
        # ======================================================

        gaps = []
        for i in range(len(lines) - 1):
            gap = lines[i + 1][0] - lines[i][1]
            if gap > 0:
                gaps.append(gap)

        if len(gaps) > 0:
            average_gap = np.median(gaps)
        else:
            average_gap = 10

        paragraph_gap = average_gap * 2.5
        para_start = lines[0][0]

        for i in range(len(lines) - 1):
            current_end = lines[i][1]
            next_start = lines[i + 1][0]
            gap = next_start - current_end

            if gap > paragraph_gap:
                paragraphs.append(
                    (left, para_start, right, current_end)
                )
                para_start = next_start

        paragraphs.append(
            (left, para_start, right, lines[-1][1])
        )

    # ==========================================================
    # SORT PARAGRAPHS BY READING ORDER
    # ==========================================================

    paragraphs = sorted(
        paragraphs,
        key=lambda p: (p[1], p[0])  # y position, x position
    )

    # ==========================================================
    # Remove only false paragraphs caused by images/tables
    # ==========================================================

    filtered_paragraphs = []

    for para in paragraphs:
        px1, py1, px2, py2 = para
        fake = False

        for (x, y, w, h) in image_regions + table_regions:
            ox1 = x
            oy1 = y
            ox2 = x + w
            oy2 = y + h

            overlap_width = max(0, min(px2, ox2) - max(px1, ox1))
            overlap_height = max(0, min(py2, oy2) - max(py1, oy1))
            overlap_area = overlap_width * overlap_height
            paragraph_area = (px2 - px1) * (py2 - py1)

            if paragraph_area > 0:
                if overlap_area / paragraph_area > 0.7:
                    fake = True
                    break

        if not fake:
            filtered_paragraphs.append(para)

    paragraphs = filtered_paragraphs

    # ==========================================================
    # Detection Result Display
    # ==========================================================

    display_image = image.copy()

    # Table = Red
    for (x, y, w, h) in table_regions:
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Image = Green
    for (x, y, w, h) in image_regions:
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Paragraph = Blue
    for idx, (left, top, right, bottom) in enumerate(paragraphs):
        cv2.rectangle(display_image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            display_image,
            f"P{idx + 1}",
            (left, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # ==========================================================
    # Save Paragraph Crops
    # ==========================================================

    padding = 10
    paragraph_count = 0

    for i, (left, top, right, bottom) in enumerate(paragraphs):
        band = binary[top:bottom, left:right]
        ink_cols = np.where(np.sum(band == 255, axis=0) > 0)[0]

        if len(ink_cols) > 0:
            tight_left = left + ink_cols.min()
            tight_right = left + ink_cols.max()
        else:
            tight_left = left
            tight_right = right

        x1 = max(0, tight_left - padding)
        y1 = max(0, top - padding)
        x2 = min(width, tight_right + padding)
        y2 = min(height, bottom + padding)

        crop = image[y1:y2, x1:x2]

        if crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        paragraph_count += 1

        # Save paragraph crop as PNG

        cv2.imwrite(
            f"{img_name[:-4]}_Paragraph_{paragraph_count}.png",
            crop
        )