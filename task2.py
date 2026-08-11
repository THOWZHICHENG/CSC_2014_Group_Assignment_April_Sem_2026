# -*- coding: utf-8 -*-

"""
Paragraph Detection using Projection Profile

Features:

* Table detection
* Image detection
* Paragraph detection
* Column detection
* Column-aware reading order
* False paragraph removal
* Paragraph crop extraction
  """

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================

# Image List

# ==========================================================

# List of paper images to be processed.

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

# Folder containing the converted paper images.

folder = "Converted Paper (8)/"

# ==========================================================

# Process Each Image

# ==========================================================

for img_name in images:

    # Read the input image.
    
    image = cv2.imread(folder + img_name)
    
    if image is None:
        print("Unable to read:", img_name)
        continue
    
    # Convert to grayscale and get image dimensions.
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    
    # ======================================================
    # Thresholding
    # ======================================================
    
    # Otsu thresholding separates text from the background.
    # Inverted binary makes text white and background black.
    
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # Copy the binary image for removing non-text regions.
    
    clean_binary = binary.copy()
    
    
    # ======================================================
    # Table Detection
    # ======================================================
    
    # Detect long horizontal lines found in tables.
    
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
    
    # Detect long vertical lines found in tables.
    
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
    
    # Combine horizontal and vertical table lines.
    
    table_mask = cv2.add(
        horizontal_lines,
        vertical_lines
    )
    
    # Connect nearby table structures.
    
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
    
    # Find connected table regions.
    
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        table_mask,
        cv2.CV_32S
    )
    
    table_regions = []
    
    for i in range(1, num_labels):
    
        x, y, w, h, area = stats[i]
    
        # Keep only sufficiently large table regions.
    
        if area > 2000:
    
            table_regions.append(
                (x, y, w, h)
            )
    
            # Remove the table from paragraph detection.
    
            clean_binary[
                max(0, y - 10):min(height, y + h + 10),
                max(0, x - 10):min(width, x + w + 10)
            ] = 0
    
    
    # ======================================================
    # Image Detection
    # ======================================================
    
    # Detect large connected regions after removing tables.
    
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        clean_binary,
        cv2.CV_32S
    )
    
    image_regions = []
    
    for i in range(1, num_labels):
    
        x, y, w, h, area = stats[i]
    
        # Large regions are treated as possible images.
    
        if (
            area > 3000
            and w > width * 0.20
            and h > height * 0.05
        ):
    
            image_regions.append(
                (x, y, w, h)
            )
    
            # Remove detected images from paragraph detection.
    
            clean_binary[
                max(0, y - 20):min(height, y + h + 20),
                max(0, x - 20):min(width, x + w + 20)
            ] = 0
    
    # Use the cleaned image for text detection.
    
    binary = clean_binary
    
    
    # ======================================================
    # Column Detection
    # ======================================================
    
    # Vertical projection counts text pixels in each column.
    
    vertical_hist = np.sum(
        binary == 255,
        axis=0
    )
    
    text_columns = []
    start = None
    
    # Detect continuous horizontal text regions.
    
    for i, value in enumerate(vertical_hist):
    
        if value > 0 and start is None:
    
            start = i
    
        elif value == 0 and start is not None:
    
            # Ignore very narrow regions.
    
            if i - start > 30:
    
                text_columns.append(
                    (start, i)
                )
    
            start = None
    
    # Handle a column reaching the right edge.
    
    if start is not None:
    
        text_columns.append(
            (start, width)
        )
    
    # Remove narrow false column detections.
    
    text_columns = [
        (left, right)
        for left, right in text_columns
        if right - left > 30
    ]
    
    
    # ======================================================
    # Paragraph Detection
    # ======================================================
    
    paragraphs = []
    
    # Process each detected column separately.
    
    for column_index, (left, right) in enumerate(text_columns):
    
        column = binary[:, left:right]
    
        # Horizontal projection counts text pixels in each row.
    
        horizontal_hist = np.sum(
            column == 255,
            axis=1
        )
    
        lines = []
        start = None
    
        # Detect individual text lines.
    
        for y, value in enumerate(horizontal_hist):
    
            if value > 0 and start is None:
    
                start = y
    
            elif value == 0 and start is not None:
    
                if y - start > 5:
    
                    lines.append(
                        (start, y)
                    )
    
                start = None
    
        # Handle a text line reaching the bottom.
    
        if start is not None:
    
            lines.append(
                (start, height)
            )
    
        if len(lines) == 0:
            continue
    
    
        # Calculate the spacing between text lines.
    
        gaps = []
    
        for i in range(len(lines) - 1):
    
            gap = (
                lines[i + 1][0]
                - lines[i][1]
            )
    
            if gap > 0:
                gaps.append(gap)
    
        # Median line spacing is used as the normal spacing.
    
        if len(gaps) > 0:
            average_gap = np.median(gaps)
        else:
            average_gap = 10
    
        # Larger gaps are treated as paragraph boundaries.
    
        paragraph_gap = average_gap * 2.5
    
    
        # Group text lines into paragraphs.
    
        para_start = lines[0][0]
    
        for i in range(len(lines) - 1):
    
            current_end = lines[i][1]
            next_start = lines[i + 1][0]
    
            gap = next_start - current_end
    
            if gap > paragraph_gap:
    
                paragraphs.append(
                    (
                        left,
                        para_start,
                        right,
                        current_end,
                        column_index
                    )
                )
    
                para_start = next_start
    
        # Add the final paragraph.
    
        paragraphs.append(
            (
                left,
                para_start,
                right,
                lines[-1][1],
                column_index
            )
        )
    
    
    # ======================================================
    # Reading Order
    # ======================================================
    
    # Process columns from left to right.
    
    ordered_paragraphs = []
    
    sorted_columns = sorted(
        enumerate(text_columns),
        key=lambda c: c[1][0]
    )
    
    for original_index, _ in sorted_columns:
    
        # Select paragraphs belonging to this column.
    
        column_paragraphs = [
            p for p in paragraphs
            if p[4] == original_index
        ]
    
        # Sort paragraphs from top to bottom.
    
        column_paragraphs = sorted(
            column_paragraphs,
            key=lambda p: p[1]
        )
    
        ordered_paragraphs.extend(
            column_paragraphs
        )
    
    
    # ======================================================
    # False Paragraph Removal
    # ======================================================
    
    filtered_paragraphs = []
    
    for para in ordered_paragraphs:
    
        px1, py1, px2, py2, column_index = para
        fake = False
    
        # Compare paragraphs with detected tables and images.
    
        for (x, y, w, h) in (
            image_regions + table_regions
        ):
    
            ox1 = x
            oy1 = y
            ox2 = x + w
            oy2 = y + h
    
            # Calculate the overlapping area.
    
            overlap_width = max(
                0,
                min(px2, ox2)
                - max(px1, ox1)
            )
    
            overlap_height = max(
                0,
                min(py2, oy2)
                - max(py1, oy1)
            )
    
            overlap_area = (
                overlap_width
                * overlap_height
            )
    
            paragraph_area = (
                (px2 - px1)
                * (py2 - py1)
            )
    
            if paragraph_area > 0:
    
                # Calculate the percentage of overlap.
    
                overlap_ratio = (
                    overlap_area
                    / paragraph_area
                )
    
                # Remove the paragraph if most of it belongs
                # to a detected table or image.
    
                if overlap_ratio > 0.7:
    
                    fake = True
                    break
    
        if not fake:
    
            filtered_paragraphs.append(
                para
            )
    
    paragraphs = filtered_paragraphs
    
    
    # ======================================================
    # Display Detection Results
    # ======================================================
    
    display_image = image.copy()
    
    # Draw detected tables in red.
    
    for (x, y, w, h) in table_regions:
    
        cv2.rectangle(
            display_image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )
    
    # Draw detected images in green.
    
    for (x, y, w, h) in image_regions:
    
        cv2.rectangle(
            display_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )
    
    # Draw detected paragraphs in blue and label them.
    
    for idx, (
        left,
        top,
        right,
        bottom,
        column_index
    ) in enumerate(paragraphs):
    
        cv2.rectangle(
            display_image,
            (left, top),
            (right, bottom),
            (255, 0, 0),
            2
        )
    
        cv2.putText(
            display_image,
            f"P{idx + 1}",
            (left, max(20, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )
    
    # Display the final detection result.
    
    plt.figure(figsize=(12, 16))
    
    plt.imshow(
        cv2.cvtColor(
            display_image,
            cv2.COLOR_BGR2RGB
        )
    )
    
    plt.title(
        f"Paragraph Detection - {img_name}"
    )
    
    plt.axis("off")
    plt.show()
    
    
    # ======================================================
    # Save Paragraph Crops
    # ======================================================
    
    # Add a small margin around each paragraph crop.
    
    padding = 30
    paragraph_count = 0
    
    for i, (
        left,
        top,
        right,
        bottom,
        column_index
    ) in enumerate(paragraphs):
    
        # Extract the paragraph area from the binary image.
    
        band = binary[
            top:bottom,
            left:right
        ]
    
        # Find columns containing actual text pixels.
    
        ink_cols = np.where(
            np.sum(
                band == 255,
                axis=0
            ) > 0
        )[0]
    
        if len(ink_cols) > 0:
    
            # Tighten the horizontal crop around the text.
    
            tight_left = left + ink_cols.min()
            tight_right = left + ink_cols.max() + 1
    
        else:
    
            tight_left = left
            tight_right = right
    
        # Add padding while keeping coordinates inside the image.
    
        x1 = max(0, tight_left - padding)
        y1 = max(0, top - padding)
        x2 = min(width, tight_right + padding)
        y2 = min(height, bottom + padding)
    
        # Crop the paragraph from the original colour image.
    
        crop = image[
            y1:y2,
            x1:x2
        ]
    
        # Ignore crops that are too small.
    
        if (
            crop.shape[0] < 20
            or crop.shape[1] < 20
        ):
            continue
    
        paragraph_count += 1
    
        # Generate the output filename.
    
        output_name = (
            f"{img_name[:-4]}"
            f"_Paragraph_{paragraph_count}.png"
        )
    
        # Save the paragraph crop.
    
        cv2.imwrite(
            output_name,
            crop
        )
    
    
    # ======================================================
    # Processing Summary
    # ======================================================
    
    # Display the number of paragraphs extracted from the image.
    
    print(
        f"{img_name}: "
        f"{paragraph_count} paragraphs extracted"
    )

print()
print("==========================================")
print("ALL TASKS COMPLETED")
print("==========================================")
