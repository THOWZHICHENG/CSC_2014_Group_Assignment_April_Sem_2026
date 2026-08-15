# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 14:34:12 2026

@author: jjlim
"""

import cv2
import numpy as np


# ==========================================================
# CONFIGURATION
# ==========================================================

CASCADE_PATH = "face_detector.xml"

# Question 1
BRIGHTNESS_THRESHOLD = 100
BRIGHTNESS_VALUE = 60

# Question 3
TALKING_WIDTH = 300
TALKING_HEIGHT = 170

# Question 4
WATERMARK_THRESHOLD = 125


# ==========================================================
# QUESTION 1
# DAY / NIGHT DETECTION
# ==========================================================

def detect_day_or_night(video_path):

    """
    Detect whether a video was recorded during
    daytime or nighttime using the average
    percentage of bright pixels.
    """

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise IOError("Cannot open video: " + video_path)

    bright_ratios = []

    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    for frame_count in range(total_frames):

        success, frame = video.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        histogram = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        )

        total_pixels = gray.size

        # Pixels with brightness above 150
        bright_pixels = np.sum(
            histogram[150:256]
        )

        bright_ratio = (
            bright_pixels / total_pixels
        )

        bright_ratios.append(
            bright_ratio
        )

    video.release()

    if len(bright_ratios) == 0:
        raise IOError(
            "No frames found in: " + video_path
        )

    average_bright_ratio = np.mean(
        bright_ratios
    )

    if average_bright_ratio < 0.20:

        print(
            video_path,
            "Bright pixel ratio:",
            round(average_bright_ratio, 3),
            "-> NIGHT"
        )

        return True

    else:

        print(
            video_path,
            "Bright pixel ratio:",
            round(average_bright_ratio, 3),
            "-> DAY"
        )

        return False


# ==========================================================
# QUESTION 1
# INCREASE BRIGHTNESS
# ==========================================================

def increase_brightness(frame, value):

    """
    Increase brightness using HSV colour space.
    Only the V channel is modified.
    """

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    hsv = hsv.astype(np.int16)

    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2] + value,
        0,
        255
    )

    hsv = hsv.astype(np.uint8)

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


# ==========================================================
# QUESTION 2
# FACE DETECTION + BLURRING
# ==========================================================

def blur_faces(
    frame,
    face_detector
):

    """
    Detect camera-facing faces using Haar Cascade
    and blur all detected faces.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x, y, w, h) in faces:

        face = frame[
            y:y+h,
            x:x+w
        ]

        blurred = cv2.GaussianBlur(
            face,
            (35, 35),
            30
        )

        frame[
            y:y+h,
            x:x+w
        ] = blurred

    return frame


# ==========================================================
# QUESTION 4
# WATERMARK FUNCTION
# ==========================================================

def apply_watermark(
    frame,
    watermark
):

    """
    Apply only the bright parts of the watermark.

    The dark background of the watermark is removed
    using a binary mask, leaving the bright text.
    """

    # Convert watermark to grayscale
    gray = cv2.cvtColor(
        watermark,
        cv2.COLOR_BGR2GRAY
    )

    # Detect bright parts of watermark
    _, mask = cv2.threshold(
        gray,
        WATERMARK_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Extract bright watermark text
    watermark_text = cv2.bitwise_and(
        watermark,
        watermark,
        mask=mask
    )

    # Remove watermark text area from video
    background = cv2.bitwise_and(
        frame,
        frame,
        mask=cv2.bitwise_not(mask)
    )

    # Combine original video and watermark text
    result = cv2.add(
        background,
        watermark_text
    )

    return result


# ==========================================================
# QUESTIONS 2-5
# PROCESS ONE VIDEO
#
# Flow:
#
# Input Video
#     ↓
# Day/Night Detection
#     ↓
# Brightness Adjustment
#     ↓
# Face Detection + Blurring
#     ↓
# Talking Video Overlay
#     ↓
# Watermark 1
#     ↓
# Watermark 2
#     ↓
# End Screen
#     ↓
# Final Output
# ==========================================================

def process_video(
    input_video,
    talking_video,
    endscreen_video,
    watermark1,
    watermark2,
    output_video
):

    print()
    print("==========================================")
    print("Processing:", input_video)
    print("==========================================")

    # ------------------------------------------------------
    # QUESTION 1
    # Detect whether video is day or night
    # ------------------------------------------------------

    night = detect_day_or_night(
        input_video
    )


    # ------------------------------------------------------
    # Load Face Detector
    # ------------------------------------------------------

    face_detector = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    if face_detector.empty():

        raise IOError(
            "Cannot load face detector: "
            + CASCADE_PATH
        )


    # ------------------------------------------------------
    # Open Main Video
    # ------------------------------------------------------

    mainCap = cv2.VideoCapture(
        input_video
    )

    if not mainCap.isOpened():

        raise IOError(
            "Cannot open main video: "
            + input_video
        )


    # ------------------------------------------------------
    # Open Talking Video
    # ------------------------------------------------------

    talkCap = cv2.VideoCapture(
        talking_video
    )

    if not talkCap.isOpened():

        raise IOError(
            "Cannot open talking video: "
            + talking_video
        )


    # ------------------------------------------------------
    # Open End Screen Video
    # ------------------------------------------------------

    endCap = cv2.VideoCapture(
        endscreen_video
    )

    if not endCap.isOpened():

        raise IOError(
            "Cannot open end screen video: "
            + endscreen_video
        )


    # ------------------------------------------------------
    # Get Main Video Properties
    # ------------------------------------------------------

    width = int(
        mainCap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        mainCap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = mainCap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0


    # ------------------------------------------------------
    # Create Output Video
    # ------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )


    # ------------------------------------------------------
    # Load Watermarks
    # ------------------------------------------------------

    wm1 = cv2.imread(
        watermark1
    )

    wm2 = cv2.imread(
        watermark2
    )


    if wm1 is None:

        raise IOError(
            "Cannot load watermark: "
            + watermark1
        )


    if wm2 is None:

        raise IOError(
            "Cannot load watermark: "
            + watermark2
        )


    # ------------------------------------------------------
    # Resize Watermarks
    # ------------------------------------------------------

    wm1 = cv2.resize(
        wm1,
        (width, height)
    )

    wm2 = cv2.resize(
        wm2,
        (width, height)
    )


    # ======================================================
    # PROCESS MAIN VIDEO
    # ======================================================

    while True:

        ret, frame = mainCap.read()

        if not ret:
            break


        # --------------------------------------------------
        # QUESTION 1
        # Increase brightness if nighttime
        # --------------------------------------------------

        if night:

            frame = increase_brightness(
                frame,
                BRIGHTNESS_VALUE
            )


        # --------------------------------------------------
        # QUESTION 2
        # Detect and blur faces
        # --------------------------------------------------

        frame = blur_faces(
            frame,
            face_detector
        )


        # --------------------------------------------------
        # QUESTION 3
        # Overlay Talking Video
        # --------------------------------------------------

        retTalk, talkFrame = (
            talkCap.read()
        )


        # Restart talking video when it ends
        if not retTalk:

            talkCap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            retTalk, talkFrame = (
                talkCap.read()
            )


        if retTalk:

            talkFrame = cv2.resize(
                talkFrame,
                (
                    TALKING_WIDTH,
                    TALKING_HEIGHT
                )
            )

            x = 20
            y = 20

            frame[
                y:y + TALKING_HEIGHT,
                x:x + TALKING_WIDTH
            ] = talkFrame


        # --------------------------------------------------
        # QUESTION 4
        # Add Watermark 1
        # --------------------------------------------------

        frame = apply_watermark(
            frame,
            wm1
        )


        # --------------------------------------------------
        # QUESTION 4
        # Add Watermark 2
        # --------------------------------------------------

        frame = apply_watermark(
            frame,
            wm2
        )


        # --------------------------------------------------
        # Save Processed Frame
        # --------------------------------------------------

        writer.write(
            frame
        )


    # ======================================================
    # QUESTION 5
    # APPEND END SCREEN
    # ======================================================

    while True:

        ret, endFrame = (
            endCap.read()
        )

        if not ret:
            break


        endFrame = cv2.resize(
            endFrame,
            (width, height)
        )


        writer.write(
            endFrame
        )


    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------

    mainCap.release()
    talkCap.release()
    endCap.release()
    writer.release()

    cv2.destroyAllWindows()

    print(
        "Finished:",
        output_video
    )


# ==========================================================
# MAIN PROGRAM
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # ALL FOUR VIDEOS
    # ======================================================

    videos = [

        (
            "Recorded Videos (4)/alley.mp4",
            "processed_alley.mp4"
        ),

        (
            "Recorded Videos (4)/office.mp4",
            "processed_office.mp4"
        ),

        (
            "Recorded Videos (4)/singapore.mp4",
            "processed_singapore.mp4"
        ),

        (
            "Recorded Videos (4)/traffic.mp4",
            "processed_traffic.mp4"
        )

    ]


    # ======================================================
    # PROCESS ALL FOUR VIDEOS
    # ======================================================

    for input_video, output_video in videos:

        process_video(

            input_video,

            "talking.mp4",

            "endscreen.mp4",

            "watermark1.png",

            "watermark2.png",

            output_video

        )


    # ======================================================
    # COMPLETION MESSAGE
    # ======================================================

    print()
    print("==========================================")
    print("ALL FOUR VIDEOS COMPLETED")
    print("==========================================")