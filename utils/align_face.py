import matplotlib.pyplot as plt
import numpy as np
from mtcnn import MTCNN
# import mediapipe as mp


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def aligner():
    return MTCNN(min_face_size=30)


def align_blazeface(orig_img, aligner=None):
    if orig_img.ndim < 2:
        return None
    if orig_img.ndim == 2:
        orig_img = to_rgb(orig_img)
    orig_img = orig_img[:, :, 0:3]
    height, width = orig_img.shape[:2]

    cropped_arr = []
    bounding_boxes_arr = []
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(np.uint8(orig_img))

        # Draw face detections of each face.
        if results.detections:
            for detection in results.detections:
                xmin_rel = max(0, detection.location_data.relative_bounding_box.xmin)
                ymin_rel = max(0, detection.location_data.relative_bounding_box.ymin)
                width_rel = min(1, detection.location_data.relative_bounding_box.width)
                height_rel = min(1, detection.location_data.relative_bounding_box.height)
                bb = [int(xmin_rel * width),
                      int(ymin_rel * height),
                      int(xmin_rel * width + width_rel * width),
                      int(ymin_rel * height + height_rel * height)]
                if bb[2] - bb[0] < 30 or bb[3] - bb[1] < 30:
                    continue
                cropped = orig_img[bb[0]:bb[2], bb[1]:bb[3], :]
                cropped_arr.append(np.copy(cropped))
                bounding_boxes_arr.append(bb)
    return cropped_arr, bounding_boxes_arr


def align(orig_img, aligner):
    """ run MTCNN face detector """

    if orig_img.ndim < 2:
        return None
    if orig_img.ndim == 2:
        orig_img = to_rgb(orig_img)
    orig_img = orig_img[:, :, 0:3]

    detect_results = aligner.detect_faces(orig_img)
    cropped_arr = []
    bounding_boxes_arr = []
    for dic in detect_results:
        if dic['confidence'] < 0.9:
            continue
        x, y, width, height = dic['box']

        if width < 30 or height < 30:
            continue
        bb = [x, y, x + width, y + height]
        assert bb[0] <= bb[2] and bb[1] < bb[3]
        cropped = orig_img[bb[0]:bb[2], bb[1]:bb[3], :]
        cropped_arr.append(np.copy(cropped))
        bounding_boxes_arr.append(bb)

    return cropped_arr, bounding_boxes_arr
