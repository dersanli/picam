import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500
from picamera2.devices.imx500.postprocess import COCODrawer

MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 1280


def parse_detections(metadata, imx500, picam2):
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return []
    input_w, input_h = imx500.get_input_size()
    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        if score < 0.5:
            continue
        # box is [y0, x0, y1, x1] normalised
        y0, x0, y1, x1 = box
        detections.append({
            "box": (x0, y0, x1, y1),
            "score": float(score),
            "class": int(cls),
        })
    return detections


def draw_detections(frame, detections, labels):
    h, w = frame.shape[:2]
    for d in detections:
        x0, y0, x1, y1 = d["box"]
        x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        label = f"{labels[d['class']]}: {d['score']:.2f}" if labels else f"{d['class']}: {d['score']:.2f}"
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, label, (x0, max(y0 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def main():
    imx500 = IMX500(MODEL)
    picam2 = Picamera2(imx500.camera_num)

    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080), "format": "RGB888"},
        controls={"FrameRate": 30},
    )
    picam2.configure(config)
    picam2.start()

    labels = imx500.network_intrinsics.labels

    cv2.namedWindow("picam", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("picam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        frame, metadata = picam2.capture_array("main"), picam2.capture_metadata()
        detections = parse_detections(metadata, imx500, picam2)
        frame = draw_detections(frame, detections, labels)

        # rotate 90° clockwise to fill portrait display
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # scale to display resolution
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        cv2.imshow("picam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
