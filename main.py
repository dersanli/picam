import cv2
import numpy as np
import libcamera
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.devices.ai_camera.libcamera_config import LibcameraConfig
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 1280
CONFIDENCE = 0.55

# Patch LibcameraConfig to apply 180° rotation at the sensor level
_orig_libcamera_config_init = LibcameraConfig.__init__

def _patched_libcamera_config_init(self, *args, **kwargs):
    _orig_libcamera_config_init(self, *args, **kwargs)
    self.camera_config["transform"] = libcamera.Transform(hflip=1, vflip=1)

LibcameraConfig.__init__ = _patched_libcamera_config_init


def fit_to_display(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y = (target_h - new_h) // 2
    x = (target_w - new_w) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def main():
    device = AiCamera(frame_rate=15)
    model = SSDMobileNetV2FPNLite320x320()
    device.deploy(model)
    annotator = Annotator(thickness=2, text_thickness=1, text_scale=0.4)

    cv2.namedWindow("picam", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("picam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > CONFIDENCE]
            labels = [
                f"{model.labels[class_id]}: {score:.2f}"
                for _, score, class_id, _ in detections
            ]
            annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)

            img = frame.image
            img = fit_to_display(img, DISPLAY_WIDTH, DISPLAY_HEIGHT)

            cv2.imshow("picam", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
