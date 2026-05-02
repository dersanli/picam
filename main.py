import cv2
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 1280
CONFIDENCE = 0.55


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
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            cv2.imshow("picam", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
