import cv2
from typing import List
from app.schemas import DetectionMeta
from app.config import settings


def process_detections(image, results) -> List[DetectionMeta]:
    detections = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < settings.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append(
                DetectionMeta(confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)
            )

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"Plate {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    return detections
