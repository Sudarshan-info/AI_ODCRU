from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io

from app.model import load_model
from app.detector import process_detections
from app.config import settings

app = FastAPI(
    title="License Plate Detection API",
    description="Detects license plates using YOLO and returns annotated images",
)

model = load_model()


@app.post("/detect", tags=["Detection"])
async def detect_license_plate(file: UploadFile = File(...)):
    try:
        # Read image
        bytes_data = await file.read()
        img_array = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run YOLO
        results = model(image)

        # Process detections + draw boxes
        detections = process_detections(image, results)

        # Optional: do not raise 404; just return empty image
        # if not detections:
        #     raise HTTPException(status_code=404, detail="No license plate detected")

        # Encode image
        _, encoded_img = cv2.imencode(settings.image_format, image)

        return StreamingResponse(
            io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
