from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import numpy as np
import cv2
import io
from datetime import datetime

app = FastAPI()
def draw_boxes(image: np.ndarray, results) -> np.ndarray:
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        # Vẽ bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Vẽ nhãn
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return image
# # Load a model
model = YOLO("best_model.pt")  # load an official model
# # Predict with the model
# results = model("D:\\licensePlate_am\\License-Plate-Detection-Pipeline-with-Experiment-Tracking\\src\\dataset\\clip4_new_12.jpg")  # predict on an image
#
# # Access the results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Đọc ảnh từ request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Dự đoán với YOLOv8
    results = model.predict(source=image, save=False)

    # Vẽ các box lên ảnh
    image_with_boxes = draw_boxes(image, results)

    # Chuyển ảnh về dạng bytes để trả về
    _, img_encoded = cv2.imencode(".jpg", image_with_boxes)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

