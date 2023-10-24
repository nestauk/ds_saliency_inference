from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware

from .model import predict
import json

app = FastAPI()

# CORS
origins = [
    "http://localhost:8080",
    "http://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/predict")
def img_object_detection_to_img(file: bytes = File(...)):
    """
    Object Detection from an image plot bbox on image

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        The json representation of the prediction
    """
    prediction = predict(file)
    return json.dumps(prediction.tolist())
