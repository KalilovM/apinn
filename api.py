import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile,Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import resnet50
import mnist



app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    top5_prob, top5_classes = resnet50.predict(image)
    output = {}
    for i in range(len(top5_classes)):
      output[top5_classes[i]] = top5_prob[i].item()
    return output


@app.get("/", response_class=HTMLResponse)
async def main():
    # get html file from static folder
    with open("static/mnist.html") as f:
        html = f.read()
    return html

@app.post("/predict_mnist")
async def predict_mnist(image: UploadFile):
    contents = await image.read()
    image = Image.open(io.BytesIO(contents))
    result = mnist.predict(image)
    return result


@app.post("/feedback_mnist")
async def feedback_mnist(image: UploadFile, label: int = Form(...)):
    image = Image.open(io.BytesIO(await image.read()))
    mnist.handle_user_feedback(image, label)
    return {"message": "Feedback received and model updated."}