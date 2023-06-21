import io

import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import mnist
import resnet50
from camera.camera import Camera

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


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
async def main(request: Request):
    return templates.TemplateResponse("mnist.html", {"request": request})


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


@app.get("/object_detection", response_class=HTMLResponse)
async def object_detection(request: Request):
    return templates.TemplateResponse("object_detection.html", {"request": request})


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get('/video_feed', response_class=HTMLResponse)
async def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return StreamingResponse(gen(Camera()),
                             media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print('stop: ctrl+c')
    uvicorn.run(app, host="0.0.0.0", port=8000)
