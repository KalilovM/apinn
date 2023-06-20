# APINN

This project is an image recognition API that uses ResNet50 to predict the contents of an image. It also includes a `/predict_mnist` endpoint that predicts the digit that you draw in a web page. The MNIST model also learns from feedback.

## Getting Started

To get start with this project, you'll need to install Docker. You can find instructions for installing Docker on your operating system [here](https://docs.docker.com/install/).

```
git clone https://github.com/kalilovm/apinn.git
cd apinn
docker build -t apinn .
docker run -p 80:80 apinn
```

This will start the server on `http://localhost:80`.

## API Endpoints

### `/predict`

This endpoint takes an image file as input and returns a JSON object with the predicted class and probability for each class. You can use this endpoint by sending a POST request to `http://localhost:5000/predict` with the image file as the request body.

### `/predict_mnist`

This endpoint takes an image of a handwritten digit as input and returns the predicted digit. You can use this endpoint by visiting `http://localhost:5000/predict_mnist` in your web browser and drawing a digit in the canvas.

## Feedback

The `/predict_mnist` endpoint learns from feedback. After each submit of handwritten digit webpage will ask the real answer. If it's correct, the model will be updated. If it's not, the model will be updated with the correct answer.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. We welcome contributions of all kinds, including bug fixes, new features, and documentation improvements.

## License
