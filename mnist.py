import torch
import numpy as np
from mnist_cnn_model import Net
import matplotlib.pyplot as plt

def show_image(image):
    image = image.reshape(28, 28)
    plt.imshow(image, cmap="gray")
    plt.show()


model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt"))

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28)
    image = torch.from_numpy(image).float()
    return image



def predict(image):
    image = preprocess_image(image)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.exp(outputs)
        _, predicted = torch.max(outputs, 1)
        class_probabilities = probabilities[0].tolist()
        class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        result = {
            "predicted_label": predicted.item(),
            "class_probabilities": {label: prob for label, prob in zip(class_labels, class_probabilities)}
        }
    return result


def update_model(image, feedback_label):
    image = preprocess_image(image)
    feedback_label = torch.tensor([feedback_label])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    output = model(image)
    loss = criterion(output, feedback_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Completed training ...")


def handle_user_feedback(image, feedback_label):
    update_model(image, feedback_label)
    print("saving model ...")
    torch.save(model.state_dict(), "mnist_cnn.pt")

