from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
import json
import torch


model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229,0.224,0.225]
    )
])

with open('imagenet_class_index.json', 'r') as f:
    class_idx = json.load(f)

class_names = [class_idx[str(k)][1] for k in range(len(class_idx))]

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.inference_mode():
        out = model(image)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    top5_classes = [class_names[idx] for idx in top5_catid]
    return top5_prob, top5_classes
    
