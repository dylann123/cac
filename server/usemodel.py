# put file location as command argument when using

import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define CNN model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*8*8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64*8*8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Load the saved model parameters
model_state_dict = torch.load("./model/model.pth")

# Recreate the model architecture
model = CNN()
model.load_state_dict(model_state_dict)
model.eval()

# Load the image from the provided file path
file_path = sys.argv[1] # will error if user does not do '...model.py [image path]'
image = Image.open(file_path).convert('RGB')  # Ensure that the image is RGB
image = transform(image).unsqueeze(0)  # Add a batch dimension

# Perform inference
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

# Determine the predicted class
if predicted.item() == 0:
    print("The image is a lowercase a")
elif predicted.item() == 1:
    print("The image is an uppercase A")
else:
    print("Unknown type.")