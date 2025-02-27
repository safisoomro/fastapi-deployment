import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torchvision.models as models
import os

# Get the current directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mvtec_model.pth")

# Initialize ResNet18 model with the correct number of output classes (2)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)  # Modify the fully connected layer

# Load the saved model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

# Set model to evaluation mode
model.eval()

# Define transformations (should match validation transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize FastAPI
app = FastAPI()

# Root endpoint to prevent 404 errors
@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI server!"}

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        result = "Defective" if predicted.item() == 1 else "Good"
        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}
