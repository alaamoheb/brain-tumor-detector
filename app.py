from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import cv2
import os

app = Flask(__name__)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = torch.sigmoid(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model_path = 'brain_tumor_model.pth'  
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  

    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])  
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor


def predict_image(image_path):
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = (output >= 0.5).float().item()
    return "Tumor detected" if prediction == 1 else "No tumor detected"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join('./uploads', imagefile.filename)
    os.makedirs('./uploads', exist_ok=True)
    imagefile.save(image_path)

    
    prediction = predict_image(image_path)

    
    # return jsonify({'prediction': prediction})
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
