import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
import glob
from torch.optim.lr_scheduler import StepLR

class MRI(Dataset):
    def __init__(self, image_dir_yes, image_dir_no):
        tumor = []
        no_tumor = []
        
        for f in glob.iglob(image_dir_yes):
            img = cv2.imread(f)
            img = cv2.resize(img, (128,128)) 
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])  
            img = np.transpose(img, (2, 0, 1))  # Change to C, H, W format
            tumor.append(img)
            
        for f in glob.iglob(image_dir_no):
            img = cv2.imread(f)
            img = cv2.resize(img, (128,128)) 
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            img = np.transpose(img, (2, 0, 1))
            no_tumor.append(img)

        tumor = np.array(tumor, dtype=np.float32)
        no_tumor = np.array(no_tumor, dtype=np.float32)

        tumor_label = np.ones(tumor.shape[0], dtype=np.float32)
        no_tumor_label = np.zeros(no_tumor.shape[0], dtype=np.float32)

        self.images = np.concatenate((tumor, no_tumor), axis=0)
        self.labels = np.concatenate((tumor_label, no_tumor_label))

        self.normalize()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        sample = {'image': self.images[index], 'label': self.labels[index]}
        return sample
    
    def normalize(self):
        self.images = self.images / 255.0  


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


def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()  
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for D in train_loader:
            images = D['image'].to(device)
            labels = D['label'].to(device).float()

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs >= 0.5).float()  
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for D in val_loader:  
                images = D['image'].to(device)
                labels = D['label'].to(device).float()

                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs >= 0.5).float()
                correct_val_predictions += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct_val_predictions / total_val_samples
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return model


def predict(model, image_path, device):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))

    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))  
    img = img / 255.0  

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():  
        output = model(img_tensor)
        prediction = (output >= 0.5).float()  

    return "Tumor detected" if prediction.item() == 1 else "No tumor detected"



mri_dataset = MRI("./data/brain_tumor_dataset/yes/*.jpg", "./data/brain_tumor_dataset/no/*.jpg")

train_size = int(0.8 * len(mri_dataset))
val_size = len(mri_dataset) - train_size

train_dataset, val_dataset = random_split(mri_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)


trained_model = train_model(model, train_loader, val_loader, device)


torch.save(trained_model.state_dict(), 'brain_tumor_model.pth')
print("Model saved successfully!")


trained_model.load_state_dict(torch.load('brain_tumor_model.pth'))
result = predict(trained_model, 'test.jpg', device)
print(result)
