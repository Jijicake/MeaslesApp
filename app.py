from flask import Flask, request, render_template, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
import uuid
torch.set_num_threads(1)
app = Flask(__name__)

b = os.path.dirname(os.path.abspath(__file__))
upload = os.path.join(b, 'uploads')

if not os.path.exists(upload):
    os.makedirs(upload)

app.config['UPLOAD_FOLDER'] = upload
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten_dim = 24 * 53 * 53
        self.fcl1 = nn.Linear(self.flatten_dim, 120)
        self.dropout = nn.Dropout(0.5)
        self.fcl2 = nn.Linear(120, 84)
        self.fcl3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fcl1(x))
        x = self.dropout(x)
        x = F.relu(self.fcl2(x))
        x = self.dropout(x)
        x = self.fcl3(x)
        return x

device = torch.device('cpu')
net = NeuralNet().to(device)
net.load_state_dict(torch.load('MODEL PATH', map_location=device))
net.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class_names = ['measles', 'not measles']

def transform_image(image_path):
    print("open")
    image = Image.open(image_path).convert('RGB')
    print("did it")
    image = transform(image)
    return image.unsqueeze(0).to(device)

def get_prediction(image_tensor):
    with torch.no_grad():
        output = net(image_tensor)
        _, predicted = torch.max(output, 1)
    print("predict done")
    return class_names[predicted.item()]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            print("image is up")
            file = request.files['file']
            if file.filename != '':
                filename = f"{uuid.uuid4().hex}.jpg"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print("file saving rn")
                file.save(file_path)
                print("successfully saved")

                image_tensor = transform_image(file_path)
                prediction = get_prediction(image_tensor)

                return render_template('index.html', prediction=prediction, filename=filename)

        else:
            print("trying to clear")
            folder = app.config['UPLOAD_FOLDER']
            deleted_files = []
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_files.append(filename)
                except Exception as e:
                    print(f"failed at {file_path}: {e}")

            return render_template('index.html', deleted=deleted_files)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=False)
