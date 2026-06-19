# 🐱🐶 Cat vs Dog Image Classifier

A Deep Learning project that classifies images as **Cat** or **Dog** using **Transfer Learning with ResNet18** and provides predictions through a **Streamlit Web Application**.

---

## 🚀 Project Overview

This project uses a pre-trained **ResNet18** model from PyTorch and fine-tunes the final classification layer to distinguish between cats and dogs.

The application allows users to upload an image and receive:

* Predicted Class (Cat or Dog)
* Prediction Confidence Score

---

## 📸 Demo

Upload an image:

* Cat Image → Prediction: Cat
* Dog Image → Prediction: Dog

The model returns a confidence percentage along with the prediction.

---

## ✨ Features

* Binary Image Classification
* Transfer Learning using ResNet18
* Streamlit Web Interface
* Confidence Score Display
* Easy Deployment on Streamlit Cloud

---

## 🛠️ Tech Stack

### Deep Learning

* Python
* PyTorch
* Torchvision

### Frontend

* Streamlit

### Image Processing

* Pillow (PIL)

---

## 📂 Project Structure

```text
cat_vs_dog_classifier/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── src/
    ├── model.py
    ├── train.py
    └── Cat_vs_dog_model.pth
```

---

## 🧠 Model Architecture

### Base Model

* ResNet18 (Pretrained on ImageNet)

### Custom Classification Head

```python
nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 2)
)
```

### Classes

```text
0 → Cat
1 → Dog
```

---

## 📊 Dataset

Dataset contains two classes:

```text
dataset/
│
├── cats/
└── dogs/
```

The dataset is automatically split into:

* 80% Training Data
* 20% Validation Data

using:

```python
random_split()
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/cat-vs-dog-classifier.git

cd cat-vs-dog-classifier
```

Create Virtual Environment:

```bash
python -m venv .venv
```

Activate Environment:

### macOS/Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Run:

```bash
cd src

python train.py
```

After training completes:

```text
Cat_vs_dog_model.pth
```

will be generated.

---

## ▶️ Run Streamlit App

From project root:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

in your browser.

---

## 📈 Results

The transfer learning approach significantly improves performance compared to a CNN trained from scratch.

Metrics depend on dataset size and training configuration.

---

## 🔮 Future Improvements

* Grad-CAM Visualization
* Data Augmentation
* Mobile Deployment
* Multi-Class Pet Classification
* Docker Deployment
* CI/CD Pipeline

---

## 👨‍💻 Author

Vikas Yadav

Aspiring Full Stack & AI Developer

GitHub:
https://github.com/vk60

```
```
