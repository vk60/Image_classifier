import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.model import get_model
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    layout="centered"
)
st.title("Cat vs Dog Classifier")

st.write(
    "Upload an image and the CNN model will predict whether it is a Cat or Dog."
)
# ------Device selection-------
device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
# ------Load model--------------
@st.cache_resource
def load_model():
    model=get_model()
    model.load_state_dict(
        torch.load("src/restnet_Cat_vs_dog_model.pth",map_location=device)
    )
    model.to(device)
    model.eval()
    return model
model=load_model()
# ------Image Transform-----------
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
# -------File Upload----------------
uploaded_file=st.file_uploader("Choose an Image",type=["jpg","jpeg","png"])

# ------Prediction----------

if uploaded_file:
    image=Image.open(uploaded_file).convert("RGB")
    st.image(image,caption="Upload image",use_container_width=True)
    image_tensor=transform(image).unsqueeze(0)
    image_tensor=image_tensor.to(device)
    with torch.no_grad():
        outputs=model(image_tensor)
        probabilities=torch.softmax(outputs,dim=1)
        confidence,predicted=torch.max(probabilities,1)
        class_names=["Cat","Dog"]
        prediction=class_names[predicted.item()]
        st.success(f"Prediction:{prediction}")
        st.write(f"Confidence:{confidence.item()*100:.2f}%")
