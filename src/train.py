import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision import transforms,datasets
from model import get_model
import os


device='cuda' if torch.cuda.is_available() else 'cpu'

print(F"Using device:{device}")
#--------TRANSFORMS----------
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
   
    transforms.Normalize(mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
)
                      
])
# -------DATASET----------------
data=datasets.ImageFolder(root='../data/training_set',transform=transform)
print("Classes:",data.classes)

# Train-test Split
train_size=int(0.8*len(data))
test_size=len(data)-train_size
train_dataset,test_dataset=random_split(data,[train_size,test_size])

print("Train Images=",len(train_dataset))
print("Test Images=",len(test_dataset))

# ------DATALOADER----------------
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

# -------MODEL,LOSS,OPTIMIZER----------
model=get_model().to(device)
creterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(model.fc.parameters(),lr=0.001)

# -------Training Loop-----------------
epochs=10
for epoch in range(epochs):
    model.train()
    running_loss=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        loss=creterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
       

#-------Validation-----------
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
                images=images.to(device)
                labels=labels.to(device)
                outputs=model(images)
                _,predicted=torch.max(outputs,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
                accuracy=100*correct/total
    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"loss={running_loss/len(train_loader):.4f}")
    print(f"val accuracy:{accuracy:.2f}%")           

# -----Save Model--------------
torch.save(
    model.state_dict(),"restnet_Cat_vs_dog_model.pth"
)
print("Model Saved Successfully")
        