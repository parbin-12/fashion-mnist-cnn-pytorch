import pandas as pd 
from sklearn.model_selection import train_test_split
import torch 
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 

df = pd.read_csv("fashion-mnist_train.csv")
# print(df.head())

#train test split
X = df.drop(columns=['label']).values
y = df['label'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

#normalization from 0.0 to 1.0
X_train = X_train/255.0
X_test = X_test/255.0

class CustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features = torch.tensor(features, dtype = torch.float32).reshape(-1,1,28,28) #converting to 2D 
        self.labels = torch.tensor(labels,dtype = torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self,index):
        return self.features[index],self.labels[index]
    
#loading training and testing dataset 

train_dataset = CustomDataset(X_train,y_train)
test_dataset = CustomDataset(X_test,y_test)

#loading dataset with different parameters 

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,pin_memory=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False,pin_memory=True)

#defining the architecture of simple CNN

class MyNN(nn.Module):
    def __init__(self,input_features):
        super().__init__()
        self.features = nn.Sequential(
            #first conv layer 
            nn.Conv2d(input_features,32,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #second conv layer
            nn.Conv2d(32,64,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
#defining the learning rate and epochs 

learning_rate = 0.01
epochs = 15

model = MyNN(1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = learning_rate,weight_decay=1e-4)

#training loop 

for epoch in range(epochs):
    model.train()
    total_epoch_loss = 0 
    for batch_features,batch_labels in train_loader:
        #forward pass 
        outputs  = model(batch_features)
        #calc loss 
        loss = criterion(outputs,batch_labels)
        #backward pass 
        optimizer.zero_grad()
        loss.backward()
        #update grad
        optimizer.step()
        total_epoch_loss += loss.item()
    avg_loss = total_epoch_loss/len(train_loader)
    print(f"Epoch:{epoch+1},Loss:{avg_loss}")

model.eval()

#evaluation on train data
total =0
correct = 0
with torch.no_grad():
    for batch_features,batch_labels in train_loader:
        outputs = model(batch_features)
        _,predicted = torch.max(outputs,1)
        total = total +batch_labels.shape[0]
        correct = correct +(predicted == batch_labels).sum().item()
    print(f"train_data:{correct/total}")
#evaluation on test data 
total = 0
correct = 0
with torch.no_grad():
    for batch_features,batch_labels in test_loader:
        outputs = model(batch_features)
        _,predicted = torch.max(outputs,1)
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()
    print(f"test_data:{correct/total}")



