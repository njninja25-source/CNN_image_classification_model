import torch as ts
import torch.nn as nn
import torchvision 
import matplotlib as plt

from torchvision.transforms import transforms

trans = transforms.Compose( [
    transforms.ToTensor(),# image to tensor -> scale(0,1)
    transforms.Normalize((0.5 , 0.5 , 0.5 ) , (0.5 , 0.5 , 0.5 )) # set of values for normalizing( -1 , 1)
])

from torchvision.datasets import CIFAR10

train_data = CIFAR10(root="./data" ,train=True ,download= True ,  transform=trans)
test_data = CIFAR10(root="./data" ,train=False ,download= True ,  transform=trans)

from torch.utils.data import DataLoader

X_train = DataLoader(train_data , batch_size= 64 , shuffle= True)
X_test = DataLoader(test_data , batch_size= 64 )

# model

class CNN(nn.Module):

    def __init__(self):
        super(CNN , self).__init__()

        self.convo_layer = nn.Sequential(
            # 1st convo layer (32 x 32 x 3)
            nn.Conv2d(3 , 32 , kernel_size= 3 , padding = 1 ), # (IN_CHANNEL = RBG  , OUTCHANNEL = NO 0F FILTERS , KERNAL , PADDING)
            nn.ReLU() , 
            nn.MaxPool2d(2 , 2), # ( kernal_size , stride)

            # o/p = (n - f +2P )/stride + 1 ##{ n = dimension of image  , f = filter dimension , p = 1 , stride = 1}
            # o/p = (32 - 3 + 2) +1 = 32

             # 2nd convo layer ( 16 x 16 x {o/p = 32}) #note 32/pooling kernal = 16 
            nn.Conv2d(32 , 64 , kernel_size= 3 , padding = 1 ), # (IN_CHANNEL = 32 filters  , OUTCHANNEL = NO 0F FILTERS , KERNAL , PADDING)
            nn.ReLU() ,
            nn.MaxPool2d(2 , 2),
            
             # 3rd convo layer ( 8 x 8 x 128)
            nn.Conv2d(64 , 128 , kernel_size= 3 , padding = 1 ), # (IN_CHANNEL = 64  , OUTCHANNEL = NO 0F FILTERS , KERNAL , PADDING)
            nn.ReLU() ,
            nn.MaxPool2d(2 , 2)
        )

        self.fc_layer= nn.Sequential(

            # 1st hidden layer 
            nn.Linear(4*4*128 , 256),
            nn.ReLU(),

            # 2nd hidden layer 
            nn.Linear(256,10)
        )
        
    def forward(self , x):
            
            x = self.convo_layer(x)
            x = x.view(x.size(0) , -1) # flatting 
            x = self.fc_layer(x)
            return x 

model = CNN()

critiria = nn.CrossEntropyLoss() # comes with softmax
optimmizer = ts.optim.Adam(model.parameters())

epoches=  10
train_loss = []
val_loss = []

for epoch in range(epoches):
     
     # train 

     model.train()

     runnning_loss =0.00

     for image , label in X_train:
          
          optimmizer.zero_grad()
          outputs = model(image)
          loss = critiria(outputs , label)
          loss.backward()
          optimmizer.step()
          runnning_loss += loss.item()

     epoch_train_loss = runnning_loss/len(X_train)
     train_loss.append(epoch_train_loss)

     # validation & testing

     model.eval()
     total = 0
     correcct = 0

     with ts.no_grad():
          running_val_loss =0.00
          for image , label in X_test:
               
               outputs = model.forward(image)
               loss = critiria(outputs , label)
               running_val_loss +=loss.item()
               _ , predicted = ts.max( outputs , 1)
               correcct += (predicted==label).sum().item() 
               total += label.size(0)
          epoch_val_losses =running_val_loss/len(X_test)
          val_loss.append(epoch_val_losses)
              
     print(f"epoch_train_loss {epoch+1} = {epoch_train_loss} & epoch_val_loss {epoch+1} = {epoch_val_losses} ")
     print(f" model accuracy = {correcct *100/total} %")

# plotting
import pandas as pd

loss_df = pd.DataFrame({
    "Training Loss": train_loss,
    "Validation Loss": val_loss
})

plt.plot(loss_df["Training Loss"], label = "Training Loss")
plt.plot(loss_df["Validation Loss"], label = "Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Losses")

plt.legend()
plt.show()
