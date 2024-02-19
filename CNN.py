import os
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn as nn
transform=ToTensor()

train_folder = '/kaggle/input/unibuc-dhc-2023/train.csv'

train_images='/kaggle/input/unibuc-dhc-2023/train_images'


output_dir = '/kaggle/working/'
    
file=pd.read_csv(train_folder)

images_names = file['Image'].tolist()
labels = file['Class'].tolist()

images=[]

for name in images_names:       #salvarea imaginilor ca si tensori
    image_path = os.path.join(train_images, name)
    image = Image.open(image_path)
    images.append(transform(image))



import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
class Data(Dataset):        #clasa Data care ne permite incarcarea datelor folosind DataLoader
    def __init__(self, images, labels,ev=0):
        self.images = images
        self.labels = labels
        self.transform= transforms.Compose([        #diferite transformari care sa ajute la antrenarea modelului
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomRotation(degrees=30)
])
        if ev==1:
            self.transform=transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        image=self.transform(image)
            
        return image, torch.tensor(label)
    



from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

best_acc = 0  
best_model_path = output_dir+'best_model.pth'


model1=nn.Sequential(       #constructia modelului
    nn.Conv2d(3, 25, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(25),
    nn.MaxPool2d(2, 2),
    nn.ReLU(),
    
    nn.Conv2d(25,50,kernel_size=3,stride=1,padding=1),
    nn.BatchNorm2d(50),
    nn.MaxPool2d(2,2),
    
    nn.ReLU(),
    
    nn.Conv2d(50,128,kernel_size=3,stride=1,padding=1),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(2,2),
    nn.ReLU(),
    nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    
    nn.MaxPool2d(2,2),
    nn.ReLU(),
    
    
    nn.Flatten(),
    nn.Linear(256*4*4,5000),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(5000,1000),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(1000,500),
    nn.ReLU(),
    nn.Linear(500,96)
    
    
    
)

dataset = Data(images, labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)      #incarcarea datelor folosind batchuri de cate 64
optimizer =  torch.optim.Adam(model1.parameters(), lr=0.001)  #optimizatorul Adam cu learning rate de 0,001
f = nn.CrossEntropyLoss()  #functia de pierdere folosita


for epoch in range(50):
    model1.train()
    print(epoch)
    for  images, labels in dataloader:
        optimizer.zero_grad()  #reseteaza gradientii pentru noua iteratie
        outputs = model1(images)
        loss = f(outputs, labels)  #discrepanta dintre ce am presupus noi si valorile adevarate
        loss.backward()  #calculeaza gradientul tinand cont de parametrii modelului
        optimizer.step() #actualizeaza parametrii
        
    model1.eval()
    total_correct=0
    for images, labels in dataset_val:
        outputs = model1(images)
        _, predicted = torch.max(outputs, dim=1)        #acuratetea pe parcurs
        total_correct += (predicted == labels).sum().item()
    print(total_correct/1000)
    if total_correct/1000 > best_acc:
        best_acc = total_correct/1000       #actualizam cel mai bun model
        print(best_acc)
        torch.save(model1.state_dict(), best_model_path)
        
        
best_model =nn.Sequential(
    nn.Conv2d(3, 25, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(25),
    nn.MaxPool2d(2, 2),
    nn.ReLU(),
    
    nn.Conv2d(25,50,kernel_size=3,stride=1,padding=1),
    nn.BatchNorm2d(50),
    nn.MaxPool2d(2,2),
    
    nn.ReLU(),
    
    nn.Conv2d(50,128,kernel_size=3,stride=1,padding=1),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(2,2),
    nn.ReLU(),
    nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    
    nn.MaxPool2d(2,2),
    nn.ReLU(),
    
    
    nn.Flatten(),
    nn.Linear(256*4*4,5000),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(5000,1000),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(1000,500),
    nn.ReLU(),
    nn.Linear(500,96)
    
    
    
)
best_model.load_state_dict(torch.load(best_model_path))
    



val_folder = '/kaggle/input/unibuc-dhc-2023/val.csv'

val_images='/kaggle/input/unibuc-dhc-2023/val_images'

file_v=pd.read_csv(val_folder)

images_names_val = file_v['Image'].tolist()
labels_val = file_v['Class'].tolist()


images_val=[]

for name in images_names_val:
    image_path = os.path.join(val_images, name)
    image = Image.open(image_path)
    images_val.append(transform(image))

validation_dataset=Data(images_val,labels_val,ev=1)
dataset_val=torch.utils.data.DataLoader(validation_dataset, batch_size=1000, shuffle=False)



best_model.eval()
total_correct=0
for images, labels in dataset_val:
    outputs = best_model(images)
    _, predicted = torch.max(outputs, dim=1)
    total_correct = (predicted == labels).sum().item()

accuracy = total_correct / 1000
print(accuracy)

from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt      # acuratete, precizie, recall, matricea de confuzie pe modelul cel mai bun

precision = precision_score(labels, predicted, average=None,zero_division=0)
recall = recall_score(labels, predicted, average=None,zero_division=0)

plt.figure()
plt.plot([i for i in range(0,96)],precision)
plt.xlabel("Class labels")
plt.ylabel("precision")
plt.show()

plt.figure()
plt.plot([i for i in range (0,96)],recall)
plt.xlabel("Class labels")
plt.ylabel("recall")
plt.show()


confusion_mat = confusion_matrix(labels, predicted)


plt.imshow(confusion_mat,cmap="Blues")


#scrierea datelor in fisierul final csv
import csv
header = ['Image', 'Class']

data=[]


test_folder = '/kaggle/input/unibuc-dhc-2023/test.csv'

test_images='/kaggle/input/unibuc-dhc-2023/test_images'

file_train=pd.read_csv(test_folder)

test_images_names = file_train['Image'].tolist()


final=[]

dummy=[0]*5000

for name in test_images_names:
    image_path = os.path.join(test_images, name)
    image = Image.open(image_path)
    final.append(transform(image))

    
test_data=Data(final,dummy)
dataset_test=torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

best_model.eval()
total_correct=0
i=0
for image,labels in dataset_test:
    outputs = best_model(image)
    _, predicted = torch.max(outputs, dim=1)
    data.append([test_images_names[i],predicted.item()])
    i+=1
    
    

    
with open(output_dir+'output5.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  
    writer.writerows(data) 