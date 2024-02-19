import os
import pandas as pd
from PIL import Image
import numpy as np


#citirea datelor de antrenare
train_folder = '/kaggle/input/unibuc-dhc-2023/train.csv'

train_images='/kaggle/input/unibuc-dhc-2023/train_images'


output_dir = '/kaggle/working/'
    
file=pd.read_csv(train_folder)

images_names = file['Image'].tolist()
labels = file['Class'].tolist()

images=[]

for name in images_names:
    image_path = os.path.join(train_images, name)
    image = Image.open(image_path)
    images.append(image)




#impartirea fiecarui pixel in histograme
def histogram(image, bins=(8, 8, 8)):
     rgb_pixels = np.array(image)  #transformam imaginea in numpy array
     
     hist, _ = np.histogramdd(rgb_pixels.reshape(-1, 3), bins=bins, range=[(0, 256), (0, 256), (0, 256)])  #calculam histograma
    
     hist /= np.sum(hist)  #normalizam

     return hist.flatten()  #intoarcem sub forma unui array 1-dimensional
train_data=[]

for image in images: 
    train_data.append(histogram(image))  #datele de antrenare



#citirea datelor de validare
val_folder = '/kaggle/input/unibuc-dhc-2023/val.csv'

val_images='/kaggle/input/unibuc-dhc-2023/val_images'

file_val=pd.read_csv(val_folder)

val_images_names = file_val['Image'].tolist()
val_labels = file_val['Class'].tolist()

val=[]

for name in val_images_names:
    image_path = os.path.join(val_images, name)
    image = Image.open(image_path)
    val.append(histogram(image))  #pregatirea imaginilor

from sklearn.neighbors import KNeighborsClassifier
nb=[6,9,12,15,18]
x=[]
for n in nb:
    model = KNeighborsClassifier(n_neighbors=n,weights="distance",p=1)
    model.fit(train_data,labels)
    x.append(model.score(val,val_labels))  #antrenarea modelului folosind un numar variabil de vecini, distanta Manhattan, iar vecinii mai apropiati au o influenta mai mare asupra predictiei


import matplotlib.pyplot as plt
plt.plot(x, nb)
plt.xlabel('accuracy')
plt.ylabel('neighbours')            #graficul acuratetei


from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
np.set_printoptions(threshold=np.inf)


model = KNeighborsClassifier(n_neighbors=9,weights="distance",p=1)
model.fit(train_data,labels)                # acuratete, precizie, recall, matricea de confuzie pe modelul cel mai bun

acc = model.score(val, val_labels)
print(acc)

pred=model.predict(val)

precision = precision_score(val_labels, pred, average=None,zero_division=0)
recall = recall_score(val_labels, pred, average=None,zero_division=0)

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


confusion_mat = confusion_matrix(val_labels, pred)


plt.imshow(confusion_mat,cmap="Blues")


#scrierea datelor in fisierul final csv
import csv
header = ['Image', 'Class']

data=[]

test_folder = '/kaggle/input/unibuc-dhc-2023/test.csv'

test_images='/kaggle/input/unibuc-dhc-2023/test_images'

    
file_test=pd.read_csv(test_folder)

test_images_names = file_test['Image'].tolist()

final=[]

for name in test_images_names:
    image_path = os.path.join(test_images, name)
    image = Image.open(image_path)
    final.append(histogram(image))
    
    
predictie=model.predict(final)  #predictia

for i in range(len(test_images_names)):
    data.append([test_images_names[i],predictie[i]])
    

with open(output_dir+'output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  
    writer.writerows(data)  