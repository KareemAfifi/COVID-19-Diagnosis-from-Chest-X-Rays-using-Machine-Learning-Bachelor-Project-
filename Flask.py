from flask import Flask
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import base64
from flask import request
from flask import jsonify
import io
import keras
from keras.preprocessing.image import img_to_array
import shutil
import random
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import os
import cv2
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report ,confusion_matrix,ConfusionMatrixDisplay
from PIL import Image
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dirs = {
    'train': '/content/Data/train',
    'val': '/content/Data/val',
    'test': '/content/Data/test'
}

transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'eval': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_all_preds(model, loader):
    model.eval()
    with torch.no_grad():
        all_preds = torch.tensor([], device=device)
        for batch in loader:
            images = batch[0].to(device)
            preds = model(images)
            all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


def get_confmat(targets, preds):
    stacked = torch.stack(
        (torch.as_tensor(targets, device=device),
         preds.argmax(dim=1)), dim=1
    ).tolist()
    confmat = torch.zeros(3,3, dtype=torch.int16)
    for t, p in stacked:
        confmat[t, p] += 1

    return confmat


def get_results(confmat, classes):
    results = {}
    d = confmat.diagonal()
    for i, l in enumerate(classes):
        tp = d[i].item()
        tn = d.sum().item() - tp
        fp = confmat[i].sum().item() - tp
        fn = confmat[:, i].sum().item() - tp

        accuracy = (tp+tn)/(tp+tn+fp+fn)
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        f1score = (2*precision*recall)/(precision+recall)

        results[l] = [accuracy, recall, precision, f1score]

    return results


def fit(epochs, model, criterion, optimizer, train_dl, valid_dl):
    model_name = type(model).__name__.lower()
    valid_loss_min = np.Inf
    len_train, len_valid = 2200, 200
    fields = [
        'epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'
    ]
    rows = []

    for epoch in range(epochs):
        train_loss, train_correct = 0, 0
        train_loop = tqdm(train_dl)

        model.train()
        for batch in train_loop:
            images, labels = batch[0].to(device), batch[1].to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += get_num_correct(preds, labels)

            train_loop.set_description(f'Epoch [{epoch+1:2d}/{epochs}]')
            train_loop.set_postfix(
                loss=loss.item(), acc=train_correct/len_train
            )
        train_loss = train_loss/len_train
        train_acc = train_correct/len_train

        model.eval()
        with torch.no_grad():
            valid_loss, valid_correct = 0, 0
            for batch in valid_dl:
                images, labels = batch[0].to(device), batch[1].to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                valid_loss += loss.item() * labels.size(0)
                valid_correct += get_num_correct(preds, labels)

            valid_loss = valid_loss/len_valid
            valid_acc = valid_correct/len_valid

            rows.append([epoch, train_loss, train_acc, valid_loss, valid_acc])

            train_loop.write(
                f'\n\t\tAvg train loss: {train_loss:.6f}', end='\t'
            )
            train_loop.write(f'Avg valid loss: {valid_loss:.6f}\n')

            # save model if validation loss has decreased
            # (sometimes also referred as "Early stopping")
            if valid_loss <= valid_loss_min:
                train_loop.write('\t\tvalid_loss decreased', end=' ')
                train_loop.write(f'({valid_loss_min:.6f} -> {valid_loss:.6f})')
                train_loop.write('\t\tsaving model...\n')
                torch.save(
                    model.state_dict(),
                    f'models/lr3e-5_{model_name}_{device}.pth'
                )
                valid_loss_min = valid_loss

    # write running results for plots
    with open(f'outputs/CSVs/{model_name}.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        csv_writer.writerows(rows)


# worker init function for randomness in multiprocess dataloading
# https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
def wif(id):
    process_seed = torch.initial_seed()
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


def load_image(path):
    image = Image.open(path)
    image = transform['eval'](image).unsqueeze(0)
    return image


def deprocess_image(image):
    image = image.cpu().numpy()
    image = np.squeeze(np.transpose(image[0], (1, 2, 0)))
    image = image * np.array((0.229, 0.224, 0.225)) + \
        np.array((0.485, 0.456, 0.406))  # un-normalize
    image = image.clip(0, 1)
    return image


def save_image(image, path):
    # while saving PIL assumes the image is in BGR, and saves it as RGB.
    # But here the image is in RGB, therefore it is converted to BGR first.
    image = image[:, :, ::-1]  # RGB -> BGR
    image = Image.fromarray(image)
    image.save(path)  # saved as RGB
    print(f'GradCAM masked image saved to "{path}".')




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_densenet121(pretrained=False, out_features=None, path=None):
    model = torchvision.models.densenet121(pretrained=pretrained)
    if out_features is not None:
        model.classifier = torch.nn.Linear(
            in_features=1024, out_features=out_features
        )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)



def preprocess_image(image,target_size):
    if image.mode!="RGB":
        image=image.convert("RGB")
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    return image

def getseverityindex(a):
    if(a[0]>a[1] and a[0]>a[2] and a[0]>a[3]):
        return 0
    elif (a[1]>a[0] and a[1]>a[2] and a[1]>a[3]):
        return 1
    elif (a[2]>a[0] and a[2]>a[1] and a[2]>a[3]):
        return 2
    else:
        return 3


densenet121 = get_densenet121(out_features=3, path='C:/Users/user/Desktop/Deploy/lr3e-5_densenet_cuda(3 Categories 99 Better than previous).pth')
print('Model Loaded')
single_model = tf.keras.models.load_model('C:/Users/user/Desktop/Deploy/Severity/')
print('Severity Model Loaded')


app=Flask(__name__)
@app.route("/predict",methods=["POST"])
def predict():

    
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    image= image.save('C:/Users/user/Desktop/Deploy/image/image1/image2/img.png')
#     print(type(image))
#     processed_image=preprocess_image(image,target_size=(300,300))
#     print(type(processed_image))
#     cv2.imwrite('C:/Users/user/Desktop/Deploy/image',processed_image)
    print('saving done')
    
      #train_set = datasets.ImageFolder(root='/content/multiclass/train', transform=transform['eval'])
    test_set = datasets.ImageFolder(root='C:/Users/user/Desktop/Deploy/image', transform=transform['eval'])
      #train_dl = DataLoader(train_set, batch_size=128)
    test_dl = DataLoader(test_set, batch_size=120)

    densenet121 = get_densenet121(out_features=3, path='C:/Users/user/Desktop/Deploy/lr3e-5_densenet_cuda(3 Categories 99 Better than previous).pth')
      #train_preds = get_all_preds(densenet121, train_dl)
    test_preds = get_all_preds(densenet121, test_dl)
      #train_preds.shape, test_preds.shape
      

    a=str((test_preds[0][0]).item())
    b=str((test_preds[0][1]).item())
    c=str((test_preds[0][2]).item())
    
    if(a>b and a>c):    
        image= cv2.imread('C:/Users/user/Desktop/Deploy/image/image1/image2/img.png',0)
        image =cv2.resize(image,(512,512))
        #image= cv2.equalizeHist(image)
        image2 = image[..., np.newaxis]
        image3 = np.array((image2))
        image4 =np.array([image3])
        image5 = image4.astype(np.float32)
        image5=image5/255
        severity=(single_model.predict(image5))
        lt=getseverityindex(severity[0][0][0])
        rt=getseverityindex(severity[0][0][1])
        lm=getseverityindex(severity[0][1][0])
        rm=getseverityindex(severity[0][1][1])
        lb=getseverityindex(severity[0][2][0])
        rb=getseverityindex(severity[0][2][1])
        sum=lt+rt+lm+rm+lb+rb
        if(sum<7):
            state=', Low Severity with a Score of '
        elif(sum<11):
            state=', Moderate Severity with a Score of '
        else:
            state=', High Severity with a Score of '
        response={'prediction':{'info':'It looks that you have COVID-19'+state+str(sum),'severityleft':"Left Lung [out of 3] : Upper Portion: "+ str(lt) + '    Middle Portion:' + str(lm) + '    Lower Portion:' + str(lb),
        'severityright':"Right Lung [out of 3] : Upper Portion: "+ str(rt) + '    Middle Portion:' + str(rm) + '    Lower Portion:' + str(rb), 'description':  '0 (no lung abnormalities)|| 1 (interstitial infiltrates) ||  2 (interstitial and alveolar infiltrates, interstitial dominant) || 3 (interstitial and alveolar infiltrates, alveolar dominant)'}}
    elif (b>a and b>c):
        response={'prediction':{'info':'It looks that you are Normal'}}
    else:
        response={'prediction':{'info':'It looks that you have Pneumonia'}}
    
#     response= {
#         'prediction':{
#         'dog':a,
#         'cat':b
#         }
#     }
    

    return jsonify(response)
    

