
import pandas as pd
import numpy as np
import os, datetime

import torch
from torchvision.models import resnet18, resnet34, resnet50
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from torch.optim import Adam

from imageio import imread

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#==================================================================
# prepare hashable Dataset
#==================================================================
class human_protein_Dataset(Dataset):
    '''Human Protein Atlas Image Classification Dataset'''
    
    def __init__(self, df_idx_label_map, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.df_idx_label_map = df_idx_label_map
        #self.df_idx_label_map = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df_idx_label_map)
    
    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df_idx_label_map.iloc[idx, 0])                                                        
        image = torch.Tensor(
                [imread(img_name + '_' + channel + '.png') 
                for channel in ['red', 'green', 'yellow', 'blue']]
                )         

        if self.transform:
            image = self.transform(image)

        labels = np.array(self.df_idx_label_map.iloc[idx, 1:].values, dtype=np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        sample = {  'ID': self.df_idx_label_map.iloc[idx, 0], 
                 'image': image, 
                'labels': labels }
        return sample

# image preprocessing
image_transform = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    #transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Normalize(
            mean=[0.5, 0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225, 0.225]
            )
])


#==================================================================
# read metadata; one-hot encoding of labels, split train/test if necessary
#==================================================================
df_meta_data = pd.read_csv('train.csv')

y_label = [[int(i) for i in j.split()] for j in df_meta_data.Target]
encoder = MultiLabelBinarizer() 
y = encoder.fit_transform(y_label)

df_meta_data = df_meta_data.drop('Target', axis=1).join(pd.DataFrame(y))
#df_meta_data.to_csv('idx_label_map.csv', index=False)

#df_train_meta, df_test_meta = train_test_split(df_meta_data, test_size=0.1)


#==================================================================
# prepare training data generator
#==================================================================
loader_params = {  'batch_size': 24, 
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
            #'collate_fn': torch.stack 
            }
train_set = human_protein_Dataset(df_idx_label_map = df_meta_data,
                                  root_dir='./train',
                                  transform=image_transform)
train_generator = DataLoader(train_set, **loader_params)

#test_set = human_protein_Dataset(df_idx_label_map = df_test_meta,
#                                 root_dir='./train',
#                                 transform=image_transform)                                
#test_generator = DataLoader(test_set, **loader_params)


#==================================================================
# model
#==================================================================
n_channels = 4
n_classes = 28
# resnet18 instantiated for 4-channel input images
model = resnet50(pretrained=True)
model.conv1 = nn.Conv2d(
        n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )        
model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
model.to(device)  

#if restart
#model = torch.load('./ResNet50_model_saved/resnet50_human_protein_cpu_11epochs.torch')

#==================================================================
#training
#==================================================================
# class-weight vector for BCEWithLogitsLoss():
label_counts = np.array(df_meta_data.iloc[:, 1:].sum(axis=0))
weight = torch.tensor(len(df_meta_data)/label_counts - 1.0, dtype=torch.float).log()
weight.to(device)

# loss function
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
loss_fn.to(device)

# optimizer
n_epochs = 30
learning_rate = 1e-4 #changed to 1e-5 after ~10 epochs
optimizer = Adam(model.parameters(), lr=learning_rate)

# training loop; save model after every epoch
for i in range(n_epochs):

    for bn, batch in enumerate(train_generator):
        X_train = batch['image'].to(device, non_blocking=True)
        y_label = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad()            
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_label)
        loss.backward()
        optimizer.step()

        print(f'{datetime.datetime.now().strftime("%H:%M:%S")}\t' +\
              f'epoch {i}:{bn}\t train-loss {loss.item():.8f}')
      
    torch.save(model, f'./ResNet50_model_saved/resnet50_human_protein_cpu_{i+1}epochs.torch')   


#==================================================================
#evaluate test data for submission
#==================================================================
model = torch.load(
    './ResNet50_model_saved/' +\
    'resnet50_human_protein_cpu_6epochs.torch'
)

df_test_meta = pd.read_csv('sample_submission.csv')
test_set = human_protein_Dataset(df_idx_label_map = df_test_meta,
                                 root_dir='./test',
                                 transform=image_transform)

test_loader_params = { 
            'batch_size': 24, 
            'shuffle': False,
            'num_workers': 8,
            #'pin_memory': True,
            #'collate_fn': torch.stack 
            }
test_generator = DataLoader(test_set, **test_loader_params)

all_preds = dict()
with torch.no_grad():
    for batch in test_generator:
        X_test = batch['image'].to(device, non_blocking=True)
        IDs = batch['ID']
        y_pred = model(X_test)
        y_pred = y_pred.sigmoid().numpy() 
        all_preds.update(dict(zip(IDs, y_pred)))

import pickle
pickle.dump(all_preds, open('all_preds_dict.pickle', 'wb'))

threshold = 0.3  #need to bracket threshold based on submissions
def threshold_fn(x): 
    labels = np.nonzero((all_preds[x] > threshold).astype(int))[0]
    return ' '.join(str(i) for i in labels)    
df_test_meta['Predicted'] = df_test_meta['Id'].apply(threshold_fn)  
 
df_test_meta.to_csv('submission.csv', index=False)


#==================================================================


