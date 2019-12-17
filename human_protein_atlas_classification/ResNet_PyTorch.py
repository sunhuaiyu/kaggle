
import pandas as pd
import numpy as np
import os

import torch
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from torch.optim import Adam

from imageio import imread

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hashable Dataset
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
                ) / 255.          
        
        labels = np.array(self.df_idx_label_map.iloc[idx, 1:].values, dtype=np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels
    

# process initial data; one-hot encoding of labels, split train/test
df_meta_data = pd.read_csv('train.csv')

y_label = [[int(i) for i in j.split()] for j in df_meta_data.Target]
encoder = MultiLabelBinarizer() 
y = encoder.fit_transform(y_label)

df_meta_data = df_meta_data.drop('Target', axis=1).join(pd.DataFrame(y))
df_meta_data.to_csv('idx_label_map.csv', index=False)

df_train_meta, df_test_meta = train_test_split(df_meta_data, test_size=0.1)

# weight vector for BCEWithLogitsLoss():
label_counts = np.array(df_meta_data.iloc[:, 1:].sum(axis=0))
weight = torch.tensor(len(df_meta_data) / label_counts - 1.0, dtype=torch.float)

# If on laptop -- 
# df_train_meta, df_test_meta = train_test_split(df_meta_data.iloc[:100, :], test_size=0.1)

#
image_transform = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    #transforms.ToTensor(),
    transforms.Normalize(
            mean=[0.5, 0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225, 0.225]
            )
])

# prepare data generators
loader_params = {  'batch_size': 20, 
            'shuffle': True,
            'num_workers': 4,
            #'collate_fn': torch.stack 
            }

train_set = human_protein_Dataset(df_idx_label_map = df_meta_data, #df_train_meta,
                                  root_dir='./train',
                                  transform=image_transform)
train_generator = DataLoader(train_set, **loader_params)
#b = next(iter(train_generator))

test_set = human_protein_Dataset(df_idx_label_map = df_test_meta,
                                 root_dir='./train',
                                 transform=image_transform)                                
test_generator = DataLoader(test_set, **loader_params)


# model selection
n_channels = 4
n_classes = 28
# resnet50 instantiated for 4-channel input images
model = resnet50(pretrained=False)
model.conv1 = nn.Conv2d(
        n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )        
model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
#model.to(device)  

#training
n_epochs = 5
learning_rate = 0.0001
optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)


for i in range(n_epochs):

    for batch in train_generator:
        X_train, y_label = batch
        #X_train, y_label = X_train.to(device), y_label.to(device)
        
        optimizer.zero_grad()            
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_label)
        loss.backward()
        optimizer.step()

        print(f'epoch {i}     train-loss {loss}')
      
torch.save('191210_human_protein_model1.torch')   
   
'''
    with torch.no_grad():
        for batch in test_generator:
            X_test, y_label = batch
            #X_test, y_label = X_test.to(device), y_label.to(device)
            y_pred = model(X_test)
            loss2 = loss_fn(y_pred, y_label)
            print(f'test-loss {loss2}')
'''
        