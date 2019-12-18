
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

        sample = {'ID': self.df_idx_label_map.iloc[idx, 0], 'image': image, 'labels': labels}
        return sample
    
# image pre-processing
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
#evaluate test data for submission
#==================================================================
model = torch.load(
    './ResNet50_model_saved/resnet50_human_protein_cpu_16epochs.torch'
    )

df_test_meta = pd.read_csv('sample_submission.csv')
test_set = human_protein_Dataset(df_idx_label_map = df_test_meta,
                                 root_dir='./test',
                                 transform=image_transform)

test_loader_params = { 
            'batch_size': 24, 
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': torch.cuda.is_available(),
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

threshold = 0.05   #need to bracket threshold based on submissions
def threshold_fn(x): 
    labels = np.nonzero((all_preds[x] > threshold).astype(int))[0]
    return ' '.join(str(i) for i in labels)    
df_test_meta['Predicted'] = df_test_meta['Id'].apply(threshold_fn)  
 
df_test_meta.to_csv('submission.csv', index=False)

