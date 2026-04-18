import os
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset

group_activity_classes = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {class_name:i for i, class_name in enumerate(group_activity_classes)}



class VolleyBallDataset(Dataset):
    def __init__(self, videos_root,annot_path,vid_indices, transform=None):

        self.videos_root = videos_root
        self.transform = transform
        self.samples = []

        with open(annot_path, 'rb') as file:
            videos_annot = pickle.load(file)

        for vid_idx in vid_indices:
            vid_idx = str(vid_idx)
            clips = videos_annot[vid_idx]

            for clip_dir,clip_data in clips.items():
                clip_label = group_activity_labels[clip_data['category']]
                image_path = os.path.join(self.videos_root, vid_idx, clip_dir,f'{clip_dir}.jpg')
                self.samples.append((image_path, clip_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label