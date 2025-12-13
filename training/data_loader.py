import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class StegoDataset(Dataset):
    """
    מחלקה שמנהלת את טעינת התמונות לאימון.
    היא יודעת לקחת תמונות מתיקיית ה-Cover ומתיקיית ה-Stego
    ולערבב אותן לדאטה-סט אחד גדול.
    """

    def __init__(self, cover_dir, stego_dir, transform=None):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.transform = transform

        # 1. איסוף כל הנתיבים לקבצים
        # אנו תומכים ב-PNG ו-JPG
        valid_exts = ('*.png', '*.jpg')
        self.cover_paths = []
        self.stego_paths = []

        for ext in valid_exts:
            self.cover_paths.extend(glob.glob(os.path.join(cover_dir, ext)))
            self.stego_paths.extend(glob.glob(os.path.join(stego_dir, ext)))

        self.data = []

        for p in self.cover_paths:
            self.data.append((p, 0))

        for p in self.stego_paths:
            self.data.append((p, 1))

        print(f"Dataset initialized: {len(self.cover_paths)} Cover, {len(self.stego_paths)} Stego images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        try:
            #convert to gray scale
            image = Image.open(path).convert('L')

            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            return image, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            # return a black photo so training wouldnt get stuck
            return torch.zeros(1, 256, 256), torch.tensor(label, dtype=torch.long)