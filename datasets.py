import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
from utils import get_file_labels_boxes

# Заменим датасет на датасет с картами
class PlayingCardsDataset(Dataset):
    """
    Класс-датасет для задачи детекции игральных карт.
    """
    
    def __init__(self, images_path, data_path, input_size=(160, 240)):

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Решайте сами, когда и зачем вам нужна нормализация
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # константы из ImageNet
                std=[0.229, 0.224, 0.225]
            )
        ])

        images, boxes, labels = self._read_images_boxes_labels(images_path, data_path)
        self.images = images
        self.boxes = boxes
        self.labels = labels


    def _read_images_boxes_labels(self, images_path, data_path):
        """
        Загружаем все изображения, а также соотвествующие им боксы и лейблы.
        """
        data = pd.read_csv(data_path)

        images = []
        boxes = []
        labels = []

        for file in os.listdir(images_path):
            image =  self.transform(Image.open(os.path.join(images_path, file)))
            labels_, boxes_ = get_file_labels_boxes(data, file)
            images.append(image)
            labels.append(torch.tensor(labels_))
            # Все координаты боксов делим на 300,
            # потому что в SSD используются относительные координаты.
            boxes.append(torch.tensor(boxes_) / 300)
        
        return images, boxes, labels

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.boxes[idx], self.labels[idx]

    def collate_fn(self, batch):
        """
        Эта функция поясняет как кобинировать данные в батч. 
        Для этого мы будем использовать списки.
        Аргумент:
          batch: итерируемый объект из __getitem__()
        Возвращает:
          тензор картинок
        """

        images = []
        boxes = []
        labels = []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images)

        return images, boxes, labels