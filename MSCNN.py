import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import os
import numpy as np
import json
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 数据预处理
def preprocess(frame):
    frame = cv2.resize(frame, (112, 112))
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        ToTensorV2()
    ])
    augmented = transform(image=frame)
    return augmented['image']

# 统一帧数
def uniform_frames(frames, target_frames=120):
    num_frames = frames.shape[0]
    if num_frames > target_frames:
        start = random.randint(0, num_frames - target_frames)
        frames = frames[start:start + target_frames]
    elif num_frames < target_frames:
        frames = frames.repeat((target_frames // num_frames) + 1, 1, 1, 1)
        frames = frames[:target_frames]
    return frames

class VideoDataset(Dataset):
    def __init__(self, data_info, data_dir, transform=None):
        self.data_info = data_info
        self.data_dir = data_dir
        self.transform = transform
        self.videos = []
        self.labels = []
        for video_name, info in data_info.items():
            video_path = os.path.join(data_dir, video_name)
            if os.path.exists(video_path):
                label = 1 if info["label"] == "FAKE" else 0
                self.videos.append(video_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            else:
                break
        cap.release()
        frames = torch.stack(frames)
        frames = uniform_frames(frames)
        return frames, label


# MSCNN 模型
class ImprovedMSCNN(nn.Module):
    def __init__(self):
        super(ImprovedMSCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)
        x = self.conv_block(x)
        x = self.global_avg_pool(x).view(batch_size, num_frames, -1)
        x = x.mean(dim=1)  # 时间维度平均
        x = self.classifier(x)
        return x
# 训练
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device).float()
            labels = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')


# 生成 ROC 曲线
def test_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_scores = []
    all_preds = []
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device).float()
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(frames)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_scores.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            all_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten())

    print("\nClassification Report:")
    if all_labels and all_preds:
        print(classification_report(all_labels, all_preds))
    else:
        print("Warning: No labels or predictions available for classification report.")

    if all_labels and all_scores:
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        return roc_auc
    else:
        print("Warning: No labels or scores available for ROC curve calculation.")
        return None


# 主函数
if __name__ == "__main__":
    # 加载数据集信息
    json_file_path = "D:\\fakenews\\metadata.json"
    with open(json_file_path, 'r') as f:
        data_info = json.load(f)

    data_dir = "D:\\fakenews"

    full_dataset = VideoDataset(data_info, data_dir, transform=preprocess)

    # 划分训练集和测试集
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 参数
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedMSCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()  # 使用带 logits 的 BCE 损失
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    epochs = 3
    best_auc = 0.0

    for epoch in range(epochs):
        train_model(model, train_loader, criterion, optimizer, device, 1)
        current_auc = test_model(model, test_loader, device)

        if current_auc is not None and current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model with AUC: {best_auc:.4f}')

        scheduler.step(current_auc)