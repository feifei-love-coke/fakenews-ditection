import os
import cv2
import ffmpeg
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoDataset(Dataset):
    def __init__(self, json_file, num_frames=16, max_audio_len=None):
        # 读取文件
        with open(json_file, 'r') as f:
            self.data_dict = json.load(f)
        # 获取JSON文件地址
        self.root_dir = os.path.dirname(os.path.abspath(json_file))
        self.video_files = list(self.data_dict.keys())
        self.num_frames = num_frames
        self.max_audio_len = max_audio_len
        if self.max_audio_len is None:
            self._calculate_max_audio_len()

    def _calculate_max_audio_len(self):
        max_len = 0
        for idx in range(len(self.video_files)):
            video_file = self.video_files[idx]
            video_path = os.path.join(self.root_dir, video_file)
            try:
                out, _ = (
                    ffmpeg
                   .input(video_path)
                   .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='44100')
                   .run(capture_stdout=True, capture_stderr=True)
                )
                audio = np.frombuffer(out, np.int16).astype(np.float32)
                audio = audio / 32768.0
                audio_features = librosa.feature.mfcc(y=audio, sr=44100)
                max_len = max(max_len, audio_features.shape[1])
            except ffmpeg.Error as e:
                print(f"Error processing audio for {video_path}: {e.stderr.decode()}")
        self.max_audio_len = max_len

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.root_dir, video_file)
        label = 1 if self.data_dict[video_file]['label'] == 'FAKE' else 0

        # 处理图像 用cv2处理
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.transpose(2, 0, 1) / 255.0
                frames.append(frame)
            else:
                break
        cap.release()
        frames = np.array(frames)

        # 帧数改成一样的
        if len(frames) > self.num_frames:
            frames = frames[:self.num_frames]
        elif len(frames) < self.num_frames:
            padding = np.zeros((self.num_frames - len(frames), 3, 224, 224))
            frames = np.concatenate((frames, padding), axis=0)

        # 处理音频
        try:
            # ffmpeg-python一定要下载
            out, _ = (
                ffmpeg
               .input(video_path)
               .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='44100')
               .run(capture_stdout=True, capture_stderr=True)
            )
            audio = np.frombuffer(out, np.int16).astype(np.float32)
            audio = audio / 32768.0  # 归一化
            audio_features = librosa.feature.mfcc(y=audio, sr=44100)
        except ffmpeg.Error as e:
            print(f"Error processing audio for {video_path}: {e.stderr.decode()}")
            audio_features = np.zeros((20, 1))  # 如果处理失败，返回零矩阵

        # 音频特征长度变成一样长
        if audio_features.shape[1] > self.max_audio_len:
            audio_features = audio_features[:, :self.max_audio_len]
        elif audio_features.shape[1] < self.max_audio_len:
            padding = np.zeros((20, self.max_audio_len - audio_features.shape[1]))
            audio_features = np.concatenate((audio_features, padding), axis=1)

        return torch.tensor(frames, dtype=torch.float32), torch.tensor(audio_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# 模型
class VideoFakeDetector(nn.Module):
    def __init__(self):
        super(VideoFakeDetector, self).__init__()

        self.cnn = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.audio_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 先不定义第一个全连接层
        self.fc_layers = None

    def forward(self, frames, audio_features):
        batch_size, seq_len, c, h, w = frames.shape
        frames = frames.view(-1, c, h, w)
        image_features = self.cnn(frames)
        image_features = image_features.view(batch_size, -1)

        audio_features = audio_features.unsqueeze(1)
        audio_features = self.audio_conv(audio_features)
        audio_features = audio_features.view(audio_features.size(0), -1)

        combined_features = torch.cat((image_features, audio_features), dim=1)

        # 第一次前向传播时，确定输入维度并创建全连接层
        if self.fc_layers is None:
            input_dim = combined_features.size(1)
            self.fc_layers = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ).to(device)  # 将全连接层移动到GPU

        output = self.fc_layers(combined_features)
        return output


if __name__ == "__main__":
    # 参数
    json_file = "D:\\fakenews\\metadata.json" # 文件
    batch_size = 4 # 大小
    num_epochs = 4 # 次数
    num_frames = 8 # 视频帧数

    # 创建数据集和数据加载器
    dataset = VideoDataset(json_file, num_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型并移动到GPU
    model = VideoFakeDetector().to(device)

    # 损失函数
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    all_labels = []
    all_preds = []

    # 训练
    for epoch in range(num_epochs):
        running_loss = 0.0
        for frames, audio_features, labels in dataloader:
            # 将数据移动到GPU
            frames = frames.to(device)
            audio_features = audio_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames, audio_features)
            labels = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(outputs.detach().cpu().numpy().flatten())

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.4f}')

    # ROC-AUC曲线
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    # ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()