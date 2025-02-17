from __future__ import print_function
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.cross_vit import CrossViT
import pickle
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import time
import psutil


# print(f"Torch: {torch.__version__}")

# Training settings


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed = 42
seed_everything(seed)

# Dataset preparation --------------------------------------------------------------------------------

# Define image labels
img_label = ['lung_aca', 'lung_n', 'lung_scc']
# Define batch size
batch_size = 128

# Path to image directory
img_path = "F:/histopathology_dataset/LC2500_dataset/lung_colon_image_set/lung_image_sets/"

# # Load image paths and labels into DataFrame
# img_list = []
# label_list = []
# for label in img_label:
#     for img_file in os.listdir(os.path.join(img_path, label)):
#         img_list.append(os.path.join(img_path, label, img_file))
#         label_list.append(label)
#
# df = pd.DataFrame({'img': img_list, 'label': label_list})
#
# # # Create label mapping
# df_labels = {
#     'lung_aca': 1,
#     'lung_n': 0,
#     'lung_scc': 2,
# }
#
# # Encode labels
# df['encode_label'] = df['label'].map(df_labels)
#
#
# # Define dataset class
# class CustomDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = df
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         img_path = self.df.iloc[idx]['img']
#         image = cv2.imread(img_path)
#         image = cv2.resize(image, (128, 128))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = image.astype(np.float32) / 255.0
#         label = self.df.iloc[idx]['encode_label']
#         if self.transform:
#             image = self.transform(image)
#         return image, label
#
# # Define transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
#
# # Create datasets and dataloaders
# dataset = CustomDataset(df, transform=transform)
# train_data, val_test_data = train_test_split(dataset, test_size=0.1, shuffle=True)
# val_data, test_data = train_test_split(val_test_data, test_size=0.5, shuffle=True)
#
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define filenames for saving
train_filename = "../histopathology_dataset/pkl/lung cancer/train_data.pkl"
val_filename = "../histopathology_dataset/pkl/lung cancer/val_data.pkl"
test_filename = "../histopathology_dataset/pkl/lung cancer/test_data.pkl"

# # Save the datasets
# with open(train_filename, 'wb') as f:
#     pickle.dump(train_data, f)
#
# with open(val_filename, 'wb') as f:
#     pickle.dump(val_data, f)
#
# with open(test_filename, 'wb') as f:
#     pickle.dump(test_data, f)

# Load the datasets
with open(train_filename, 'rb') as f:
    train_data = pickle.load(f)

with open(val_filename, 'rb') as f:
    val_data = pickle.load(f)

with open(test_filename, 'rb') as f:
    test_data = pickle.load(f)

# Recreate the loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Model building --------------------------------------------------------------------------------

device = 'cuda'
lr = 3e-4

model = CrossViT(
    image_size = 128,
    num_classes = 3,
    depth = 2,               # number of multi-scale encoding blocks
    sm_dim = 32,            # high res dimension
    sm_patch_size = 8,      # high res patch size (should be smaller than lg_patch_size)
    sm_enc_depth = 1,        # high res depth
    sm_enc_heads = 4,        # high res heads
    sm_enc_mlp_dim = 32,   # high res feedforward dimension
    lg_dim = 64,            # low res dimension
    lg_patch_size = 16,      # low res patch size
    lg_enc_depth = 1,        # low res depth
    lg_enc_heads = 4,        # low res heads
    lg_enc_mlp_dim = 32,   # low res feedforward dimensions
    cross_attn_depth = 1,    # cross attention rounds
    cross_attn_heads = 4,    # cross attention heads
    dropout = 0.2,
    emb_dropout = 0.2
).to(device)

# print(model)
summary(model, (3, 128, 128), device=device)
# 计算需训练参数的总数
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters in the model: {total_trainable_params}")

# Model training --------------------------------------------------------------------------------

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
epochs = 200
train_losses = []
val_losses = []
train_accs = []
val_accs = []


# 在训练之前记录时间
start_train_time = time.time()
for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 推理阶段
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

torch.save(model.state_dict(), './result/train_new_cross_vit.pth')

# 保存训练和推理时间到 DataFrame
train_losses = pd.DataFrame(train_losses)
train_accs = pd.DataFrame(train_accs)
val_losses = pd.DataFrame(val_losses)
val_accs = pd.DataFrame(val_accs)

merged_df = pd.concat([train_losses, train_accs, val_losses, val_accs], axis=1)
merged_df.columns = ['train_losses', 'train_accs', 'val_losses', 'val_accs']
merged_df = merged_df.reset_index(drop=False).rename(columns={'index': 'epochs'})


# 保存合并后的 DataFrame 到 Excel
with pd.ExcelWriter('./result/history.xlsx') as writer:
    merged_df.to_excel(writer, sheet_name='TrainingHistory', index=False)

# Plotting
plt.figure(figsize=(12, 6))

# Loss curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o', label='Train Loss')
plt.plot(val_losses, marker='o', label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(train_accs, marker='o', label='Train Accuracy')
plt.plot(val_accs, marker='o', label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./result/history.jpg', dpi=1000, bbox_inches='tight')
# plt.show()

# Model test --------------------------------------------------------------------------------

# Evaluate the model
# 加载本地模型文件
model.load_state_dict(torch.load('./result/train_new_cross_vit.pth'))  # 导入网络的参数
model.eval()

correct = 0
total = 0
all_outputs = []
true_labels = []
inference_times = []  # 记录每个epoch的推理时间
inference_memories = []  # 记录每个epoch的内存使用情况

inference_start_time = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        true_labels.append(labels.cpu().numpy())
        outputs = model(images)
        outputs = F.softmax(outputs, dim=1)
        all_outputs.append(outputs.cpu().numpy())

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print('Test Accuracy: {:.2f}%'.format(100 * accuracy))

true_labels = np.concatenate(true_labels, axis=0)
pd.DataFrame(true_labels).to_excel('./result/output.xlsx', sheet_name='test_true_labels', index=False)
all_outputs = pd.DataFrame(np.concatenate(all_outputs), columns=img_label)

# 写入Excel文件
with pd.ExcelWriter('./result/output.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    all_outputs.to_excel(writer, index=False, sheet_name='test_Predicted Probability')

# # Model val --------------------------------------------------------------------------------
#
# # Evaluate the model
# # 加载本地模型文件
# model.load_state_dict(torch.load('./result_train_new_cross_vit/6_3-7_soft segmentation + fixed position coding + skip connection/train_new_cross_vit.pth'))  # 导入网络的参数
# model.eval()
#
# correct = 0
# total = 0
# all_outputs = []
# true_labels = []
# with torch.no_grad():
#     for images, labels in val_loader:
#         images, labels = images.to(device), labels.to(device)
#         true_labels.append(labels.cpu().numpy())
#         outputs = model(images)
#         outputs = F.softmax(outputs, dim=1)
#         all_outputs.append(outputs.cpu().numpy())
#
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# accuracy = correct / total
# print('Val Accuracy: {:.2f}%'.format(100 * accuracy))
#
# true_labels = np.concatenate(true_labels, axis=0)
# all_outputs = pd.DataFrame(np.concatenate(all_outputs), columns=img_label)
# with pd.ExcelWriter('./result_train_new_cross_vit/6_3-7_soft segmentation + fixed position coding + skip connection/output.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
#     pd.DataFrame(true_labels).to_excel(writer, index=False, sheet_name='val_true_labels')
#     all_outputs.to_excel(writer, index=False, sheet_name='val_Predicted Probability')
#
# correct = 0
# total = 0
# all_outputs = []
# true_labels = []
# with torch.no_grad():
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         true_labels.append(labels.cpu().numpy())
#         outputs = model(images)
#         outputs = F.softmax(outputs, dim=1)
#         all_outputs.append(outputs.cpu().numpy())
#
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# accuracy = correct / total
# print('Train Accuracy: {:.2f}%'.format(100 * accuracy))
#
# true_labels = np.concatenate(true_labels, axis=0)
# all_outputs = pd.DataFrame(np.concatenate(all_outputs), columns=img_label)
# with pd.ExcelWriter('./result_train_new_cross_vit/6_3-7_soft segmentation + fixed position coding + skip connection/output.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
#     pd.DataFrame(true_labels).to_excel(writer, index=False, sheet_name='train_true_labels')
#     all_outputs.to_excel(writer, index=False, sheet_name='train_Predicted Probability')



