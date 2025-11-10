import torch, os, json, math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Orchid2024Dataset
from vit import ViT
import matplotlib.pyplot as plt

# ---------------- 基本配置 ----------------
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT   = './Orchid2024'
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-4
PATIENCE    = 10
IMG_SIZE    = 512
NUM_WORKERS = 4

# ---------------- 数据变换 ----------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------- 数据集 ----------------
train_set = Orchid2024Dataset(DATA_ROOT, 'train', train_tf)
val_set   = Orchid2024Dataset(DATA_ROOT, 'validation', val_tf)
num_classes = len(train_set.label2id)
print(f'类别数：{num_classes}')

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ---------------- 模型 & 优化器 ----------------
model = ViT(num_classes=num_classes, img_size=IMG_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = torch.nn.CrossEntropyLoss()

best_acc = 0.0
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

# ---------------- 训练/验证一轮 ----------------
def train_one_epoch():
    model.train()
    running_loss = 0.
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    return running_loss / total

@torch.no_grad()
def validate():
    model.eval()
    running_loss = 0.
    correct = total = 0
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

# ---------------- 绘图 ----------------
def plot_history():
    plt.figure(figsize=(10, 4))
    # loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    # acc
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='val acc', color='orange')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    print('训练曲线已保存 -> training_curve.png')

# ---------------- 主循环 ----------------
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch()
    val_loss, val_acc = validate()
    lr = scheduler.get_last_lr()[0]
    scheduler.step()

    # 记录历史
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(lr)

    # 每轮一行信息
    print(f'Epoch {epoch:02d}/{EPOCHS} | '
          f'train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f} | lr: {lr:.2e}')

    # 早停 & 最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'model_state': model.state_dict(),
            'label2id': train_set.label2id,
            'num_classes': num_classes,
            'history': history
        }, 'best_model.pt')
        print(f'↑ 新最佳 acc={val_acc:.4f}，模型已保存')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print('Early stopping triggered!')
            break

print('训练完成，最终 best acc:', best_acc)
plot_history()