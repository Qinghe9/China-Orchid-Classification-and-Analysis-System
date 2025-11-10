
import os, math, json, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Orchid2024Dataset
from vit import ViT
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ---------------- 基本配置 ----------------
DATA_ROOT   = './Orchid2024'
PRETRAIN_PT = 'best_model.pt'      # 原 best_model.pt 路径
BATCH_SIZE  = 32
EPOCHS      = 20                   # 少量 epoch 即可
LR          = 3e-4
PATIENCE    = 5                   # 耐心值
IMG_SIZE    = 512
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
LORA_RANK   = 16
LORA_ALPHA  = 32
LORA_DROPOUT= 0.05
SAVE_DIR    = 'lora_weights'       # 仅保存 LoRA 参数
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- 数据 ----------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_set = Orchid2024Dataset(DATA_ROOT, 'train', train_tf)
val_set   = Orchid2024Dataset(DATA_ROOT, 'validation', val_tf)
num_classes = len(train_set.label2id)
'''
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)
'''
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=True)  # 修改这里
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)  # 修改这里

# ---------------- 加载原模型 ----------------
print('>>> 加载原模型…')
ckpt = torch.load(PRETRAIN_PT, map_location='cpu')
base_model = ViT(num_classes=num_classes, img_size=IMG_SIZE)
base_model.load_state_dict(ckpt['model_state'])
base_model = base_model.to(DEVICE)

# ---------------- 加 LoRA ----------------
# 只对 ViT 中的 qkv proj 与 MLP 线性层插入 LoRA
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=['patch_emb', 'cls_linear'],   # patch_emb 即 Linear(patch²→emb), cls_linear 即分类头
    lora_dropout=LORA_DROPOUT,
    bias='none',
    task_type=TaskType.FEATURE_EXTRACTION         # 纯特征提取，无 seq2seq
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()               # 查看可训参数量
model.to(DEVICE)

# ---------------- 优化器 ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ---------------- 训练/验证函数 ----------------
def train_one_epoch():
    model.train()
    total, loss_sum = 0, 0.
    for imgs, lbs in train_loader:
        imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, lbs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        total += imgs.size(0)
    return loss_sum / total

@torch.no_grad()
def validate():
    model.eval()
    total, loss_sum, correct = 0, 0., 0
    for imgs, lbs in val_loader:
        imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, lbs)
        loss_sum += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == lbs).sum().item()
        total += imgs.size(0)
    return loss_sum / total, correct / total

# ---------------- 主循环 ----------------
best_acc = 0.
patience = 0
for epoch in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch()
    val_loss, val_acc = validate()
    scheduler.step()
    print(f'Epoch {epoch:02d}/{EPOCHS} | '
        f'train loss: {tr_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}', flush=True)

    if val_acc > best_acc:
        best_acc = val_acc
        patience = 0
        # 保存 LoRA 权重（不含 base）
        model.save_pretrained(SAVE_DIR)
        # 同时把 label2id 写进去，方便推理
        json.dump(ckpt['label2id'], open(os.path.join(SAVE_DIR, 'label2id.json'), 'w'))
        print(f'↑ 最佳 acc={val_acc:.4f}，LoRA 权重已保存至 {SAVE_DIR}')
        print(f'Epoch {epoch:02d}/{EPOCHS} | ...', flush=True)
    else:
        patience += 1
        if patience >= PATIENCE:
            print('Early stopping!')
            break

print('>>> LoRA 微调完成！')
