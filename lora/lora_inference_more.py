# 多张推理
import torch
import json
import pathlib
from PIL import Image
from torchvision import transforms
from vit import ViT
from peft import PeftModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 512
IMG_EXT = ('*.jpg', '*.jpeg', '*.png', '*.bmp')   # 支持扩展名

# 1. 加载模型
label2id = json.load(open('lora_weights/label2id.json'))
num_classes = len(label2id)
base_model = ViT(num_classes=num_classes, img_size=IMG_SIZE)

ckpt = torch.load('best_model.pt', map_location='cpu')
base_model.load_state_dict(ckpt['model_state'])

model = PeftModel.from_pretrained(base_model, 'lora_weights')
model = model.merge_and_unload()
model.to(DEVICE).eval()

# 2. 预处理
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

id2label = {int(v): k for k, v in label2id.items()}

# 3. 批量推理
def infer_one(img_path: pathlib.Path):
    image = Image.open(img_path).convert('RGB')
    x = tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top3 = torch.topk(probs, k=3, dim=1)
    return [(id2label[top3.indices[0, i].item()],
             top3.values[0, i].item() * 100) for i in range(min(3, top3.indices.size(1)))]

def main(img_dir='images', out_file='lora_inference3.txt'):
    img_dir = pathlib.Path(img_dir)
    img_list = [p for ext in IMG_EXT for p in img_dir.glob(ext)]
    if not img_list:
        print(f'❌ 目录 {img_dir.absolute()} 中没有找到图片')
        return

    with open(out_file, 'w', encoding='utf-8') as f:
        for p in img_list:
            top3 = infer_one(p)
            line = f'\n>>> {p.name}\n'
            for rank, (name, score) in enumerate(top3, 1):
                line += f'{rank}. {name}  {score:.2f}%\n'
            print(line, end='')
            f.write(line)
    print(f'✅ 推理完成，结果已写入 {out_file}')

if __name__ == '__main__':
    # 默认扫描当前目录下 images 文件夹；可传参改变
    import sys
    main(img_dir=sys.argv[1] if len(sys.argv) > 1 else 'images',
         out_file=sys.argv[2] if len(sys.argv) > 2 else 'lora_inference3.txt')