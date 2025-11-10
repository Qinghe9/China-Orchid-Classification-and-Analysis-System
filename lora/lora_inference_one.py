#单张推理
import torch, json  
from PIL import Image
from torchvision import transforms
from vit import ViT
from peft import PeftModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 512

# 1. 加载 base 模型
label2id = json.load(open('lora_weights/label2id.json'))   
num_classes = len(label2id)                                
base_model = ViT(num_classes=num_classes, img_size=IMG_SIZE)  
ckpt = torch.load('best_model.pt', map_location='cpu')
base_model.load_state_dict(ckpt['model_state'])

# 2. 套 LoRA
model = PeftModel.from_pretrained(base_model, 'lora_weights')
model = model.merge_and_unload()
model.to(DEVICE).eval()

# 3. 预处理
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image = Image.open('惠兰.jpg').convert('RGB')
x = tf(image).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
top3 = torch.topk(probs, k=3, dim=1)
label2id = json.load(open('lora_weights/label2id.json'))
id2label = {int(v): k for k, v in label2id.items()}
for i in range(3):
    idx = top3.indices[0, i].item()
    print(f'{i+1}. {id2label[idx]}  {top3.values[0, i].item()*100:.2f}%')