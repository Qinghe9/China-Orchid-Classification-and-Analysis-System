'''
import torch
import json
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from vit import ViT

# def load_model_and_breed_info(model_path='best_model.pt', device=None):
def load_model_and_breed_info(model_path='best_model.pt', device=None):
    """加载训练好的模型和兰花品种详细信息"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型权重和配置
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请确保模型已训练并保存")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 重建模型
    model = ViT(
        num_classes=checkpoint['num_classes'],
        img_size=512,
        patch_size=16,
        emb_size=256
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # 加载品种详细信息映射表
    breed_info_map = {}
    # 尝试从训练数据中获取品种信息
    try:
        # 查找训练数据parquet文件
        candidates = [
            os.path.join('./Orchid2024', 'train.parquet'),
            os.path.join('./Orchid2024', 'train', 'train.parquet'),
            'train.parquet'
        ]
        parquet_path = next((cp for cp in candidates if os.path.isfile(cp)), None)
        
        if parquet_path:
            # 只读取需要的列
            df = pd.read_parquet(
                parquet_path, 
                columns=[
                    'Label', 'Cultivar_Name', 'Species_Name', 
                    'Chinese_Cultivar_Name', 'Chinese_Species_Name'
                ]
            )
            # 去重并创建映射
            breed_info_map = df.drop_duplicates('Label').set_index('Label').to_dict('index')
            print(f"成功加载 {len(breed_info_map)} 条品种信息")
        else:
            print("未找到训练数据，无法加载完整品种信息")
            
    except Exception as e:
        print(f"加载品种信息时出错: {e}")
    
    return model, device, breed_info_map, checkpoint['label2id']

def predict(image_path, model, device, breed_info_map, label2id):
    """对单个图片进行预测，返回完整品种信息"""
    # 定义与训练时相同的预处理
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 执行推理
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            top_prob, top_idx = torch.topk(probabilities, k=3)
        
        # 整理结果，包含所有需要显示的字段
        results = []
        # 反转label2id以通过id查找原始Label值
        id2label = {v: k for k, v in label2id.items()}
        
        for i in range(top_idx.size(1)):
            class_id = top_idx[0][i].item()
            prob = top_prob[0][i].item()
            
            # 获取原始Label值
            original_label = id2label.get(class_id, f"未知标签_{class_id}")
            
            # 获取品种详细信息
            breed_info = breed_info_map.get(original_label, {})
            
            # 构建结果字典，包含所有需要显示的字段
            result = {
                'ranking': i + 1,  # 排名
                'label': original_label,  # Label (Value)
                'cultivar_name': breed_info.get('Cultivar_Name', 'N/A'),  # Cultivar_Name (Value)
                'species_name': breed_info.get('Species_Name', 'N/A'),  # Species_Name (Value)
                'chinese_cultivar_name': breed_info.get('Chinese_Cultivar_Name', 'N/A'),  # 中文栽培品种名
                'chinese_species_name': breed_info.get('Chinese_Species_Name', 'N/A'),  # 中文物种名
                'probability': round(prob * 100, 2)  # 置信度
            }
            
            results.append(result)
        
        return results, image
    
    except Exception as e:
        print(f"预测失败: {e}")
        return None, None

if __name__ == "__main__":
    # 命令行测试模式
    import sys
    if len(sys.argv) != 2:
        print("用法: python 推理代码.py <图片路径>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"错误: 图片文件 {image_path} 不存在")
        sys.exit(1)
    
    try:
        model, device, breed_info_map, label2id = load_model_and_breed_info()
        results, _ = predict(image_path, model, device, breed_info_map, label2id)
        
        if results:
            print("兰花品种识别结果:")
            for res in results:
                print(f"\n排名: {res['ranking']} (置信度: {res['probability']}%)")
                print(f"Label: {res['label']}")
                print(f"品种名称: {res['cultivar_name']}")
                print(f"物种名称: {res['species_name']}")
                print(f"中文品种名称: {res['chinese_cultivar_name']}")
                print(f"中文物种名称: {res['chinese_species_name']}")
        else:
            print("识别失败，请尝试其他图片")
            
    except Exception as e:
        print(f"执行出错: {e}")
        '''





# import torch
# import json
# import os
# import pandas as pd
# from PIL import Image
# from torchvision import transforms
# from vit import ViT
# from peft import PeftModel

# def load_model_and_breed_info(base_model_path='best_model.pt', lora_path='lora_weights', device=None):
#     """加载基础模型和LoRA权重，以及兰花品种详细信息"""
#     if device is None:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 加载基础模型
#     if not os.path.exists(base_model_path):
#         raise FileNotFoundError(f"基础模型文件 {base_model_path} 不存在")
    
#     checkpoint = torch.load(base_model_path, map_location=device)
    
#     # 重建基础模型
#     base_model = ViT(
#         num_classes=checkpoint['num_classes'],
#         img_size=512,
#         patch_size=16,
#         emb_size=256
#     ).to(device)
    
#     base_model.load_state_dict(checkpoint['model_state'])
    
#     # 加载LoRA权重
#     if not os.path.exists(lora_path):
#         raise FileNotFoundError(f"LoRA权重目录 {lora_path} 不存在")
    
#     model = PeftModel.from_pretrained(base_model, lora_path)
#     model.eval()
    
#     # 加载LoRA保存的label2id
#     lora_label2id_path = os.path.join(lora_path, 'label2id.json')
#     if os.path.exists(lora_label2id_path):
#         with open(lora_label2id_path, 'r') as f:
#             label2id = json.load(f)
#     else:
#         label2id = checkpoint['label2id']
    
#     # 加载品种详细信息映射表
#     breed_info_map = {}
#     try:
#         # 查找训练数据parquet文件
#         candidates = [
#             os.path.join('./Orchid2024', 'train.parquet'),
#             os.path.join('./Orchid2024', 'train', 'train.parquet'),
#             'train.parquet'
#         ]
#         parquet_path = next((cp for cp in candidates if os.path.isfile(cp)), None)
        
#         if parquet_path:
#             # 只读取需要的列
#             df = pd.read_parquet(
#                 parquet_path, 
#                 columns=[
#                     'Label', 'Cultivar_Name', 'Species_Name', 
#                     'Chinese_Cultivar_Name', 'Chinese_Species_Name'
#                 ]
#             )
#             # 去重并创建映射
#             breed_info_map = df.drop_duplicates('Label').set_index('Label').to_dict('index')
#             print(f"成功加载 {len(breed_info_map)} 条品种信息")
#         else:
#             print("未找到训练数据，无法加载完整品种信息")
            
#     except Exception as e:
#         print(f"加载品种信息时出错: {e}")
    
#     return model, device, breed_info_map, label2id

# def predict(image_path, model, device, breed_info_map, label2id):
#     """对单个图片进行预测，返回完整品种信息"""
#     # 修正：Resize参数应该是元组 (512, 512)
#     transform = transforms.Compose([
#         transforms.Resize((512, 512)),  # 修正这里
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
#     try:
#         # 加载并预处理图片
#         image = Image.open(image_path).convert('RGB')
#         image_tensor = transform(image).unsqueeze(0).to(device)
        
#         # 执行推理
#         with torch.no_grad():
#             logits = model(image_tensor)
#             probabilities = torch.nn.functional.softmax(logits, dim=1)
#             top_prob, top_idx = torch.topk(probabilities, k=3)
        
#         # 整理结果，包含所有需要显示的字段
#         results = []
#         # 反转label2id以通过id查找原始Label值
#         id2label = {v: k for k, v in label2id.items()}
        
#         for i in range(3):
#             class_id = top_idx[0][i].item()
#             prob = top_prob[0][i].item()
            
#             # 获取原始Label值
#             original_label = id2label.get(class_id, f"未知标签_{class_id}")
            
#             # 获取品种详细信息
#             breed_info = breed_info_map.get(original_label, {})
            
#             # 构建结果字典，包含所有需要显示的字段
#             result = {
#                 'ranking': i + 1,  # 排名
#                 'label': original_label,  # Label (Value)
#                 'cultivar_name': breed_info.get('Cultivar_Name', 'N/A'),  # Cultivar_Name (Value)
#                 'species_name': breed_info.get('Species_Name', 'N/A'),  # Species_Name (Value)
#                 'chinese_cultivar_name': breed_info.get('Chinese_Cultivar_Name', 'N/A'),  # 中文栽培品种名
#                 'chinese_species_name': breed_info.get('Chinese_Species_Name', 'N/A'),  # 中文物种名
#                 'probability': round(prob * 100, 2)  # 置信度
#             }
            
#             results.append(result)
        
#         return results, image
    
#     except Exception as e:
#         print(f"预测失败: {e}")
#         return None, None

# if __name__ == "__main__":
#     # 命令行测试模式
#     import sys
#     if len(sys.argv) != 2:
#         print("用法: python inference.py <图片路径>")
#         sys.exit(1)
    
#     image_path = sys.argv[1]
#     if not os.path.exists(image_path):
#         print(f"错误: 图片文件 {image_path} 不存在")
#         sys.exit(1)
    
#     try:
#         # 加载基础模型和LoRA权重
#         model, device, breed_info_map, label2id = load_model_and_breed_info(
#             base_model_path='best_model.pt',  # 基础模型路径
#             lora_path='lora_weights'          # LoRA权重路径
#         )
#         results, _ = predict(image_path, model, device, breed_info_map, label2id)
        
#         if results:
#             print("兰花品种识别结果:")
#             for res in results:
#                 print(f"\n排名: {res['ranking']} (置信度: {res['probability']}%)")
#                 print(f"Label: {res['label']}")
#                 print(f"品种名称: {res['cultivar_name']}")
#                 print(f"物种名称: {res['species_name']}")
#                 print(f"中文品种名称: {res['chinese_cultivar_name']}")
#                 print(f"中文物种名称: {res['chinese_species_name']}")
#         else:
#             print("识别失败，请尝试其他图片")
            
#     except Exception as e:
#         print(f"执行出错: {e}")

import torch
import json
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from vit import ViT
from peft import PeftModel
from GPT.gpt_model import NanoGPT, GPTConfig  

# ------------------ 加载 ViT + LoRA ------------------
def load_model_and_breed_info(base_model_path='best_model.pt', lora_path='lora_weights', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"基础模型文件 {base_model_path} 不存在")

    checkpoint = torch.load(base_model_path, map_location=device)
    base_model = ViT(
        num_classes=checkpoint['num_classes'],
        img_size=512,
        patch_size=16,
        emb_size=256
    ).to(device)
    base_model.load_state_dict(checkpoint['model_state'])

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA权重目录 {lora_path} 不存在")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    # label2id
    lora_label2id_path = os.path.join(lora_path, 'label2id.json')
    if os.path.exists(lora_label2id_path):
        with open(lora_label2id_path, 'r') as f:
            label2id = json.load(f)
    else:
        label2id = checkpoint['label2id']

    # 加载品种映射表
    breed_info_map = {}
    try:
        parquet_candidates = [
            os.path.join('./Orchid2024', 'train.parquet'),
            os.path.join('./Orchid2024', 'train', 'train.parquet'),
            'train.parquet'
        ]
        parquet_path = next((p for p in parquet_candidates if os.path.isfile(p)), None)
        if parquet_path:
            df = pd.read_parquet(
                parquet_path,
                columns=['Label', 'Cultivar_Name', 'Species_Name', 
                         'Chinese_Cultivar_Name', 'Chinese_Species_Name']
            )
            breed_info_map = df.drop_duplicates('Label').set_index('Label').to_dict('index')
            print(f"✅ 成功加载 {len(breed_info_map)} 条品种信息")
        else:
            print("⚠️ 未找到训练数据文件，部分信息可能缺失。")
    except Exception as e:
        print(f"加载品种信息出错: {e}")

    return model, device, breed_info_map, label2id

# ------------------ 加载 NanoGPT 模型 ------------------
def load_gpt_model(gpt_path="GPT/output/nanogpt_lan_final.pth", device='cpu'):
    if not os.path.exists(gpt_path):
        print(f"⚠️ 未找到 GPT 模型文件 {gpt_path}，将跳过描述生成。")
        return None, None, None

    checkpoint = torch.load(gpt_path, map_location=device)
    itos = checkpoint["itos"]
    stoi = checkpoint["stoi"]
    vocab_size = len(itos)

    # 尝试从 checkpoint 获取训练时配置
    train_config = checkpoint.get("config", {})
    n_layer = train_config.get("n_layer", 6)
    n_head = train_config.get("n_head", 8)
    n_embd = train_config.get("n_embd", 256)
    block_size = train_config.get("block_size", 128)

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )
    model = NanoGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    def encode(s):
        return torch.tensor([[stoi[c] for c in s if c in stoi]], dtype=torch.long).to(device)

    def decode(tensor):
        return "".join([itos[i] for i in tensor[0].tolist()])

    return model, encode, decode

# ------------------ 生成描述 ------------------
@torch.no_grad()
def generate_description(gpt_model, encode, decode, input_text, max_new_tokens=100):
    if gpt_model is None:
        return "（未加载GPT模型）"
    try:
        x = encode(input_text)
        y = gpt_model.generate(x, max_new_tokens=max_new_tokens)
        output = decode(y)
        return output[len(input_text):].strip()
    except Exception as e:
        return f"生成失败: {e}"

# ------------------ ViT 推理 ------------------
def predict(image_path, model, device, breed_info_map, label2id, gpt_model=None, encode=None, decode=None):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_prob, top_idx = torch.topk(probs, k=3)

    id2label = {v: k for k, v in label2id.items()}
    results = []

    for i in range(3):
        class_id = top_idx[0][i].item()
        prob = top_prob[0][i].item()
        label = id2label.get(class_id, str(class_id))
        info = breed_info_map.get(label, {})
        cn_species = info.get('Chinese_Species_Name', info.get('Species_Name', '未知种类'))
        cn_cultivar = info.get('Chinese_Cultivar_Name', info.get('Cultivar_Name', '未知品种'))

        desc = generate_description(gpt_model, encode, decode, f"{cn_species}{cn_cultivar}") if gpt_model else "（未加载GPT模型）"

        results.append({
            "ranking": i + 1,
            "label": label,
            "chinese_species_name": cn_species,
            "chinese_cultivar_name": cn_cultivar,
            "probability": round(prob * 100, 2),
            "description": desc
        })
    #print(f"预测结果: {results}") 
    return results

# ------------------ 主函数 ------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python inference.py <图片路径>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        sys.exit(1)

    # 加载 ViT + GPT 模型
    vit_model, device, breed_info, label2id = load_model_and_breed_info(
        base_model_path='best_model.pt',
        lora_path='lora_weights'
    )
    gpt_model, encode, decode = load_gpt_model("GPT/output/nanogpt_lan_final.pth", device=device)

    # 执行推理
    results = predict(image_path, vit_model, device, breed_info, label2id, gpt_model, encode, decode)

    print("\n🌸 兰花识别与描述生成 🌸")
    for res in results:
        print(f"\n排名: {res['ranking']} (置信度: {res['probability']}%)")
        print(f"物种名: {res['chinese_species_name']}")
        print(f"品种名: {res['chinese_cultivar_name']}")
        print(f"描述: {res['description']}")
