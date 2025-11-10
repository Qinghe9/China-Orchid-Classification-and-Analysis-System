import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import io
import os

class Orchid2024Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data_info, self.label2id = self._load_data_info()

    # ----------------------------------------------------------
    # 从 Img.bytes 列直接读图片二进制
    # ----------------------------------------------------------
    def _load_data_info(self):
        candidates = [
            os.path.join(self.root_dir, f"{self.split}.parquet"),
            os.path.join(self.root_dir, self.split, f"{self.split}.parquet"),
        ]
        parquet_path = next((cp for cp in candidates if os.path.isfile(cp)), None)
        if parquet_path is None:
            raise FileNotFoundError(
                f"Parquet file '{self.split}.parquet' not found in any of: {candidates}"
            )

        # df = pd.read_parquet(parquet_path)         原来的
        df = pd.read_parquet(parquet_path, engine='fastparquet')  #Lora修改的
        df = df.rename(columns={'Img': 'Img.bytes'})   # <--lora 新增
        print(f"[{self.split}] 实际列名：{list(df.columns)}")

        # 必备列检查
        if 'Img.bytes' not in df.columns or 'Label' not in df.columns:
            raise ValueError("需要列 'Img.bytes' 和 'Label'！")

        # 标签 → 连续整数
        label2id = {label: idx for idx, label in enumerate(df['Label'].unique())}
        df['label_id'] = df['Label'].map(label2id)

        # 只保留非空图片
        df = df.dropna(subset=['Img.bytes'])
        data_info = list(zip(df['Img.bytes'].tolist(), df['label_id'].tolist()))
        print(f"[{self.split}] 有效样本数：{len(data_info)}")
        return data_info, label2id

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_bytes, label_id = self.data_info[idx]
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label_id