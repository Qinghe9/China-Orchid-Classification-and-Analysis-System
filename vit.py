import torch
from torch import nn

class ViT(nn.Module):
    def __init__(self, num_classes, img_size=512, patch_size=16, emb_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.patch_count = img_size // patch_size
        self.conv = nn.Conv2d(3, patch_size ** 2, patch_size, patch_size, 0)
        self.patch_emb = nn.Linear(patch_size ** 2, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_emb   = nn.Parameter(torch.randn(1, self.patch_count ** 2 + 1, emb_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=8, batch_first=True
        )
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.cls_linear = nn.Linear(emb_size, num_classes)

    # ===== 兼容 PEFT 的关键字入口 =====
    def forward(self, x=None, **kwargs):#原def forward(self, x):
        if x is None and 'input_ids' in kwargs:   # 兼容 PEFT 误传+lora
            x = kwargs.pop('input_ids')#lora
        if x is None:#lora
            raise ValueError("ViT requires input tensor x")#lora

    
        x = self.conv(x)                      # (B, 256, 32, 32)
        x = x.flatten(2).transpose(1, 2)      # (B, 1024, 256)
        x = self.patch_emb(x)                 # (B, 1024, emb_size)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, 1025, emb_size)
        x = x + self.pos_emb
        x = self.transformer_enc(x)
        return self.cls_linear(x[:, 0])       # (B, num_classes)


'''
import torch
from torch import nn

class ViT(nn.Module):
    def __init__(self, num_classes, img_size=512, patch_size=16, emb_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.patch_count = img_size // patch_size
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=patch_size ** 2,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.patch_emb = nn.Linear(patch_size ** 2, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_emb = nn.Parameter(torch.randn(1, self.patch_count ** 2 + 1, emb_size))
        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_size,
                nhead=8,
                batch_first=True
            ),
            num_layers=6
        )
        self.cls_linear = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x = self.patch_emb(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_emb + x
        x = self.transformer_enc(x)
        return self.cls_linear(x[:, 0, :])显示vit.py完整代码
        
        '''