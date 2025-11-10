import json
import os


# 读取 JSON 数据
data = json.load(open("train_data_with_description.json", encoding="utf-8"))

# 写入文本文件
with open("orchid_text.txt", "w", encoding="utf-8") as f:
    for item in data["train_data"]:
        f.write(item["Description"] + "\n")
