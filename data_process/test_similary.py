import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# 加载mBERT模型和分词器（使用您本地的模型路径）
tokenizer_model_dir = '/root/cxt/construct_graph/my-bert-base-multilingual-uncased'
bert_model_dir = '/root/cxt/construct_graph/my-bert-base-multilingual-uncased'

tokenizer = BertTokenizer.from_pretrained(tokenizer_model_dir)
model = BertModel.from_pretrained(bert_model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_mbert_embedding(text):
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=512,
        add_special_tokens=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 关键修改：保持batch维度
    last_hidden = outputs.last_hidden_state  # 形状 [1, seq_len, 768]
    mean_embedding = torch.mean(last_hidden, dim=1)  # 沿序列维度平均，形状 [1, 768]
    return mean_embedding.cpu().numpy()

def calculate_similarity(text1, text2):
    embedding1 = get_mbert_embedding(text1)  # 形状 [1, 768]
    embedding2 = get_mbert_embedding(text2)  # 形状 [1, 768]
    
    # 现在输入是2D数组
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

# 示例使用（支持不同语言）
english_text = "菲律宾台风灾害"
chinese_text = "ThảmHọaPhilippines"

# 计算英-中相似度
en_zh_sim = calculate_similarity(english_text, chinese_text)
print(f"English-Chinese similarity: {en_zh_sim:.4f}")
