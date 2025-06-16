# 标准化计算流程
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

tokenizer_model_dir = '/root/cxt/construct_graph/my-bert-base-multilingual-uncased'
bert_model_dir = '/root/cxt/construct_graph/my-bert-base-multilingual-uncased'

# 加载配置时启用隐藏状态输出
config = AutoConfig.from_pretrained(bert_model_dir, output_hidden_states=True)

# 初始化模型时注入配置
model = AutoModel.from_pretrained(
    bert_model_dir,
    config=config,
    add_pooling_layer=False  # 关闭默认池化层
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_dir)

# model = AutoModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def get_similarity(text1, text2):
    inputs1 = tokenizer(text1, return_tensors="pt", 
                      padding=True, truncation=True, 
                      max_length=64)
    inputs2 = tokenizer(text2, return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=64)

    with torch.no_grad():
        # 调用时明确要求隐藏状态
        out1 = model(**inputs1, output_hidden_states=True)
        out2 = model(**inputs2, output_hidden_states=True)

        # 获取最后一层隐藏状态的均值向量
        hidden1 = out1.hidden_states[-1].mean(dim=1) 
        hidden2 = out2.hidden_states[-1].mean(dim=1)

    return torch.nn.functional.cosine_similarity(hidden1, hidden2).item()

# 示例计算
print(get_similarity("Red Cross teams", "Hội Chữ thập đỏ"))  # 应输出0.43±0.02