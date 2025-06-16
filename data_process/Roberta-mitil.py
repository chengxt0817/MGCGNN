import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import IncrementalPCA

# 从事件串中提取事件特征
def extract_time_feature(t_str):
    a = str(t_str)
    t = datetime.fromisoformat(a)
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]

# DataFrame中的时间戳数据转换为时间特征
def df_to_t_features(df):
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    return t_features

# 加载数据
text_list = np.load('/root/cxt/construct_graph_20241125/data_feature_new.npy', allow_pickle=True)
print('读取文件成功')

data_feature = pd.DataFrame(data=text_list, columns=["event_id", "tweet_id", "user_id", "created_at", "hashtags", "user_mentions", "text", "filtered_words", "Entities", "Labels", "language", "Qids"])
print(data_feature)
data_feature['index'] = range(len(data_feature))
print("提取时间特征")
combined_t_features = df_to_t_features(data_feature)
print("Time features generated.")
print(combined_t_features)

# 确保数据为字符串列表
combined_df = data_feature['text'].tolist()

# 初始化RoBERTa分词器和模型
# vocab_file = '/root/cxt/construct_graph_20241125/RoBert_model/vocab.json'
 
# merges_file = '/root/cxt/construct_graph_20241125/RoBert_model/merges.txt'
 
tokenizer = AutoTokenizer.from_pretrained("/root/cxt/construct_graph_20241125/my-XML-RoBerta-base")
model = AutoModel.from_pretrained("/root/cxt/construct_graph_20241125/my-XML-RoBerta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 文本预处理
encoded_text = []
def encode_text(text):
    inputs = tokenizer.encode_plus(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    encoded_text = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    # encoded_text = outputs[0][:, 0, :].squeeze().cpu().numpy()
    encoded_text = np.array(encoded_text)
    print("encoded_text shape:", encoded_text.shape)  # 打印形状，确保它是二维的
    # print("我走到很慢但我在走")
    return encoded_text

for text in combined_df:
    # 检查数据是否为字符串
    encoded_text.append(encode_text(str(text)))
    print("我走的很慢，但我再走")

# 转换为 numpy 数组
encoded_text = np.array(encoded_text)
print("encoded_text shape before reduction:", encoded_text.shape)  # 打印形状，确保它是二维的

# 使用 PCA 将文本特征从 768 维降维到 300 维（如果需要降维）

pca = PCA(n_components=300)
encoded_text_reduced = pca.fit_transform(encoded_text)
# ipca = IncrementalPCA(n_components=300, batch_size=1000)
# encoded_text_reduced = ipca.fit_transform(encoded_text)
print("encoded_text shape after reduction:", encoded_text_reduced.shape)  # 应该是 (num_samples, 300)

print("结合两个特征")
combined_features = np.concatenate((encoded_text_reduced, combined_t_features), axis=1)
combined_df = pd.DataFrame(combined_features)
print("combined_features shape:", combined_df.shape)  # 应该是 (num_samples, 302)

# indices = data_feature['index'].values.tolist()
# x = combined_df.iloc[indices, :]  # 使用 .iloc 方法访问行
# y = data_feature['event_id'].values
# y = [int(each) for each in y]
# np.save('/root/cxt/construct_graph_20241125/Word2Vec_pic/labels.npy', np.asarray(y))

np.save('/root/cxt/construct_graph_20241125/dataset/302dim_RoBERTa-mutil.npy', combined_df)