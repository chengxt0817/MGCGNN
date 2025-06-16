import numpy as np
import jieba
from gensim.models import Word2Vec
import torch
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import torch
import spacy
from transformers import BertTokenizer, BertModel  # 从transformers库中导入BERT的分词器和模型
from sklearn.decomposition import PCA



def extract_time_feature(t_str):   #从事件串中提取事件特征
    a = str(t_str)
    t = datetime.fromisoformat(a)  #这一行将输入的时间字符串 t_str 转换为 datetime 对象。fromisoformat 方法用于解析ISO 8601格式的日期时间字符串，并返回对应的 datetime 对象。
    OLE_TIME_ZERO = datetime(1899, 12, 30) #这一行定义了一个常量 OLE_TIME_ZERO，它表示OLE时间的起始日期，即1899年12月30日
    delta = t - OLE_TIME_ZERO #算出距离起始日期的时间
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day
    #这一行计算了时间差 delta 的天数部分和秒数部分，并将它们转换为浮点数。
    # 然后，将天数部分除以 100000.0（这个数字可能是某种特定单位的转换因子）
    # 并将秒数部分除以 86400.0（一天的秒数）以获取小数形式的时间特征。最后，将这两个特征作为列表返回。

# encode the times-tamps of all the messages in the dataframe
def df_to_t_features(df):  #DataFrame中的时间戳数据转换为时间特征。
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    # 首先从DataFrame中选择名为 created_at 的列，该列包含时间戳数据。
    # 然后，它通过列表推导式遍历 created_at 列中的每个时间戳字符串 t_str，并调用 extract_time_feature 函数将每个时间戳字符串转换为时间特征。
    # 最后，使用 NumPy 的 asarray 函数将结果转换为NumPy数组，存储在变量 t_features 中。
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

# 确保数据为字符串列表，如果是单个字符串，可以转换为列表
combined_df = data_feature['text'].tolist()


# 文本预处理
# 文本预处理
words_all = []
for text in combined_df:
    words = jieba.lcut(text)
    words_all.append(words)

# 训练Word2Vec模型
print("训练")
model = Word2Vec(sentences=words_all, vector_size=300, min_count=1)

# 将文本转换为词向量
encoded_text = []
for words in words_all:
    if words:
        vector = np.mean([model.wv[word] for word in words if word in model.wv], axis=0)
        encoded_text.append(vector)
    else:
        encoded_text.append(np.zeros(model.vector_size))

# 转换为 numpy 数组
encoded_text = np.array(encoded_text)
print("encoded_text shape before reduction:", encoded_text.shape)  # 打印形状，确保它是二维的

# # 使用 PCA 将文本特征从 300 维降维到 300 维（如果需要降维）
# pca = PCA(n_components=300)
# encoded_text_reduced = pca.fit_transform(encoded_text)
# print("encoded_text shape after reduction:", encoded_text_reduced.shape)  # 应该是 (num_samples, 300)



print("结合两个特征")
combined_features = np.concatenate((encoded_text, combined_t_features), axis=1)
combined_df = pd.DataFrame(combined_features)
print("encoded_text shape after reduction:", combined_df.shape)  # 应该是 (num_samples, 302)

indices = data_feature['index'].values.tolist()
x = combined_df.iloc[indices, :]  # 使用 .iloc 方法访问行
y = data_feature['event_id'].values
y = [int(each) for each in y]
# y = [int(each) for each in data_feature['event_id'].values] #将索引推文中的event_id转换成int形
np.save('/root/cxt/construct_graph_20241125/Word2Vec_pic/labels.npy', np.asarray(y))

np.save('/root/cxt/construct_graph_20241125/dataset/302dim_Word2Vec.npy', combined_df)