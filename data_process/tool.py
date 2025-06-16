import numpy as np
import pandas as pd
import en_core_web_trf
from datetime import datetime
import torch
import spacy
from transformers import BertTokenizer, BertModel  # 从transformers库中导入BERT的分词器和模型
from sklearn.decomposition import PCA


data_features = np.load('/root/cxt/construct_graph/data_feature_new.npy', allow_pickle=True)

df = pd.DataFrame(data=data_features, columns=["event_id", "tweet_id", "user_id", "created_at", "hashtags", "user_mentions", "text", "filtered_words", "Entities", "Labels", "language", "Qids"])
print("Data converted to dataframe.")
print(df.shape[0])

# 将 Qids 列中的 None 值替换为 []
df['Qids'] = df['Qids'].apply(lambda x: x if x is not None else [])

# 保存更新后的 DataFrame 为 .npy 文件
np.save('/root/cxt/construct_graph/data_feature_new.npy', df.to_numpy(), allow_pickle=True)




# half_df = df.iloc[:df.shape[0] // 2]
# np.save('/root/cxt/construct_graph/data_first_half.npy', half_df.to_numpy())

# f = np.load('/root/cxt/construct_graph/dataset/300dim.npy')
# print("录入嵌入维度信息")
# half_f = f[:f.shape[0] // 2]
# np.save('/root/cxt/construct_graph/300dim_half.npy', half_f)


