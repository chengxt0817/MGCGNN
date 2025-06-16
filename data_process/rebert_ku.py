import numpy as np
import pandas as pd
import en_core_web_trf
from datetime import datetime
import torch
import spacy
from transformers import BertTokenizer, BertModel  # 从transformers库中导入BERT的分词器和模型
from sklearn.decomposition import PCA



def documents_to_features(df):
    if spacy.prefer_gpu():
        spacy.require_gpu()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    nlp = en_core_web_trf.load()  #en_core_web_lg 是一个spaCy预训练的英语语言模型，具有较大的词汇和向量表示。nlp 变量现在指向了加载的语言模型
    features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector).values #对选定的列（即 filtered_words 列）应用一个函数。
    # 这个函数使用了一个 lambda 函数，它将每个列表中的单词或词语连接成一个字符串（通过 join 函数），
    # 然后将这个字符串传递给 nlp 对象，即我们之前加载的spaCy语言模型。这个语言模型将文本转换为词向量表示，并返回整个文档的向量表示
    return np.stack(features, axis=0) #这一行代码使用NumPy的 stack 函数将所有特征向量堆叠在一起，形成一个二维数组，其中每一行代表一个文档的特征向量。最后，函数返回这个特征矩阵

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

# load dataset
#调整输出宽度
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)  # 调整输出的宽度

# # 读取数据
# df_vi = pd.read_csv('/root/cxt/construct_graph/datasets/vi_tweets_with_entities_labels.csv', encoding='utf-8', engine='python')
# df_zh = pd.read_csv('/root/cxt/construct_graph/datasets/zh_tweets_with_entities.csv', encoding='utf-8', engine='python')

# # 处理越南语数据集
# vi = pd.DataFrame(data=df_vi, columns=["event_id", "tweet_id", "user_id", "created_at", "hashtags", "user_mentions", "vi_text", "vi_text_filter", "Entities", "Labels"])
# vi = vi.rename(columns={'vi_text': 'text', 'vi_text_filter': 'text_filter'})
# vi['language'] = 'vi'

# print("越南语数据行数:", vi.shape[0])
# print(vi.head(5))

# # 处理中文数据集
# zh = pd.DataFrame(data=df_zh, columns=["event_id", "tweet_id", "user_id", "created_at", "hashtags", "user_mentions", "zh_text", "zh_text_filter", "Entities", "Labels"])
# zh = zh.rename(columns={'zh_text': 'text', 'zh_text_filter': 'text_filter'})
# zh['language'] = 'zh'
# print("中文数据行数:", zh.shape[0])
# print(zh.head(5))

# 处理英文数据集
df_np_part1 = np.load('/root/cxt/construct_graph/datasets/68841_tweets_multiclasses_filtered_0722_part1.npy', allow_pickle=True)
df_np_part2 = np.load('/root/cxt/construct_graph/datasets/68841_tweets_multiclasses_filtered_0722_part2.npy', allow_pickle=True)
df_eng = np.concatenate((df_np_part1, df_np_part2), axis=0)

columns = ["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc", "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", "words", "filtered_words", "sampled_words"]
en = pd.DataFrame(df_eng, columns=columns)
columns_to_keep = ["event_id", "tweet_id", "user_id", "created_at", "hashtags", "user_mentions", "text", "filtered_words", "entities", "words"]
en = en[columns_to_keep].rename(columns={"text": "text", "filtered_words": "text_filter", "entities": "Entities", "words": "Labels"})
en['language'] = 'en'
print("英文数据行数:", en.shape[0])
print(en.head(5))

# 拼接数据集
# combined_df = pd.concat([vi, zh, en], axis=0, ignore_index=True)
# print("拼接后数据集行数:", combined_df.shape[0])
# print("形状是：",combined_df.shape)


# print("清理数据删除空格")
# # 检查数据中是否包含额外的空白字符
# print(combined_df['text'].str.contains('\s').sum())
# # 清理数据中的额外空白字符
# combined_df['text'] = combined_df['text'].str.replace(r'\s+', ' ', regex=True)

print("清理数据删除空格")
# 检查数据中是否包含额外的空白字符
print(en['text'].str.contains('\s').sum())
# 清理数据中的额外空白字符
en['text'] = en['text'].str.replace(r'\s+', ' ', regex=True)

print("打乱")
#打乱行
combined_df = en.sample(frac=1, random_state=42).reset_index(drop=True)

data_feature = pd.DataFrame(data=combined_df, columns=["event_id", "tweet_id", "user_id", "created_at", "hashtags", "user_mentions", "text", "filtered_words", "Entities", "Labels", "language"])
print(data_feature)
np.save('dataset/data_features1.npy',data_feature)


#时间特征
print("提取时间特征")
combined_t_features = df_to_t_features(combined_df)
print("Time features generated.")
print(combined_t_features)


# print("打乱")
# #打乱行
# combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# data_feature = pd.DataFrame(data=combined_df, columns=["event_id", "tweet_id", "user_id", "created_at", "hashtags", "user_mentions", "text", "filtered_words", "Entities", "Labels", "language"])
# print(data_feature)
# np.save('dataset/data_features.npy',data_feature)
# #时间特征
# print("提取时间特征")
# combined_t_features = df_to_t_features(combined_df)
# print("Time features generated.")
# print(combined_t_features)


#开始嵌入（mbert）
# 加载mBERT模型和分词器
tokenizer_model_dir = '/root/cxt/construct_graph/my-bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_model_dir)  # 加载预训练的mBERT分词器
bert_model_dir = '/root/cxt/construct_graph/my-bert-base-multilingual-uncased'
model = BertModel.from_pretrained(bert_model_dir)          # 加载预训练的mBERT模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 加载.npy文件中的数据
print('读取文件成功')
# 确保数据为字符串列表，如果是单个字符串，可以转换为列表
text_list = combined_df['text'].tolist()

encoded_text = []

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    encoded_text = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    encoded_text = np.array(encoded_text)
    print("encoded_text shape:", encoded_text.shape)  # 打印形状，确保它是二维的
    # print("我走到很慢但我在走")
    return encoded_text

# 遍历每一个文本，进行编码
for text in text_list:
    # 检查数据是否为字符串
    encoded_text.append(encode_text(str(text)))
    print("我走的很慢，但我再走")

# 转换为 numpy 数组
encoded_text = np.array(encoded_text)
print("encoded_text shape before reduction:", encoded_text.shape)  # 打印形状，确保它是二维的

# 使用 PCA 将文本特征从 768 维降维到 300 维
pca = PCA(n_components=300)
encoded_text_reduced = pca.fit_transform(encoded_text)
print("encoded_text shape after reduction:", encoded_text_reduced.shape)  # 应该是 (num_samples, 300)
np.save('/root/cxt/construct_graph_20241113/dataset/300dim_en.npy', encoded_text_reduced)
#
print("结合两个特征")
combined_features = np.concatenate((encoded_text_reduced, combined_t_features), axis=1)
combined_df = pd.DataFrame(combined_features)
print("encoded_text shape after reduction:", combined_df.shape)  # 应该是 (num_samples, 300)
np.save('/root/cxt/construct_graph_20241113/dataset/302dim_en.npy', combined_df)

# 为列命名（可选，取决于你希望在 CSV 中显示的列名）
# 假设 encoded_df 有 N 列，combined_t_features 有 M 列
# combined_df.columns = [f'feature_{i}' for i in range(combined_features.shape[1])]
#
# # 保存为 CSV 文件
# combined_df.to_csv('/root/cxt/construct_graph/combined_features.csv', index=False)
# np.save('/root/cxt/construct_graph/combined_features.npy', combined_features) #将合并的特征数组保存到文件中
# print("Initial features saved.")











