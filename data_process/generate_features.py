import numpy as np
import pandas as pd
import en_core_web_trf
from datetime import datetime
import torch
import spacy

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
# 加载 NumPy 数组
p_part1 = '/root/cxt/construct_graph/datasets/68841_tweets_multiclasses_filtered_0722_part1.npy'
p_part2 = '/root/cxt/construct_graph/datasets/68841_tweets_multiclasses_filtered_0722_part2.npy'
df_np_part1 = np.load(p_part1, allow_pickle=True)
df_np_part2 = np.load(p_part2, allow_pickle=True)
# 连接两个数组
df_eng = np.concatenate((df_np_part1, df_np_part2), axis=0)

print(df_np_part1.shape)
print(df_eng.dtype)

#
# df_vi = pd.read_csv('/root/cxt/construct_graph/datasets/vi_tweets_with_entities_labels.csv', encoding='utf-8',engine='python')
# df_zh = pd.read_csv('/root/cxt/construct_graph/datasets/zh_tweets_with_entities.csv', encoding='utf-8',engine='python')




# en_fe = pd.read_csv('/root/cxt/construct_graph/datasets/en_tweets_output.csv')
# en_f = en_fe.to_numpy()
# vi_fe = pd.read_csv('/root/cxt/construct_graph/datasets/vi_tweets_output.csv')
# vi_f = vi_fe.to_numpy()
# zh_fe = pd.read_csv('/root/cxt/construct_graph/datasets/zh_tweets_output.csv')
# zh_f = zh_fe.to_numpy()



print("读取数据")
columns = ["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
           "place_type", "place_full_name", "place_country_code", "hashtags",
           "user_mentions", "image_urls", "entities", "words", "filtered_words", "sampled_words"]
en = pd.DataFrame(df_np_part1, columns=columns)

en_f = documents_to_features(en)
print("提取文本特征后的纬度：",en_f.shape)

# vi = pd.DataFrame(data=df_vi, columns=["event_id", "tweet_id", "user_id", "created_at", "hashtags", "user_mentions",\
#     "vi_text", "vi_text_filter", "Entities", "Labels"])
# print("越南语数据转换dataframe完成") #数据转换为数据帧
# # print(vi['created_at'])
#
#
#
# zh = pd.DataFrame(data=df_zh, columns=["event_id", "tweet_id", "user_id", "created_at", "hashtags", "user_mentions",\
#     "zh_text", "zh_text_filter", "Entities", "Labels"])
# print("中文数据转换dataframe完成") #数据转换为数据帧




en_t_features = df_to_t_features(en)
print("English Time features generated.")
#
# vi_t_features = df_to_t_features(vi)
# print("越南语 Time features generated.")
#
# zh_t_features = df_to_t_features(zh)
# print("chinese Time features generated.")

en_combined_features = np.concatenate((en_f, en_t_features), axis=1)  #将文档和时间特征连接在一起
print("Concatenated english document features and time features.")
df_en_combined_features = pd.DataFrame(en_combined_features)
print(df_en_combined_features.tail())
print(df_en_combined_features.shape)
# df_en_combined_features.to_csv('en_combined_features.csv',index=False)
# vi_combined_features = np.concatenate((vi_f, vi_t_features), axis=1)  #将文档和时间特征连接在一起
# print("Concatenated english document features and time features.")
# # 将 NumPy 数组转换为 Pandas DataFrame
# vi_df = pd.DataFrame(vi_combined_features)
# print(vi_df.shape)
# # 保存 DataFrame 为 CSV 文件
# vi_df.to_csv('vi_combined_features.csv', index=False)
# zh_combined_features = np.concatenate((zh_f, zh_t_features), axis=1)  #将文档和时间特征连接在一起
# print("Concatenated english document features and time features.")
# # 将 NumPy 数组转换为 Pandas DataFrame
# zh_df = pd.DataFrame(zh_combined_features)
# print(zh_df.shape)
# # 保存 DataFrame 为 CSV 文件
# zh_df.to_csv('zh_combined_features.csv', index=False)