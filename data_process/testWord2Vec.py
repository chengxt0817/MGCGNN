import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics

# 加载嵌入特征和真实标签
embedding_path = '/root/cxt/construct_graph_20241125/graph_Word2Vec/21/features.npy'
# embedding_path = '/root/cxt/construct_graph_20241125/dataset/302dim_Word2Vec.npy'
true_labels_path = '/root/cxt/construct_graph_20241125/graph_Word2Vec/21/labels.npy'
# true_labels_path = '/root/cxt/construct_graph_20241125/Word2Vec_pic/labels.npy'

features = np.load(embedding_path)
true_labels = np.load(true_labels_path)

print(f"Loaded features with shape: {features.shape}")
print(f"Loaded true labels with shape: {true_labels.shape}")

# 确保特征和标签的数量匹配
assert features.shape[0] == true_labels.shape[0]

# 获取类别数量
n_classes = len(np.unique(true_labels))

# 进行K-Means聚类
kmeans = KMeans(n_clusters=n_classes, random_state=0)
kmeans.fit(features)
cluster_labels = kmeans.labels_

# 计算评价指标
nmi = metrics.normalized_mutual_info_score(true_labels, cluster_labels)
ari = metrics.adjusted_rand_score(true_labels, cluster_labels)
ami = metrics.adjusted_mutual_info_score(true_labels, cluster_labels, average_method='arithmetic')

print(f"NMI: {nmi}, ARI: {ari}, AMI: {ami}")

# # 可视化聚类结果
# tsne = TSNE(n_components=2, random_state=0)
# reduced_features = tsne.fit_transform(features)

# plt.figure(figsize=(8, 8))
# scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')

# # 去掉x轴和y轴上的刻度线
# plt.xticks([])
# plt.yticks([])
# plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

# # plt.colorbar(scatter)  # 显示颜色条
# # plt.title('Clustering Visualization')
# # plt.xlabel('X-axis')
# # plt.ylabel('Y-axis')

# # 保存可视化结果
# save_path = '/root/cxt/construct_graph_20241125/Word2Vec_pic/'
# data_path = os.path.join(save_path, 'pic/')
# if not os.path.exists(data_path):
#     os.makedirs(data_path)
# plt.savefig(os.path.join(data_path, '9.png'), bbox_inches='tight', pad_inches=0.05)
# plt.close()