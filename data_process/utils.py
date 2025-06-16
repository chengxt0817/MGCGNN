import dgl
import numpy as np
import torch
from scipy import sparse
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from main import args_define
import torch.nn.functional as F
from itertools import combinations
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
# %matplotlib inline
# np.set_printoptionslinewidth=100
# np.set_printoptions(threshold=np.inf)




args = args_define.args
# Dataset
class SocialDataset(Dataset):
    def __init__(self, path, index):
        self.features = np.load(path + '/' + str(index) + '/features.npy')
        # self.node_features = np.load(path + '/' + str(index) + '/node_embeddings.npy')
        temp = np.load(path + '/' + str(index) + '/labels.npy', allow_pickle=True)
        # 将标签中的每一个元素转换成整数存储在labels中
        self.labels = np.asarray([int(each) for each in temp])
        # 加载文件夹中已经切分好的邻接矩阵
        self.matrix = self.load_adj_matrix(path, index)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    # 加载对应文件夹中的矩阵的代码（同构图）
    def load_adj_matrix(self, path, index):
        s_bool_A_tid_tid = sparse.load_npz(path + '/' + str(index) + '/s_bool_A_tid_tid.npz')
        print("Sparse binary adjacency matrix loaded.")
        return s_bool_A_tid_tid

    # Used by remove_obsolete mode 1
    def remove_obsolete_nodes(self, indices_to_remove=None):  # indices_to_remove: list
        # torch.range(0, (self.labels.shape[0] - 1), dtype=torch.long)
        if indices_to_remove is not None:
            all_indices = np.arange(0, self.labels.shape[0]).tolist()
            indices_to_keep = list(set(all_indices) - set(indices_to_remove))
            self.features = self.features[indices_to_keep, :]
            self.labels = self.labels[indices_to_keep]
            self.matrix = self.matrix[indices_to_keep, :]  # keep row
            self.matrix = self.matrix[:, indices_to_keep]  # keep column
            #  remove nodes from matrix


#添加的方法
def euclidean_distance_matrix( a, b):
        # 计算两个批次中向量之间的欧氏距离
        dot_product = torch.mm(a, b.T)

        norm_squared_a = torch.sum(a ** 2, dim=1, keepdims=True)
        norm_squared_b = torch.sum(b ** 2, dim=1, keepdims=True)

        distances = torch.clamp(norm_squared_a - 2 * dot_product + norm_squared_b.t(), min=0.0)

        # 计算欧氏距离
        distances = torch.sqrt(distances + 1e-8)

        return distances


def get_cen_metric(extract_labels_set, extract_labels, extract_features):
        label_center = {}
        for l in extract_labels_set:
            l_indices = torch.where(extract_labels == l)[0]
            l_feas = extract_features[l_indices]
            l_cen = torch.mean(l_feas, 0)
            label_center[l.cpu().item()] = l_cen
        # 计算两两欧式距离
        cen_keys = np.array(list(label_center.keys()))
        cen_keys = {key: index for index, key in enumerate(cen_keys)}
        # 获取value矩阵
        cen_values = torch.stack(list(label_center.values()))
        cen_metric = euclidean_distance_matrix(cen_values, cen_values)

        return label_center, cen_keys, cen_metric, cen_values

def pairwise_sample(pre_pre_dot_metric, labels):
    labels = labels.cpu().data.numpy()
    indices = np.arange(0, len(labels), 1)
    pairs = np.array(list(combinations(indices, 2)))

    pair_labels_pos = np.where(labels[pairs[:, 0]] == labels[pairs[:, 1]], 2, 0)
    pair_labels_neg = np.where((labels[pairs[:, 0]] != labels[pairs[:, 1]]) & (pre_pre_dot_metric[pairs[:, 0], pairs[:, 1]] < 0) & (pre_pre_dot_metric[pairs[:, 1], pairs[:, 0]] == 0), -2, 0)

    # pair_labels_neg = np.where((labels[pairs[:, 0]] != labels[pairs[:, 1]]) & (pre_pre_dot_metric[pairs[:, 0], pairs[:, 1]] == 0) & (pre_pre_dot_metric[pairs[:, 1], pairs[:, 0]] == 0), -2, 0)
    pair_labels = pair_labels_neg + pair_labels_pos
    print("正样本对:", pair_labels_pos)
    print("负样本对:", pair_labels_neg)
    if np.any(pair_labels_pos == 2):
        print("有2")
        # indices = np.where(pair_labels_pos == 2)[0]
        # print("索引对应的neg位置的数字为：",pair_labels_neg[indices])
    else:
        print("没有2")
    pair_matrix = np.eye(len(labels))
    ind = np.where(pair_labels)
    pair_matrix[pairs[ind[0], 0], pairs[ind[0], 1]] = 1
    pair_matrix[pairs[ind[0], 1], pairs[ind[0], 0]] = 1

    return torch.LongTensor(pairs), torch.LongTensor(pair_labels.astype(int)), torch.LongTensor(pair_matrix)


# region
# def pairwise_sample(embeddings, labels=None, model=None):
#     if model == None:#labels is not None:
#         labels = labels.cpu().data.numpy()
#         indices = np.arange(0,len(labels),1)
#         pairs = np.array(list(combinations(indices, 2)))
#         pair_labels = (labels[pairs[:,0]]==labels[pairs[:,1]])

#         pair_matrix = np.eye(len(labels))
#         ind = np.where(pair_labels)
#         pair_matrix[pairs[ind[0],0],pairs[ind[0],1]] = 1
#         pair_matrix[pairs[ind[0],1], pairs[ind[0],0]] = 1


#         return torch.LongTensor(pairs), torch.LongTensor(pair_labels.astype(int)),torch.LongTensor(pair_matrix)

#     else:
#         # indices = np.arange(0,embeddings.shape[0],1)
#         # pairs = np.array(list(combinations(indices, 2)))
#         # model.eval()
#         # input = torch.cat((embeddings[pairs[:,0]],embeddings[pairs[:,1]]), 1)
#         # input = embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]]
#         # print(input)
#         # out = model(input)
#         # evi = relu_evidence(out)
#         # alpha = evi + 1
#         # print(evi.shape)
#         # _, pseudo_labels = torch.max(alpha, 1)
#         # S = torch.sum(alpha, dim=1, keepdim=True)
#         # u = 2 / S

#         # labels = labels.cpu().data.numpy()
#         # indices = np.arange(0, len(labels), 1)
#         # pairs = np.array(list(combinations(indices, 2)))
#         # pair_labels = (labels[pairs[:, 0]] == labels[pairs[:, 1]])

#         # return torch.LongTensor(pairs), pseudo_labels, u

#         pair_matrix = model(embeddings)
#         return pair_matrix
#endregion

#添加的方法完毕



# Compute the representations of all the nodes in g using model
def extract_embeddings(g, model, num_all_samples, args, labels):
    with torch.no_grad():
        model.eval()
    #改动代码
        # if args.use_cuda:
        #     g = g.to('cpu')  # 将图移动到 CPU
        indices = torch.LongTensor(np.arange(0,num_all_samples,1))
        if args.use_cuda:
            indices = indices.cuda()
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, 
            indices,
            sampler,
            batch_size=num_all_samples,
            shuffle=False,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            # print('blocks:',blocks)
            extract_labels = blocks[-1].dstdata['labels']
            extract_features = model(blocks)

        assert batch_id == 0
        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()

    return (extract_features, extract_labels)



    #改动完毕

    #region
    #     for batch_id, nf in enumerate(
    #             dgl.contrib.sampling.NeighborSampler(g,  # sample from the whole graph (contain unseen nodes)
    #                                                  num_all_samples,  # set batch size = the total number of nodes
    #                                                  1000,
    #                                                  # set the expand_factor (the number of neighbors sampled from
    #                                                  # the neighbor list of a vertex) to None: get error: non-int
    #                                                  # expand_factor not supported
    #                                                  neighbor_type='in',
    #                                                  shuffle=False,
    #                                                  num_workers=32,
    #                                                  num_hops=2)):
    #         nf.copy_from_parent()
    #         if args.use_dgi:
    #             extract_features, _ = model(nf)  # representations of all nodes
    #         else:
    #             extract_features = model(nf)  # representations of all nodes
    #         # 存储节点id
    #         extract_nids = nf.layer_parent_nid(-1).to(device=extract_features.device, dtype=torch.long)  # node ids
    #         #所有种子节点的标签
    #         extract_labels = labels[extract_nids]  # labels of all nodes
    #     assert batch_id == 0
    #     extract_nids = extract_nids.data.cpu().numpy()
    #     extract_features = extract_features.data.cpu().numpy()
    #     extract_labels = extract_labels.data.cpu().numpy()
    #     # generate train/test mask
    #     # 生成一组连续数组，数量为标签样本数量
    #     A = np.arange(num_all_samples)
    #     # print("A", A)
    #     # 如果 (A == extract_nids).all() 的结果为 True，则说明数组 A 和 extract_nids 中的所有元素都完全相等；
    #     # 否则，会触发 AssertionError，表示断言失败。
    #     assert (A == extract_nids).all()

    # return extract_nids, extract_features, extract_labels
    #endregion

def save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, path, counter):
    np.savetxt(path + '/features_' + str(counter) + '.tsv', extract_features, delimiter='\t')
    np.savetxt(path + '/labels_' + str(counter) + '.tsv', extract_labels, fmt='%i', delimiter='\t')
    with open(path + '/labels_tags_' + str(counter) + '.tsv', 'w') as f:
        f.write('label\tmessage_id\ttrain_tag\n')
        for (label, mid, train_tag) in zip(extract_labels, extract_nids, extract_train_tags):
            f.write("%s\t%s\t%s\n" % (label, mid, train_tag))
    print("Embeddings after inference epoch " + str(counter) + " saved.")
    print()


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def run_dbscan(save_path, extract_features, extract_labels, indices, args, isoPath=None, eps=0.5, min_samples=5):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    # t-SNE for dimensionality reduction (optional, but useful for visualization)
    tsne = TSNE(n_components=2, random_state=0)
    reduced_features = tsne.fit_transform(extract_features)

    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = dbscan.labels_

    # Handle noise points (-1 in DBSCAN labels)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise points
    print(f"Estimated number of clusters: {n_clusters}")

    # For visualization, we need to filter out noise points
    reduced_features_for_plot = reduced_features[indices]
    labels_for_plot = labels[:len(reduced_features_for_plot)]

    # Plotting the clustering results
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_features_for_plot[:, 0], reduced_features_for_plot[:, 1], c=labels_for_plot, cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    data_path = save_path + '/pic/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    plt.savefig(data_path + 'cluster_visualization.png', bbox_inches='tight', pad_inches=0.05)

    # Calculate clustering metrics (excluding noise points)
    labels_true_filtered = labels_true[labels != -1]
    labels_filtered = labels[labels != -1]

    nmi = normalized_mutual_info_score(labels_true_filtered, labels_filtered)
    ari = adjusted_rand_score(labels_true_filtered, labels_filtered)
    ami = adjusted_mutual_info_score(labels_true_filtered, labels_filtered, average_method='arithmetic')
    print("nmi:", nmi, 'ami:', ami, 'ari:', ari)
    value = nmi
    global NMI
    NMI = nmi
    global AMI
    AMI = ami
    global ARI
    ARI = ari

    if args.metrics == 'ari':
        print('use ari')
        value = ari
    if args.metrics == 'ami':
        print('use ami')
        value = ami

    # Return number of test tweets, number of clusters (excluding noise), and clustering metric value
    return (n_test_tweets, n_clusters, value)


#更改的kmeans代码
def run_kmeans(save_path,extract_features, extract_labels, indices, args,isoPath=None):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    tsne = TSNE(n_components=2, random_state=0)
    reduced_features = tsne.fit_transform(extract_features)

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    # kmeans.fit(reduced_features)
    reduced_features_for_plot = reduced_features[indices]
    labels_for_plot = labels[:len(reduced_features_for_plot)]
    print(labels_for_plot.shape)

    # 绘制聚类结果
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_features_for_plot[:, 0], reduced_features_for_plot[:, 1], c=labels_for_plot, cmap='viridis')
    # 去掉x轴和y轴上的刻度线
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plt.show()
    data_path = save_path + '/pic/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    plt.savefig(data_path + 'cluster_visualization.png', bbox_inches='tight', pad_inches=0.05)

    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    ari = metrics.adjusted_rand_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic')
    print("nmi:",nmi,'ami:',ami,'ari:',ari)
    value = nmi
    global NMI
    NMI = nmi
    global AMI
    AMI = ami
    global ARI
    ARI = ari

    if args.metrics =='ari':
        print('use ari')
        value = ari
    if args.metrics=='ami':
        print('use ami')
        value = ami
    # Return number  of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, value)

#更改结束

#region
# def run_kmeans(extract_features, extract_labels, indices, isoPath=None):
#     # Extract the features and labels of the test tweets
#     # 这个索引是测试的节点的索引
#     indices = indices.cpu().detach().numpy()

#     if isoPath is not None:
#         # Remove isolated points
#         temp = torch.load(isoPath)
#         temp = temp.cpu().detach().numpy()
#         non_isolated_index = list(np.where(temp != 1)[0])
#         indices = intersection(indices, non_isolated_index)

#     # Extract labels
#     labels_true = extract_labels[indices]
#     # Extract features
#     # 按照测试索引提取这些节点对应的特征
#     X = extract_features[indices, :]
#     assert labels_true.shape[0] == X.shape[0]
#     # 赋值n_test_tweets列表对应的特征值
#     n_test_tweets = X.shape[0]

#     # Get the total number of classes
#     # 将标签数组 labels_true 转换为列表，然后使用 set 函数将其转换为集合，这样可以获取所有唯一的标签值
#     # 得到了标签中的类别数量
#     n_classes = len(set(list(labels_true)))

#     # k-means clustering
#     # 这行代码使用 KMeans 算法对特征向量 X 进行聚类，将数据分为 n_classes 个簇。具体操作如下：
#     #
#     # KMeans(n_clusters=n_classes, random_state=0): 创建一个 KMeans 聚类器对象，其中 n_clusters 参数指定要分成的簇的数量，即类别数目。
#     # fit(X): 使用特征向量 X 对 KMeans 聚类器进行拟合，从而完成聚类过程。拟合后，聚类器会将每个样本分配到一个簇中，并确定每个簇的中心点。
#     # 最终，kmeans 变量中存储了拟合后的 KMeans 聚类器对象。
#     kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#     # KMeans 类的实例包含以下属性和方法：
#     #
#     # cluster_centers_：聚类中心的坐标，即每个簇的中心点。
#     # labels_：每个样本的簇标签，即样本被分配到的簇的索引。
#     # inertia_：聚类后的总内聚度，即所有样本到其所属簇中心的距离的总和。
#     # n_iter_：聚类过程中迭代的次数。
#     # predict(X)：对新样本进行簇标签的预测。
#     # fit_transform(X)：对数据进行聚类并返回簇标签和相对于每个聚类中心的距离的变换。
#     # 其他一些控制聚类行为的参数，如 n_clusters（簇的数量）、init（初始化簇中心的方法）、max_iter（最大迭代次数）等。
#     labels = kmeans.labels_
#     # 这段代码计算了两个聚类结果之间的标准化互信息（Normalized Mutual Information，NMI），用于评估聚类的质量。
#     # NMI是一种常用的聚类评估指标，用于衡量两个聚类结果的一致性程度，取值范围在0到1之间，值越高表示两个聚类结果越一致。
#     nmi = metrics.normalized_mutual_info_score(labels_true, labels)

#     # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
#     return (n_test_tweets, n_classes, nmi)
#endregion

# 更改的评估函数
def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, args, is_validation=True):
    message = ''
    message += '\nEpoch '
    message += str(epoch+1)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, value = run_dbscan(save_path,extract_features, extract_labels, indices, args)

    # n_tweets, n_classes, value = run_kmeans(save_path,extract_features, extract_labels, indices, args)
    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode +' '
    message += args.metrics +': '
    message += str(value)
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, value= run_kmeans(save_path,extract_features, extract_labels, indices, args,
                                              save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' value: '
        message += str(value)
        message += '\n'


    global NMI
    global AMI
    global ARI
    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI))
    print(message)

    all_value_save_path = "/".join(save_path.split('/')[0:-1])
    print(all_value_save_path)

    with open(all_value_save_path + '/evaluate.txt', 'a') as f:
        f.write("block "+ save_path.split('/')[-1])
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI) + '\n')

    return value

# 更改结束



# region
# def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, is_validation=True):
#     message = ''
#     message += '\nEpoch '
#     message += str(epoch)
#     message += '\n'

#     # with isolated nodes
#     n_tweets, n_classes, nmi = run_kmeans(extract_features, extract_labels, indices)
#     if is_validation:
#         mode = 'validation'
#     else:
#         mode = 'test'
#     message += '\tNumber of ' + mode + ' tweets: '
#     message += str(n_tweets)
#     message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
#     message += str(n_classes)
#     message += '\n\t' + mode + ' NMI: '
#     message += str(nmi)
#     if num_isolated_nodes != 0:
#         # without isolated nodes
#         message += '\n\tWithout isolated nodes:'
#         n_tweets, n_classes, nmi = run_kmeans(extract_features, extract_labels, indices,
#                                               save_path + '/isolated_nodes.pt')
#         message += '\tNumber of ' + mode + ' tweets: '
#         message += str(n_tweets)
#         message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
#         message += str(n_classes)
#         message += '\n\t' + mode + ' NMI: '
#         message += str(nmi)
#     message += '\n'

#     with open(save_path + '/evaluate.txt', 'a') as f:
#         f.write(message)
#     print(message)

#     return nmi
#endregion

def graph_statistics(G, save_path):
    message = '\nGraph statistics:\n'

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges / 2) // num_nodes
    in_degrees = G.in_degrees() # 获取入度
    isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = (in_degrees == isolated_nodes)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt')
    num_isolated_nodes = torch.sum(isolated_nodes).item()

    message += 'We have ' + str(num_nodes) + ' nodes.\n'
    message += 'We have ' + str(num_edges / 2) + ' in-edges.\n'
    message += 'Average degree: ' + str(ave_degree) + '\n'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "a") as f:
        f.write(message)

    return num_isolated_nodes



def generateMasks(length, data_split, train_i, i, validation_percent=0.2, save_path=None, num_indices_to_remove=0):
    """
        Intro:
        This function generates train and validation indices for initial/maintenance epochs and test indices for inference(prediction) epochs
        If remove_obsolete mode 0 or 1:
        For initial/maintenance epochs:
        - The first (train_i + 1) blocks (blocks 0, ..., train_i) are used as training set (with explicit labels)
        - Randomly sample validation_percent of the training indices as validation indices
        For inference(prediction) epochs:
        - The (i + 1)th block (block i) is used as test set
        Note that other blocks (block train_i + 1, ..., i - 1) are also in the graph (without explicit labels, only their features and structural info are leveraged)
        If remove_obsolete mode 2:
        For initial/maintenance epochs:
        - The (i + 1) = (train_i + 1)th block (block train_i = i) is used as training set (with explicit labels)
        - Randomly sample validation_percent of the training indices as validation indices
        For inference(prediction) epochs:
        - The (i + 1)th block (block i) is used as test set

        :param length: the length of label list
        :param data_split: loaded splited data (generated in custom_message_graph.py)
        :param train_i, i: flag, indicating for initial/maintenance stage if train_i == i and inference stage for others
        :param validation_percent: the percent of validation data occupied in whole dataset
        :param save_path: path to save data
        :param num_indices_to_remove: number of indices ought to be removed

        :returns train indices, validation indices or test indices
    """
    if args.remove_obsolete == 0 or args.remove_obsolete == 1:  # remove_obsolete mode 0 or 1
        # verify total number of nodes
        # 断言函数吗，用于验证总共的节点数量
        assert length == (np.sum(data_split[:i + 1]) - num_indices_to_remove)

        # If is in initial/maintenance epochs, generate train and validation indices
        if train_i == i:
            # randomly shuffle the training indices
            train_length = np.sum(data_split[:train_i + 1])
            train_length -= num_indices_to_remove
            # 随机排列训练数据的索引
            train_indices = torch.randperm(int(train_length))
            # get total number of validation indices
            n_validation_samples = int(train_length * validation_percent)
            # sample n_validation_samples validation indices and use the rest as training indices
            validation_indices = train_indices[:n_validation_samples]
            train_indices = train_indices[n_validation_samples:]
            if save_path is not None:
                torch.save(validation_indices, save_path + '/validation_indices.pt')
                torch.save(train_indices, save_path + '/train_indices.pt')
                validation_indices = torch.load(save_path + '/validation_indices.pt')
                train_indices = torch.load(save_path + '/train_indices.pt')
            return train_indices, validation_indices
        # If the process is in inference(prediction) epochs, generate test indices
        else:
            test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
            test_indices += (np.sum(data_split[:i]) - num_indices_to_remove)
            if save_path is not None:
                torch.save(test_indices, save_path + '/test_indices.pt')
                test_indices = torch.load(save_path + '/test_indices.pt')
            return test_indices

    else:  # remove_obsolete mode 2
        # verify total number of nodes
        assert length == data_split[i]

        # If is in initial/maintenance epochs, generate train and validation indices
        if train_i == i:
            # randomly shuffle the graph indices
            train_indices = torch.randperm(length)
            # get total number of validation indices
            n_validation_samples = int(length * validation_percent)
            # sample n_validation_samples validation indices and use the rest as training indices
            validation_indices = train_indices[:n_validation_samples]
            train_indices = train_indices[n_validation_samples:]
            if save_path is not None:
                torch.save(validation_indices, save_path +
                           '/validation_indices.pt')
                torch.save(train_indices, save_path + '/train_indices.pt')
                validation_indices = torch.load(
                    save_path + '/validation_indices.pt')
                train_indices = torch.load(save_path + '/train_indices.pt')
            return train_indices, validation_indices
        # If is in inference(prediction) epochs, generate test indices
        else:
            test_indices = torch.range(
                0, (data_split[i] - 1), dtype=torch.long)
            if save_path is not None:
                torch.save(test_indices, save_path + '/test_indices.pt')
                test_indices = torch.load(save_path + '/test_indices.pt')
            return test_indices


# Utility function, finds the indices of the values' elements in tensor
def find(tensor, values):
    return torch.nonzero(tensor.cpu()[..., None] == values.cpu())
