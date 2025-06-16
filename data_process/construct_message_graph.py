"""
This file splits the Twitter dataset into 21 message blocks (please see Section 4.3 of the paper for more details),
use the message blocks to construct heterogeneous social graphs (please see Figure 1(a) and Section 3.2 of the paper for more details)
and maps them into homogeneous message graphs (Figure 1(c)).

Note that:
# 1) We adopt the Latest Message Strategy (which is the most efficient and gives the strongest performance. See Section 4.4 of the paper for more details) here,
# as a consequence, each message graph only contains the messages of the date and all previous messages are removed from the graph;
# To switch to the All Message Strategy or the Relevant Message Strategy, replace 'G = construct_graph_from_df(incr_df)' with 'G = construct_graph_from_df(incr_df, G)' inside construct_incremental_dataset_0922().
# 2) For test purpose, when calling construct_incremental_dataset_0922(), set test=True, and the message blocks, as well as the resulted message graphs each will contain 100 messages.
# To use all the messages, set test=False, and the number of messages in the message blocks will follow Table. 4 of the paper.
"""
import os
from time import time
import dgl
# from dgl.data.utils import save_graphs
# import dgl.geometry
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
import torch as th
import torch
import dgl.data
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import matplotlib.pyplot as plt
# from weighted_metapath2vec import WeightedMetapath2VecModel
# import multiprocessing
# from stellargraph import StellarGraph
# from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec
import random

# device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
class DeepWalk:
    def __init__(self, G, walk_length=80, num_walks=15, embedding_dim=128, window_size=5, workers=4):
        self.G = G
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.workers = workers

    def simulate_walks(self):
        walks = []
        nodes = list(self.G.nodes())
        for _ in range(self.num_walks):
            print("在训练呢，别急！！！")
            for node in nodes:
                walk = [str(node)]
                for _ in range(self.walk_length - 1):
                    neighbors = list(self.G.neighbors(node))
                    if neighbors:
                        next_node = random.choice(neighbors)
                        walk.append(str(next_node))
                        node = next_node
                walks.append(walk)
        return walks

    def learn_embeddings(self, walks):
        model = Word2Vec(
            walks,
            size=self.embedding_dim,
            window=self.window_size,
            min_count=1,
            sg=1,  # 使用 Skip-gram 模型
            workers=self.workers
        )
        node_embeddings = {node_id: model.wv[node_str] for node_id, node_str in enumerate(model.wv.index2word)}
        return node_embeddings

    def run(self):
        walks = self.simulate_walks()
        node_embeddings = self.learn_embeddings(walks)
        return node_embeddings



def visualize_graph(G):
    # 将 DGL 图转换为 NetworkX 图
    G_nx = G.to_networkx()

    # 创建一个 NetworkX 图的绘图
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G_nx)  # 布局算法，可以尝试其他布局如 'circular', 'random', 'shell' 等

    # 绘制节点
    nx.draw_networkx_nodes(G_nx, pos, node_size=70, node_color='skyblue')

    # 绘制边
    nx.draw_networkx_edges(G_nx, pos, alpha=0.4)

    # 可选：绘制节点标签
    nx.draw_networkx_labels(G_nx, pos, font_size=10, font_family='sans-serif')

    # 显示图形
    plt.axis('off')  # 关闭坐标轴
    plt.show()



# construct a heterogeneous social graph using tweet ids, user ids, entities and rare (sampled) words (4 modalities)
# if G is not None then insert new nodes to G
def construct_graph_from_df(df,features, threshold=0.7, G=None):
    if G is None:
        G = nx.Graph()

    # 创建一个字典来映射节点ID到其在features数组中的索引
    node_to_index = {f't_{row["tweet_id"]}': idx for idx, (_, row) in enumerate(df.iterrows())}

    # 这是一个 for 循环语句，用于遍历迭代器中的每个元素。在这个语句中，_ 是一个占位符，表示对索引不感兴趣，
    # 而 row 则是用于接收每一行数据的变量。循环将迭代器中的每一行数据赋值给 row，并执行循环体中的代码块。
    for _, row in df.iterrows():
        # 创建节点tid：其格式是t_+tweet_id对应的名称
        tid = 't_' + str(row['tweet_id'])
        #向图 G 中添加一个新的节点，节点的标识符为 tid。如果该节点已经存在于图中，则该操作不会对图做出任何修改。
        G.add_node(tid)
        # 这一行代码设置了节点 tid 的属性 'tweet_id' 的值为 True。在 NetworkX 中，
        # 节点的属性可以用于存储与节点相关的任何信息。这里的意思是，节点 tid 具有一个名为 'tweet_id' 的属性，
        # 并且该属性的值被设置为 True。在这种情况下，可能是表示该节点对应的推特（tweet）已经存在（或者被处理）了。
        G.nodes[tid]['tweet_id'] = True  # right-hand side value is irrelevant for the lookup
        # G.nodes[tid]['type'] = 'tweet_id'
        # 将 DataFrame 中的列 'user_mentions' 的值赋给变量 user_ids。这个列包含了被推文提及的用户ID列表。
        user_ids = []
        if isinstance(row['user_mentions'], list) and row['user_mentions']:
            user_ids = row['user_mentions']
        # 然后，将当前行的用户ID（row['user_id']）添加到 user_ids 列表中。
        # 这是因为在推文中，除了在 'user_mentions' 列中提及的用户外，发布推文的用户自身也可能被提及。
        if row['user_id'] not in [None, '']:  # 检查 user_id 是否有效
            user_ids.append(row['user_id']) #这里的user_id里面有数据
        filtered_user_ids = [uid for uid in user_ids if uid not in [None, ''] and not (isinstance(uid, list) and len(uid) == 0)]
        # print('这是添加userid之后的',user_ids)
        # 对 user_ids 列表中的每个用户ID进行遍历，并将其转换为以 'u_' 为前缀的字符串形式。
        # 这个操作创建了一个新的列表，其中每个用户ID都以 'u_' 为前缀。这样做可能是为了标识这些ID与用户相关联。
        user_ids = ['u_' + str(each) for each in filtered_user_ids]

        print("userid为：",user_ids)
        # 在图G中继续增加节点user_ids
        G.add_nodes_from(user_ids)
        for each in user_ids:
            # 给图G中的每一个user_ids节点机上一个"user_id"标签
            G.nodes[each]['user_id'] = True
            # G.nodes[each]['type'] = 'user_id'
        entities = row['Entities']
        if isinstance(entities, list):  # 确保 entities 是列表
            entities = ['e_' + '_'.join(str(entity) for entity in entity_tuple) for entity_tuple in entities]
        else:
            entities = ['e_' + str(entities)]  # 如果 entities 不是列表，直接处理
        print('e_id:',entities)
        # 同样的添加实体标签和在图G中添加实体
        G.add_nodes_from(entities)
        for each in entities:
            G.nodes[each]['entity'] = True
            # G.nodes[each]['type'] = 'entity'
        # 同样的添加Qids节点并加上word标签

        
        Qids = row['Qids']
        # Qids = ['Q_' + each for each in Qids]
        print('Q_id:', Qids)
        G.add_nodes_from(Qids)
        for each in Qids:
            G.nodes[each]['Qids'] = True
            # G.nodes[each]['type'] = 'Qids'
        #添加边，将推文和用户、实体、还有Qid连成边
        edges = []
        edges += [(tid, each) for each in user_ids]
        edges += [(tid, each) for each in entities]
        edges += [(tid, each) for each in Qids]

        for edge in edges:
            if edge[0] not in G or edge[1] not in G[edge[0]]:
                G.add_edge(edge[0], edge[1], weight=1.0) 

        # 添加边到图G中
        G.add_edges_from(edges)
    
    # 计算所有节点对的相似度并添加边
    node_features = np.array([features[node_to_index[tid]] for tid in node_to_index]).astype('float32')

    d = node_features.shape[1]
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

    gpu_index.add(node_features)

    n = node_features.shape[0]
    batch_size = 2046
    edges_set = set()

    for i in range(0, n, batch_size):
        D, I = gpu_index.search(node_features[i:i+batch_size], min(n-i, batch_size))
        for j in range(D.shape[0]):
            for k in range(D.shape[1]):
                if D[j][k] > 0 and D[j][k] <= threshold:
                    tid = list(node_to_index.keys())[i + j]
                    node = list(node_to_index.keys())[i + k]
                    print("计算"+ tid + "节点和" + node +"节点相似度")
                    edge = tuple(sorted((tid, node)))
                    edges_set.add(edge)
    for edge in edges_set:
            if edge[0] not in G or edge[1] not in G[edge[0]]:
                G.add_edge(edge[0], edge[1], weight=1.0)


    G.add_edges_from(edges_set)

    print('图中有 %d 个点.' % G.number_of_nodes())
    print('图中有 %d 条边.' % G.number_of_edges())

    print("保存对于异构图中图特征的嵌入")
    # G_nx = dgl.to_networkx(G)  # 假设G是您的DGL图
    deepwalk = DeepWalk(G)
    node_embeddings = deepwalk.run()
    # embeddings_array = np.array(list(node_embeddings.values()))
    embeddings_array = np.array([node_embeddings[node] for node, _ in sorted(node_embeddings.items())])
    # print(embeddings_array)
    print("异构图的嵌入表示：",embeddings_array.shape)
    node_to_index = {node: idx for idx, node in enumerate(sorted(node_embeddings.keys()))}
    # print(node_to_index)


    return G, embeddings_array, node_to_index


# convert a heterogeneous social graph G to a homogeneous message graph following eq. 1 of the paper,
# and store the sparse binary adjacency matrix of the homogeneous message graph.
def to_dgl_graph_v3(G,node_embeddings, node_to_index, x, save_path):
    message = ''
    print('Start converting heterogeneous networkx graph to homogeneous dgl graph.')
    message += 'Start converting heterogeneous networkx graph to homogeneous dgl graph.\n'
    all_start = time()
    # 获取所有节点的列表
    print('\tGetting a list of all nodes ...')
    message += '\tGetting a list of all nodes ...\n'
    start = time()
    all_nodes = list(G.nodes) # 创建一个列表all_node并将图G中的所有节点都添加到这里
    mins = (time() - start) / 60  # 用于测量代码执行所用的时间
    print('\tDone. Time elapsed: ', mins, ' mins\n') #显示用时
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('All nodes: ', all_nodes[:10])
    print('Total number of nodes: ', len(all_nodes))

    print('\tGetting adjacency matrix ...')  # 获取邻接矩阵
    message += '\tGetting adjacency matrix ...\n'
    start = time()
    # 将图G转换为一个邻接矩阵A，节点之间有边的就用1表示，没有边就是0  更改了转换成邻接矩阵太大了，只能转换成稀疏矩阵
    A = nx.to_scipy_sparse_matrix(G, format='csr')
    # A = nx.to_numpy_matrix(G)  # Returns the graph adjacency matrix as a NumPy matrix.
    # 计算并记录用时
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # compute commuting matrices
    # 获取不同类型的节点列表
    print('\tGetting lists of nodes of various types ...')
    message += '\tGetting lists of nodes of various types ...\n'
    start = time()
    # 获取带有tweet_id属性的节点，并将其放入列表tid_nodes
    tid_nodes = list(nx.get_node_attributes(G, 'tweet_id').keys())
    # 获取带有user_id属性的节点，并将其放入列表user_nodes
    userid_nodes = list(nx.get_node_attributes(G, 'user_id').keys())
    # 获取带有word属性的节点，并将其放入列表word_nodes
    id_nodes = list(nx.get_node_attributes(G, 'Qids').keys())
    # 获取带有entity属性的节点，并将其放入列表entity_nodes
    entity_nodes = list(nx.get_node_attributes(G, 'entity').keys())
    del G  #删除变量G
    # 计算用时
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # 将节点列表转换为索引列表
    print('\tConverting node lists to index lists ...')
    message += '\tConverting node lists to index lists ...\n'
    start = time()
    #  find the index of target nodes in the list of all_nodes
    indices_tid = [all_nodes.index(x) for x in tid_nodes]  # 创建一个tid索引表，存储tid_nodes的索引位置
    indices_userid = [all_nodes.index(x) for x in userid_nodes] # 创建一个userid的索引表，存储userid_nodes的索引位置
    indices_id = [all_nodes.index(x) for x in id_nodes]  # 创建一个QID索引表，存储Qid_nodes的索引位置
    indices_entity = [all_nodes.index(x) for x in entity_nodes] # 创建一个entity索引表，存储entity_nodes的索引位置
    # 删除了其他表
    del tid_nodes
    del userid_nodes
    del id_nodes
    del entity_nodes
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # ----------------------tweet-user-tweet----------------------
    # 创建tweet-用户-tweet矩阵
    print('\tStart constructing tweet-user-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-user matrix ...')
    message += '\tStart constructing tweet-user-tweet commuting matrix ...\n\t\t\tStart constructing tweet-user ' \
               'matrix ...\n '
    start = time()
    w_tid_userid = A[np.ix_(indices_tid, indices_userid)] # 选取邻接矩阵中的关于tid索引的和关于userid索引的所有信息抽取出来构成新的w_tid_userid列表。
    #  return a N(indices_tid)*N(indices_userid) matrix, representing the weight of edges between tid and userid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    # 转换成scipy稀疏矩阵
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    # sparse.csr_matrix()函数用于将一个普通的密集矩阵（即NumPy数组）转换为稀疏矩阵。
    # 稀疏矩阵是一种数据结构，用于表示大多数元素为零的矩阵，只存储非零元素及其位置，以节省内存空间。
    # 这段代码就是将W_tid_userid这个只有关于tweetid和userid的邻接矩阵转换成稀疏矩阵，只保存非零元素和位置
    s_w_tid_userid = sparse.csr_matrix(w_tid_userid)  # matrix compression
    del w_tid_userid # 删除了原来的那个不稀疏的矩阵
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # 转置
    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    # 将稀疏矩阵进行转置
    s_w_userid_tid = s_w_tid_userid.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-user * user-tweet ...')
    message += '\t\t\tCalculating tweet-user * user-tweet ...\n'
    start = time()
    # 同构消息图就是将这个tid_userid 和自己的转置进行相乘就成了同构消息图
    s_m_tid_userid_tid = s_w_tid_userid * s_w_userid_tid  # homogeneous message graph
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    # 然后就保存了这个关于tid_userid_tid的稀疏矩阵
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_userid_tid.npz", s_m_tid_userid_tid)
        print("Sparse binary userid commuting matrix saved.")
        del s_m_tid_userid_tid
    del s_w_tid_userid
    del s_w_userid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'



    # 构建tweet_ent_tweet同构图
    # ----------------------tweet-ent-tweet------------------------
    print('\tStart constructing tweet-ent-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-ent matrix ...')
    message += '\tStart constructing tweet-ent-tweet commuting matrix ...\n\t\t\tStart constructing tweet-ent matrix ' \
               '...\n '
    start = time()
    # 和上面一样，将这个tid索引和entity索引存储到一个新的w_tid_entity列表中
    w_tid_entity = A[np.ix_(indices_tid, indices_entity)]
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    #转换成稀疏矩阵
    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    #将这个列表转换成一个稀疏矩阵
    s_w_tid_entity = sparse.csr_matrix(w_tid_entity)
    del w_tid_entity
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    #转置这个稀疏矩阵
    s_w_entity_tid = s_w_tid_entity.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-ent * ent-tweet ...')
    message += '\t\t\tCalculating tweet-ent * ent-tweet ...\n'
    start = time()
    # 将两个矩阵相乘得到同构关系图
    s_m_tid_entity_tid = s_w_tid_entity * s_w_entity_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    # 存储了这个s_m_tid_entity_tid的同构消息图
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_entity_tid.npz", s_m_tid_entity_tid)
        print("Sparse binary entity commuting matrix saved.")
        del s_m_tid_entity_tid
    del s_w_tid_entity
    del s_w_entity_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # ----------------------tweet-word-tweet----------------------
    print('\tStart constructing tweet-id-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-id matrix ...')
    message += '\tStart constructing tweet-id-tweet commuting matrix ...\n\t\t\tStart constructing tweet-word ' \
               'matrix ...\n '
    start = time()
    # 同样将tid索引和word索引提取出来放到新的列表中
    q_tid_word = A[np.ix_(indices_tid, indices_id)]
    del A
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    # 一样转换成稀疏矩阵
    s_q_tid_word = sparse.csr_matrix(q_tid_word)
    del q_tid_word
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_q_word_tid = s_q_tid_word.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-word * word-tweet ...')
    message += '\t\t\tCalculating tweet-word * word-tweet ...\n'
    start = time()
    # 转置相乘，得到最后的同构关系邻接矩阵
    s_m_tid_word_tid = s_q_tid_word * s_q_word_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_Qid_tid.npz", s_m_tid_word_tid)
        print("Sparse binary word commuting matrix saved.")
        del s_m_tid_word_tid
    del s_q_tid_word
    del s_q_word_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # ----------------------compute tweet-tweet adjacency matrix----------------------
    print('\tComputing tweet-tweet adjacency matrix ...')
    message += '\tComputing tweet-tweet adjacency matrix ...\n'
    start = time()
    if save_path is not None:
        s_m_tid_userid_tid = sparse.load_npz(save_path + "s_m_tid_userid_tid.npz")
        print("Sparse binary userid commuting matrix loaded.")
        s_m_tid_entity_tid = sparse.load_npz(save_path + "s_m_tid_entity_tid.npz")
        print("Sparse binary entity commuting matrix loaded.")
        s_m_tid_word_tid = sparse.load_npz(save_path + "s_m_tid_Qid_tid.npz")
        print("Sparse binary word commuting matrix loaded.")

    #将tid-userid矩阵和tid_entity矩阵相加，得到同构关系邻接矩阵
    s_A_tid_tid = s_m_tid_userid_tid + s_m_tid_entity_tid
    del s_m_tid_userid_tid
    del s_m_tid_entity_tid
    # 再加上tid_word矩阵，之后作布尔限定，得到最后的关系邻接矩阵
    s_bool_A_tid_tid = (s_A_tid_tid + s_m_tid_word_tid).astype('bool')  # confirm the connect between tweets
    del s_m_tid_word_tid
    del s_A_tid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'
    all_mins = (time() - all_start) / 60
    print('\tOver all time elapsed: ', all_mins, ' mins\n')
    message += '\tOver all time elapsed: '
    message += str(all_mins)
    message += ' mins\n'

    if save_path is not None:
        sparse.save_npz(save_path + "s_bool_A_tid_tid.npz", s_bool_A_tid_tid)
        print("Sparse binary adjacency matrix saved.")
        #存储在这个s_bool_A_tid_tid中
        s_bool_A_tid_tid = sparse.load_npz(save_path + "s_bool_A_tid_tid.npz")
        print("Sparse binary adjacency matrix loaded.")

    # create corresponding dgl graph
    # 将矩阵转换成图
    G = dgl.from_scipy(s_bool_A_tid_tid)
    # 可视化 DGL 图
    # visualize_graph(G)
    # G = dgl.DGLGraph(s_bool_A_tid_tid)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    # if save_path is not None:
    #     dgl.save_graphs(save_path + 'graph.bin',G)
    #     # dgl.save_graphs(save_path + 'graph.bin', {'graph': G})
    #     print("图保存在：", save_path + 'graph.bin')
    message += 'We have '
    message += str(G.number_of_nodes())
    message += ' nodes.'
    message += 'We have '
    message += str(G.number_of_edges())
    message += ' edges.\n'

    #特征融合
    # 提取节点的特征
    features = np.array([node_embeddings[node] for node in node_to_index])
    print("提取到的节点的形状：",features.shape)
    # 初始化同构图节点的特征
    num_nodes = len(x)
    combined_features = np.zeros((num_nodes, x.shape[1] + features.shape[1]))

    # 进行特征聚合
    # 进行特征聚合
    for node_idx in range(num_nodes):
        node_str = 't_' + str(node_idx)  # 假设同构图节点ID是't_'开头的字符串
        if node_str in node_to_index:
            # 获取同构图节点的初始文本嵌入
            homo_embedding = x[node_idx]
            
            # 获取与该节点相关的异构图节点的嵌入
            related_embeddings = []
            for etype in G.etypes:
                src, dst = G.all_edges(etype=etype)
                neighbors = set(dst[src == node_idx].tolist() + src[dst == node_idx].tolist())
                for nid in neighbors:
                    nid_str = str(nid)
                    if nid_str in node_to_index and nid_str in node_embeddings:
                        related_embeddings.append(node_embeddings[nid_str])
            
            # 进行池化操作，这里使用平均池化
            if related_embeddings:
                pooled_embedding = np.mean(related_embeddings, axis=0)
                # 拼接同构图节点的文本嵌入和异构图相关节点的嵌入池化结果
                combined_features[node_idx] = np.concatenate((homo_embedding, pooled_embedding))
            else:
                # 如果没有相关嵌入，则用零向量填充
                print("没有相关嵌入")
                combined_features[node_idx] = np.concatenate((homo_embedding, np.zeros(features.shape[1])))
        else:
            # 如果节点不在node_to_index中（可能是由于某些节点没有在异构图中），则用自身特征并用零向量填充
            combined_features[node_idx] = np.concatenate((x[node_idx], np.zeros(features.shape[1])))
    print("Shape of 联合特征 after pooling:", combined_features.shape)
    np.save(save_path + 'features.npy', combined_features)
    print("Features saved.")
    # 设置节点特征
    G.ndata['feat'] = torch.FloatTensor(combined_features)

    # all_mins = (time.time() - all_start) / 60
    # print('\tOver all time elapsed: ', all_mins, ' mins\n')
    # message += '\tOver all time elapsed: ' + str(all_mins) + ' mins\n'

    if save_path is not None:
        dgl.save_graphs(save_path + 'graph.bin', G)
        print("图保存在：", save_path + 'graph.bin')




    return all_mins, message

# 将Twitter数据集按日期拆分为21个消息块，使用消息块构造异构社交图，并将其映射为同构消息图
# 要注意的是:
# 1)我们采用最新消息策路(这是最有效的,并给出了最强的性能。有关更多细节,请参见本文第4.4节),因此，每个消息图仅包含该日期的消息，
# 所有先前的消息将从图中删除;要切换到所有消息策略或相关消息策路,将“G = construct_graph_from_df(incr_df)”替换为“G = construct_graph_from_df(incr_df, G)”。
# 2)为了测试目的,设置test=True,消息块和结果消息图都将包含100条消息。
# 要使用所有消息,设置test=False,消息块中的消息数将位于Table之后。论文的第4位。


# Split the Twitter dataset by date into 21 message blocks, use the message blocks to construct heterogeneous social graphs,
# and maps them into homogeneous message graphs.
# Note that:
# 1) We adopt the Latest Message Strategy (which is the most efficient and gives the strongest performance. See Section 4.4 of the paper for more details) here,
# as a consequence, each message graph only contains the messages of the date and all previous messages are removed from the graph;
# To switch to the All Message Strategy or the Relevant Message Strategy, replace 'G = construct_graph_from_df(incr_df)' with 'G = construct_graph_from_df(incr_df, G)'.
# 2) For test purpose, set test=True, and the message blocks, as well as the resulted message graphs each will contain 100 messages.
# To use all the messages, set test=False, and the number of messages in the message blocks will follow Table. 4 of the paper.
def construct_incremental_dataset_0922(df, save_path, features, test=False):  #输入的是文档时间列表、保存路径、还有一个整合特征
    # If test equals true, construct the initial graph using test_ini_size tweets
    # and increment the graph by test_incr_size tweets each day
    test_ini_size = 500
    test_incr_size = 100

    # save data splits for training/validate/test mask generation
    # 保存数据分割用于训练/验证/测试掩码生成
    data_split = []
    # save time spent for the heterogeneous -> homogeneous conversion of each graph
    # 节省每个图从异构到同构的时间
    all_graph_mins = []
    message = ""
    # extract distinct dates
    distinct_dates = df.date.unique()  # 2012-11-07
    # 这行代码将 DataFrame df 中 date 列中的唯一日期值提取出来，并将其存储在名为 distinct_dates 的变量中，这个变量是一个包含了所有唯一日期值的数组或列表。
    print("Distinct dates: ", distinct_dates)
    print("Number of distinct dates: ", len(distinct_dates))
    print()
    message += "Number of distinct dates: "
    message += str(len(distinct_dates))
    message += "\n"

    # split data by dates and construct graphs
    # first week -> initial graph (20254 tweets)
    print("Start constructing initial graph ...")
    message += "\nStart constructing initial graph ...\n"
    # ini_df = df
    ini_df = df.loc[df['date'].isin(distinct_dates[:7])]  # find top 7 dates
    # 从 DataFrame df 中选取出日期列（date）中在 distinct_dates 的前7个唯一日期中出现的行，并将这些行组成一个新的 DataFrame ini_df。
    # extract and save the features of corresponding tweets


    indices = ini_df['index'].values.tolist() # 将index列提取出来保存在indices（索引）中；.tolist()表示将数组变成列表
    x = features[indices, :] # 从名为 features 的变量中按照给定的 indices 索引选择子集，并将选择的子集赋值给变量 x
    
    if test:
        ini_df = ini_df[:test_ini_size]  # top test_ini_size dates 赋值ini_df为前500行数据
    G,node_embeddings,node_to_index = construct_graph_from_df(ini_df,x)  # 生成图数据
    
    path = save_path + '0/'
    os.mkdir(path)   #在生成一个0文件夹
    grap_mins, graph_message = to_dgl_graph_v3(G, node_embeddings, node_to_index, x, save_path=path)  #调用将异构网络图转换为同构网络图的方法，输入为异构图，返回转换时间和消息。
    message += graph_message
    print("Initial graph saved")
    message += "Initial graph saved\n"
    # record the total number of tweets
    data_split.append(ini_df.shape[0])  # 在一个列表data_split中添加一个值，该值是一个DataFrame（ini_df）的行数。
    # record the time spent for graph conversion
    all_graph_mins.append(grap_mins)  # 将时间添加到这个列表中。
    # extract and save the labels of corresponding tweets
    # 提取并保存对应的tweets标签
    y = ini_df['event_id'].values  #从ini_df中提取event_id的值然后添加到y中
    y = [int(each) for each in y]  # 将y中的变量转换为int形
    np.save(path + 'labels.npy', np.asarray(y)) # 将标签存储到本地。（存储的是event_id）
    print("Labels saved.")
    message += "Labels saved.\n"
    # # extract and save the features of corresponding tweets
    # indices = ini_df['index'].values.tolist() # 将index列提取出来保存在indices（索引）中；.tolist()表示将数组变成列表
    # x = features[indices, :] # 从名为 features 的变量中按照给定的 indices 索引选择子集，并将选择的子集赋值给变量 x
    # 表示选择indices列表中的数字所在的行和所有列
    # np.save(path + 'features.npy', x)
    print("Features saved.")
    message += "Features saved.\n\n"
    # np.save(path + 'node_embeddings.npy', node_embeddings)
    # print("node_embeddings saved.")
    # message += "node_embeddings saved.\n\n"

    # subsequent days -> insert tweets day by day (skip the last day because it only contains one tweet)
    for i in range(7, len(distinct_dates) - 1): # 从索引值为7开始，一直到最后一个元素之前的一个范围。
        print("Start constructing graph ", str(i - 6), " ...")
        message += "\nStart constructing graph "
        message += str(i - 6)
        message += " ...\n"
        incr_df = df.loc[df['date'] == distinct_dates[i]] # 从DataFrame df 中选择日期列等于 distinct_dates[i] 的行，
        # 并将结果存储在名为 incr_df 的新DataFrame中。
        if test:
            incr_df = incr_df[:test_incr_size]

        # All/Relevant Message Strategy: keeping all the messages when constructing the graphs
        # (for the Relevant Message Strategy, the unrelated messages will be removed from the graph later on).
        # G = construct_graph_from_df(incr_df, G)

        # Latest Message Strategy: construct graph using only the data of the day
        indices = incr_df['index'].values.tolist()
        x = features[indices, :]
        G,node_embeddings,node_to_index = construct_graph_from_df(incr_df,x)

        path = save_path + str(i - 6) + '/'
        os.mkdir(path)
        grap_mins, graph_message = to_dgl_graph_v3(G, node_embeddings, node_to_index, x, save_path=path) # 将异构图转换为同构图
        message += graph_message
        print("Graph ", str(i - 6), " saved")
        message += "Graph "
        message += str(i - 6)
        message += " saved\n"
        # record the total number of tweets 统计推文数量
        data_split.append(incr_df.shape[0])
        # record the time spent for graph conversion 统计计算时间
        all_graph_mins.append(grap_mins)
        # extract and save the labels of corresponding tweets
        # y = np.concatenate([y, incr_df['event_id'].values], axis = 0)
        # 下面的代码就是给每一个图对应的文件中保存labels和features文件
        y = [int(each) for each in incr_df['event_id'].values] #将索引推文中的event_id转换成int形
        np.save(path + 'labels.npy', y)
        print("Labels saved.")
        message += "Labels saved.\n"
        # extract and save the features of corresponding tweets
        
        # x = np.concatenate([x, x_incr], axis = 0)
        # np.save(path + 'features.npy', x)
        print("Features saved.")
        message += "Features saved.\n"
        # np.save(path + 'node_embeddings.npy', node_embeddings)
        # print("node_embeddings saved.")
        # message += "node_embeddings saved.\n\n"

    return message, data_split, all_graph_mins

# def construct_full_dataset_graph(df):
#     G = construct_graph_from_df(df)
#     all_mins, message, G_dgl = to_dgl_graph_v3(G)
#     return G_dgl


save_path = '/root/cxt/construct_graph_20241125/graph_20250606/'



data_features = np.load('/root/cxt/construct_graph_20241125/data_feature_new.npy', allow_pickle=True)
df = pd.DataFrame(data=data_features, columns=["event_id", "tweet_id", "user_id", "created_at", "hashtags", "user_mentions", "text", "filtered_words", "Entities", "Labels", "language", "Qids"])
print("Data converted to dataframe.")
print(df.shape)



# 将 'created_at' 列从字符串转换为日期时间对象
df['created_at'] = pd.to_datetime(df['created_at'])

# 用时间排序
df = df.sort_values(by='created_at').reset_index() #重设索引为created_time，按照这个进行排序，默认升序（降序用ascending=False）
# 添加date列
df['date'] = [d.date() for d in df['created_at']] #将 DataFrame 中的 created_at 列中的每个日期时间对象转换为日期对象，并将转换后的日期对象存储在新的 date 列中。

print(df['date'].value_counts())

#读入嵌入信息
f = np.load('/root/cxt/construct_graph_20241125/dataset/302dim.npy')
print("录入嵌入维度信息")


# generate test graphs, features, and labels
message, data_split, all_graph_mins = construct_incremental_dataset_0922(df, save_path, f, False)
# 将message写入制定路径下的文件其中"w"表示写入
with open(save_path + "node_edge_statistics.txt", "w") as text_file:
    text_file.write(message)
# 保存切分后的数据
np.save(save_path + 'data_split.npy', np.asarray(data_split))
print("Data split: ", data_split)
np.save(save_path + 'all_graph_mins.npy', np.asarray(all_graph_mins))
print("Time sepnt on heterogeneous -> homogeneous graph conversions: ", all_graph_mins)

# #构图开始
# print('开始构图初始化图——————————————————————————————————————————！！！！！！！！！！')
# G = construct_graph_from_df(df)
# print('异构图已经完成，开始转换为同构图——————————————————————————————————！！！！！！！！！！')
# all_mins, message, G_dgl = to_dgl_graph_v3(G,save_path='/root/cxt/construct_graph/graph/')

# #写入总共时间和message信息

# # 将message写入制定路径下的文件其中"w"表示写入
# with open(save_path + "node_edge_statistics.txt", "w") as text_file:
#     text_file.write(message)

# # np.save(save_path + 'dataset/all_graph_mins.npy', np.asarray(all_mins))
# print("Time sepnt on heterogeneous -> homogeneous graph conversions: ", all_mins)



