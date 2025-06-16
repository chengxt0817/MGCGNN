from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False):
        super(GATLayer, self).__init__()
        # equation (1) reference: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
        # 初始化全连接层，用于特征映射
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    #修改的代码
    def forward(self, blocks, layer_id):
        h = blocks[layer_id].srcdata['features']
        z = self.fc(h)
        blocks[layer_id].srcdata['z'] = z
        z_dst = z[:blocks[layer_id].number_of_dst_nodes()]

        blocks[layer_id].dstdata['z'] = z_dst
        blocks[layer_id].apply_edges(self.edge_attention)
        # equation (3) & (4)
        blocks[layer_id].update_all(  # block_id – The block to run the computation.
                         self.message_func,  # Message function on the edges.
                         self.reduce_func)  # Reduce function on the node.

        # nf.layers[layer_id].data.pop('z')
        # nf.layers[layer_id + 1].data.pop('z')

        if self.use_residual:
            return z_dst + blocks[layer_id].dstdata['h']  # residual connection
        return blocks[layer_id].dstdata['h']


    #修改完毕


    # def forward(self, nf, layer_id):
    #     h = nf.layers[layer_id].data['h']
    #     # equation (1)
    #     z = self.fc(h)
    #     nf.layers[layer_id].data['z'] = z
    #     # print("test test test")
    #     A = nf.layer_parent_nid(layer_id)
    #     # print(A)
    #     # print(A.shape)
    #     A = A.unsqueeze(-1)
    #     B = nf.layer_parent_nid(layer_id + 1)
    #     # print(B)
    #     # print(B.shape)
    #     B = B.unsqueeze(0)

    #     _, indices = torch.topk((A == B).int(), 1, 0)
    #     # print(indices)
    #     # print(indices.shape)
    #     # indices = np.asarray(indices)
    #     indices = indices.cpu().data.numpy()

    #     nf.layers[layer_id + 1].data['z'] = z[indices]
    #     # print(nf.layers[layer_id+1].data['z'].shape)
    #     # equation (2)
    #     nf.apply_block(layer_id, self.edge_attention)
    #     # equation (3) & (4)
    #     nf.block_compute(layer_id,  # block_id _ The block to run the computation.
    #                      self.message_func,  # Message function on the edges.
    #                      self.reduce_func)  # Reduce function on the node.

    #     nf.layers[layer_id].data.pop('z')
    #     nf.layers[layer_id + 1].data.pop('z')

    #     if self.use_residual:
    #         return z[indices] + nf.layers[layer_id + 1].data['h']  # residual connection
    #     return nf.layers[layer_id + 1].data['h']


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual))
        self.merge = merge

    def forward(self, nf, layer_id):
        head_outs = [attn_head(nf, layer_id) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1) # 沿着第二个维度进行拼接，所以最后输出到结果为（batch_size,num_heads * out_dim）
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs)) #按照新的维度进行堆叠,不拼接就取平均值


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)

    # 修改的代码
    def forward(self, blocks):
        h = self.layer1(blocks, 0)
        h = F.elu(h)
        # print(h.shape)
        blocks[1].srcdata['features'] = h
        h = self.layer2(blocks, 1)
        #h = F.normalize(h, p=2, dim=1)
        return h

    # 修改完毕



    # nf的形状是G
    # def forward(self, nf, corrupt=False):
    #     features = nf.layers[0].data['features']
    #     if corrupt: # 是否打乱
    #         nf.layers[0].data['h'] = features[torch.randperm(features.size()[0])]
    #     else:
    #         nf.layers[0].data['h'] = features
    #     # 传入图层0进行特征拼接，得到（batch_size,num_heads * out_dim）的特征
    #     h = self.layer1(nf, 0)
    #     # 将变量 h 应用 ELU 激活函数，将负值区域的值进行变换，从而产生更平滑的输出。
    #     h = F.elu(h)
    #     # print(h.shape)
    #     # 将第一层的节点特征输出作为第二层的图特征输入
    #     nf.layers[1].data['h'] = h
    #     h = self.layer2(nf, 1)
    #     return h

# Applies an average on seq, of shape (nodes, features)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)
        # print("testing, shape of logits: ", logits.size())
        return logits


class DGI(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(DGI, self).__init__()
        self.gat = GAT(in_dim, hidden_dim, out_dim, num_heads, use_residual)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim)

    def forward(self, nf):
        h_1 = self.gat(nf, False)
        c = self.read(h_1)
        c = self.sigm(c)
        h_2 = self.gat(nf, True)
        ret = self.disc(c, h_1, h_2)
        return h_1, ret

    # Detach the return variables
    def embed(self, nf):
        h_1 = self.gat(nf, False)
        return h_1.detach()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        # 这里是选出对应的三元组矩阵
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        # 从三元组矩阵中选取对应的冒点对应的词嵌入，然后和正样本进行求欧式距离然后求和，求出来是和正样本的距离平方和
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        # 从三元组矩阵中选取对应的冒点对应的词嵌入，然后和负样本进行求欧式距离然后求和，求出来是和负样本的距离平方和
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        # 然后根据三元组损失公式，正距离减负距离加上偏置进行激活函数，获得一个损失矩阵，
        # 这里使用了ReLU函数来确保损失始终为非负值，即只有当ap_distances - an_distances + self.margin大于零时才会有损失，否则损失为零。
        losses = F.relu(ap_distances - an_distances + self.margin)
        # 返回样本的损失值的平均值和三元组矩阵的长度
        return losses.mean(), len(triplets)


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        # 使用 pdist 函数计算嵌入向量之间的距离矩阵。pdist 通常用于计算一组向量之间的欧氏距离，返回一个距离矩阵。
        distance_matrix = pdist(embeddings)
        # 如果 self.cpu 为真，则将计算得到的距离矩阵转移到 CPU 上进行处理。
        distance_matrix = distance_matrix.cpu()
        # 将标签 labels 转移到 CPU 上，并将其转换为 NumPy 数组。
        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            # 创建一个标签掩码，用于确定属于当前标签的样本。
            label_mask = (labels == label)
            # 使用 np.where() 函数找到掩码中值为 True 的位置索引，也就是当前标签的所有样本在原始数据中的索引。
            # 然后，取其中的第一个返回值 [0]，以获取这些样本的索引数组。
            label_indices = np.where(label_mask)[0]   # 正样本
            # 如果某个标签对应的样本数量小于2，那么这个标签就无法构成正样本对（anchor-positive pair），
            # 因为至少需要两个样本才能形成一对。因此，当发现某个标签的样本数量小于2时，代码会跳过该标签，继续处理下一个标签。
            if len(label_indices) < 2:
                continue
            # 首先，label_mask是一个布尔数组，其中True表示样本属于当前标签，False表示不属于当前标签。
            # 然后，np.logical_not(label_mask)会将True转换为False，False转换为True，即将标签掩码取反，
            # 得到的数组表示不属于当前标签的样本。最后，np.where()函数找到所有为True的索引，
            # 即得到了所有不属于当前标签的样本的索引，这些索引就是负样本的索引。
            negative_indices = np.where(np.logical_not(label_mask))[0]  # 负样本
            # 生成这个索引列表中的所有的正样本的两两组合
            # 将生成的组合转换为列表的函数。在这里，将生成的所有组合转换为列表的目的是为了方便后续处理。
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            # 将列表转换为NumPy数组。
            anchor_positives = np.array(anchor_positives)
            # anchor_positives[:, 0]表示取所有anchor-positive对中的第一个样本的索引，
            # anchor_positives[:, 1]表示取所有anchor-positive对中的第二个样本的索引。
            # 然后，这些索引被用来从距离矩阵 distance_matrix 中获取相应的距离值，即两个样本之间的距离。
            # 这样，ap_distances 中就存储了所有anchor-positive对之间的距离。
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]

            # 对于每个anchor-positive对及其对应的距离，执行下面的操作
            # 是一个内置函数，它接受多个可迭代对象作为参数，并返回一个迭代器，该迭代器生成由每个可迭代对象中对应位置的元素组成的元组
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                # 表示获取 anchor-positive 对的第一个样本与所有负样本之间的距离。
                # 这里通过索引操作获取了特定行和特定列上的元素，得到的是一个包含了所有这些距离的向量。
                # 是一个常数，表示边界值或者说阈值，用于控制损失的界限
                # 表示获取 anchor-positive 对的第一个样本与所有负样本之间的距离。
                # 这里通过索引操作获取了特定行和特定列上的元素，得到的是一个包含了所有这些距离的向量。
                # loss_values 就是每个 anchor-positive 对与其所有负样本之间的损失值的向量。
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy() # 损失值在cpu处理转换为numpy数组
                hard_negative = self.negative_selection_fn(loss_values)  # 这里返回的是一个索引
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative] # 这里变成一个值
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets) # 这里将三元组变成一个矩阵

        return torch.LongTensor(triplets)


def random_hard_negative(loss_values):
    # 查找满足条件 loss_values > 0 的元素的索引
    hard_negatives = np.where(loss_values > 0)[0]
    # 从 hard_negatives 中随机选择一个元素（即一个索引），如果 hard_negatives 不为空的话。
    # 如果 hard_negatives 为空，即没有满足条件的损失值大于 0 的元素，那么返回 None。
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


# 使用 np.argmax() 函数找到损失值列表中的最大值，并返回其索引值，即对应的样本。
# 如果最大的损失值大于0，则返回找到的最大损失值的索引，表示该样本为“最难”的负样本。
# 如果最大的损失值不大于0，表示没有明显的“最难”的负样本，那么返回 None。
def hardest_negative(loss_values):
    # 这行代码通过 np.argmax() 函数找到了 loss_values 中最大值所对应的索引，即最困难的负样本的索引。
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                              negative_selection_fn=hardest_negative,
                                                                                              cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)
