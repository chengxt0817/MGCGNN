import numpy as np
import json
import argparse
from torch.utils.data import Dataset
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
from metrics import AverageNonzeroTripletsMetric
import time
from time import localtime, strftime
import os
import pickle
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn import metrics
from utils import SocialDataset
from utils import *
from main import args_define
from layers import *

"""
    KPGNN Model
    Paper: Knowledge-Preserving Incremental Social Event Detection via Heterogeneous GNNs
    Author: Yuwei Cao et al.
    github: https://github.com/RingBDStack/KPGNN
"""

class KPGNN():
    # Inference(prediction)
    def infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, train_indices=None, model=None,
            loss_fn_dgi=None, indices_to_remove=[]):
        args = args_define.args
        # make dir for graph i
        save_path_i = embedding_save_path + '/block_' + str(i)
        if not os.path.isdir(save_path_i):
            os.mkdir(save_path_i)

        # load data
        data = SocialDataset(args.data_path, i)
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        in_feats = features.shape[1]  # feature dimension
        print(in_feats)

        # Construct graph
        g = dgl.DGLGraph(data.matrix)
        # graph that contains message blocks 0, ..., i if remove_obsolete = 0 or 1; graph that only contains message block i if remove_obsolete = 2
        num_isolated_nodes = graph_statistics(g, save_path_i)

        # if remove_obsolete is mode 1, resume or use the passed indices_to_remove to remove obsolete nodes from the graph
        if args.remove_obsolete == 1:

            if ((args.resume_path is not None) and (not args.resume_current) and (i == args.resume_point + 1) and (i > args.window_size)) \
                    or (indices_to_remove == [] and i > args.window_size):  # Resume indices_to_remove from the last maintain block

                temp_i = max(((i - 1) // args.window_size) * args.window_size, 0)
                indices_to_remove = np.load(
                    embedding_save_path + '/block_' + str(temp_i) + '/indices_to_remove.npy').tolist()

            if indices_to_remove != []:
                # remove obsolete nodes from the graph
                data.remove_obsolete_nodes(indices_to_remove)
                features = torch.FloatTensor(data.features)
                labels = torch.LongTensor(data.labels)
                # Reconstruct graph
                g = dgl.DGLGraph(data.matrix)  # graph that contains tweet blocks 0, ..., i
                num_isolated_nodes = graph_statistics(g, save_path_i)

        # generate or load test mask
        if args.mask_path is None:
            mask_path = save_path_i + '/masks'
            if not os.path.isdir(mask_path):
                os.mkdir(mask_path)
            test_indices = generateMasks(len(labels), data_split, train_i, i, args.validation_percent, mask_path, len(indices_to_remove))
        else:
            test_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/test_indices.pt')

        # Suppress warning
        g.set_n_initializer(dgl.init.zero_initializer)
        g.readonly()

        device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
        if args.use_cuda:
            g = g.to('cuda:0')
            features, labels = features.cuda(), labels.cuda()
            test_indices = test_indices.cuda()

        g.ndata['features'] = features
        g.ndata['labels'] = labels


        if (args.resume_path is not None) and (not args.resume_current) and (
                i == args.resume_point + 1):  # Resume model from the previous block and train_indices from the last initil/maintain block

            # Declare model
            if args.use_dgi:
                model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
            else:
                model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

            if args.use_cuda:
                model.cuda()

            # Load model from resume_point
            model_path = embedding_save_path + '/block_' + str(args.resume_point) + '/models/best.pt'
            model.load_state_dict(torch.load(model_path))
            print("Resumed model from the previous block.")

            # Use resume_path as a flag
            args.resume_path = None

        if train_indices is None:  # train_indices is used for continue training then predict
            if args.remove_obsolete == 0 or args.remove_obsolete == 1:
                # Resume train_indices from the last initil/maintain block
                temp_i = max(((i - 1) // args.window_size) * args.window_size, 0)
                train_indices = torch.load(embedding_save_path + '/block_' + str(temp_i) + '/masks/train_indices.pt')
            else:
                if args.n_infer_epochs != 0:
                    print("==================================\n'continue training then predict' is unimplemented under remove_obsolete mode 2, will skip infer epochs.\n===================================\n")
                    args.n_infer_epochs = 0

        # record test nmi of all epochs
        all_test_nmi = []
        # record the time spent in seconds on direct prediction
        time_predict = []

        # Directly predict
        message = "\n------------ Directly predict on block " + str(i) + " ------------\n"
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        start = time.time()
        # Infer the representations of all tweets
        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args, labels)
        # Predict: conduct kMeans clustering on the test(newly inserted) nodes and report NMI
        test_nmi = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes, save_path_i, args, False)
        seconds_spent = time.time() - start
        message = '\nDirect prediction took {:.2f} seconds'.format(seconds_spent)
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        all_test_nmi.append(test_nmi)
        time_predict.append(seconds_spent)
        np.save(save_path_i + '/time_predict.npy', np.asarray(time_predict))

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Continue training then predict (leverage the features and structural info of the newly inserted block)
        if args.n_infer_epochs != 0:
            message = "\n------------ Continue training then predict on block " + str(i) + " ------------\n"
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
        # record the time spent in seconds on each batch of all infer epochs
        seconds_infer_batches = []
        # record the time spent in mins on each epoch
        mins_infer_epochs = []
        for epoch in range(args.n_infer_epochs):
            start_epoch = time.time()
            losses = []
            total_loss = 0
            if args.use_dgi:
                losses_triplet = []
                losses_dgi = []
            for metric in metrics:
                # 重制度量指标
                metric.reset()



            #开始修改代码
            extract_features, extract_labels = extract_embeddings(g, model, len(labels), args, labels)
            # print(extract_features)
            # label_center = {}
            # for l in set(extract_labels):
            #     l_indices = np.where(extract_labels==l)[0]
            #     l_feas = extract_features[l_indices]
            #     l_cen = np.mean(l_feas,0)
            #     label_center[l] = l_cen


            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, train_indices, sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                )
            

            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
                blocks = [b.to(device) for b in blocks]
                batch_labels = blocks[-1].dstdata['labels']

                start_batch = time.time()
                model.train()
                # forward
                pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).


                #修改
                

                # loss_outputs = loss_fn(pred, batch_labels)
                # loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                if args.add_pair:
                    batch_labels_set = set(batch_labels)
                    label_center, cen_keys, cen_metrics, cen_values = get_cen_metric(batch_labels_set,batch_labels,pred)
                    cen_pred_key = np.array([cen_keys[l] for l in batch_labels.cpu().data.numpy()])
                    cen_metrics = cen_values.detach().cpu().numpy()[cen_pred_key]  # 从label_center获取cen_values

                    # 获取任意两个向量
                    cen_pred_metrics = cen_metrics - pred.detach().cpu().numpy()
                    pre_pre_metric = (pred.unsqueeze(1) - pred.unsqueeze(0)).detach().cpu().numpy()
                    cen_pred_metrics = np.tile(np.expand_dims(cen_pred_metrics, axis=1), (1, len(batch_labels), 1))
                    pre_pre_dot_metric = np.sum(
                        (cen_pred_metrics / ((np.linalg.norm(cen_pred_metrics, ord=2, axis=-1, keepdims=True)) + 1e-2)) * (
                            pre_pre_metric / (((np.linalg.norm(pre_pre_metric, ord=2, axis=-1, keepdims=True))) + 1e-2)),
                        axis=-1)
                    pairs, pair_labels, pair_matrix = pairwise_sample(pre_pre_dot_metric, batch_labels)

                    pairs = pairs.cuda()
                    pair_matrix = pair_matrix.cuda()
                    pos_indices = torch.where(pair_labels > 0)[0].cuda()
                    neg_indices = torch.where(pair_labels < 0)[0].cuda()

                    max_val = neg_indices.shape[0]

                    if max_val < 100:
                        dis = torch.empty([0, 1]).cuda()
                        for l in set(batch_labels.cpu().data.numpy()):
                            label_indices = torch.where(batch_labels == l)[0]
                            l_center = torch.FloatTensor(label_center[l]).cuda()
                            # 确保 label_indices 在 CPU 上
                            label_indices = label_indices.cpu().numpy()
                            dis_l = (pred[label_indices].cpu() - l_center.cpu()).pow(2).sum(1).unsqueeze(-1)
                            dis = torch.cat([dis, dis_l.cuda()], 0)
                        # 确保 cen_metrics 是 PyTorch 张量并移动到 GPU
                        cen_metrics_tensor = torch.tensor(cen_metrics).cuda()
                        loss = torch.clamp(dis.mean() - cen_metrics_tensor.mean() + 16, min=0.0)
                    else:
                        pairs = pairs.cuda()
                        max_sample = max_val if max_val > pos_indices.shape[0] else pos_indices.shape[0]
                        neg_ind = torch.randint(0, neg_indices.shape[0], [max_sample]).cuda()
                        pos_ind = torch.randint(0, pos_indices.shape[0], [max_sample]).cuda()


                        neg_dis = (pred[pairs[neg_indices[neg_ind], 0]] - pred[pairs[neg_indices[neg_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
                        pos_dis = (pred[pairs[pos_indices[pos_ind], 0]] - pred[pairs[pos_indices[pos_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
                        pairs_indices = torch.where(torch.clamp(pos_dis + 16 - neg_dis, min=0.0) > 0)
                        loss = torch.mean(torch.clamp(pos_dis + 16 - neg_dis, min=0.0)[pairs_indices[0]])

                    pred = F.normalize(pred, 2, 1)
                    pair_out = torch.mm(pred, pred.t())
                    if args.add_ort:
                        pair_loss = (pair_matrix - pair_out).pow(2).mean()
                        print("pair loss:", loss.item(), "pair orthogonal loss:  ", 100 * pair_loss.item())
                        loss += 100 * pair_loss

                losses.append(loss.item())
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_seconds_spent = time.time() - start_batch
                seconds_infer_batches.append(batch_seconds_spent)
                # end one batch
            #修改代码结束

            #region
            # g: 图数据，表示待采样的图。
            # args.batch_size: 批量大小，即每个批次中包含的样本数量。
            # args.n_neighbors: 每个节点采样的邻居数量。
            # neighbor_type='in': 邻居类型，这里设置为 'in'，表示采样节点的入边邻居。
            # shuffle=True: 是否对节点进行随机采样。
            # num_workers=32: 并行处理的工作线程数，用于加速采样过程。
            # num_hops=2: 采样的邻居跳数，即在邻居采样中遍历的跳数。
            # seed_nodes=train_indices: 设置种子节点，即从这些节点开始采样。
            # for batch_id, nf in enumerate(dgl.contrib.sampling.NeighborSampler(g,
            #                                                                 args.batch_size,
            #                                                                 args.n_neighbors,
            #                                                                 neighbor_type='in',
            #                                                                 shuffle=True,
            #                                                                 num_workers=32,
            #                                                                 num_hops=2,
            #                                                                 seed_nodes=train_indices)):
            #     start_batch = time.time()
            #     # 将从父图中采样的图复制到子图中
            #     nf.copy_from_parent()
            #     model.train()
            #     # 开始调用forward函数


            #     # forward
            #     if args.use_dgi:
            #         pred, ret = model(nf)  # pred: representations of the sampled nodes (in the last layer of the NodeFlow), ret: discriminator results
            #     else:
            #         # 这里的pred是输出的图权重表示矩阵
            #         pred = model(nf)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
            #         # 获取采样到的小批量数据在父图中的节点ID，获取到的节点ID张量转移到GPU上，并将其数据类型设置为torch.long。
            #         # 这样处理后，batch_nids就是包含采样到的小批量数据在父图中节点ID的张量。
            #         batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
            #         # 提取对应的标签列表
            #         batch_labels = labels[batch_nids]
            #         # 得出损失的矩阵
            #         # loss_fn 变量将是一个包含两个值的元组。
            #         # 你可以通过索引来访问这两个值loss_outputs[0]:损失的平均值； loss_outputs[1]:三元组的长度
            #         loss_outputs = loss_fn(pred, batch_labels)
            #         # 检查第一个元组是不是列表类型，然后就直接负值给loss，就是损失平均值
            #         loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            #     if args.use_dgi:
            #         n_samples = nf.layer_nid(-1).size()[0]
            #         lbl_1 = torch.ones(n_samples)
            #         lbl_2 = torch.zeros(n_samples)
            #         lbl = torch.cat((lbl_1, lbl_2), 0)
            #         if args.use_cuda:
            #             lbl = lbl.cuda()
            #         losses_triplet.append(loss.item())
            #         loss_dgi = loss_fn_dgi(ret, lbl)
            #         losses_dgi.append(loss_dgi.item())
            #         loss += loss_dgi
            #         losses.append(loss.item())
            #     else:
            #         losses.append(loss.item())
            #     total_loss += loss.item()

            #     for metric in metrics:
            #         # 记录每一次的三元组损失的长度信息
            #         metric(pred, batch_labels, loss_outputs)

            #     if batch_id % args.log_interval == 0:
            #         message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #             batch_id * args.batch_size, train_indices.shape[0],
            #             100. * batch_id / (train_indices.shape[0] // args.batch_size), np.mean(losses))
            #         if args.use_dgi:
            #             message += '\tLoss_triplet: {:.6f}'.format(np.mean(losses_triplet))
            #             message += '\tLoss_dgi: {:.6f}'.format(np.mean(losses_dgi))
            #         for metric in metrics:
            #             message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            #         print(message)
            #         with open(save_path_i + '/log.txt', 'a') as f:
            #             f.write(message)
            #         losses = []

            #     # 将优化器中的梯度清零，以便接收新的梯度信息
            #     optimizer.zero_grad()
            #     # 根据当前损失值，执行反向传播，计算模型参数的梯度
            #     loss.backward()
            #     # 根据梯度更新模型参数，通过调用优化器的 step() 方法实现参数更新
            #     optimizer.step()

            #     batch_seconds_spent = time.time() - start_batch
            #     seconds_infer_batches.append(batch_seconds_spent)
            #     # end one batch
            #endregion


            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_infer_epochs, total_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            mins_spent = (time.time() - start_epoch) / 60
            message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
            message += '\n'
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            mins_infer_epochs.append(mins_spent)

            # Validation
            # Infer the representations of all tweets
            extract_features, extract_labels = extract_embeddings(g, model, len(labels), args, labels)
            # Save the representations of all tweets
            np.save(save_path_i + '/features_' + str(epoch) + '.npy', extract_features)
            np.save(save_path_i + '/labels_' + str(epoch) + '.npy', extract_labels)
            # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
            # Evaluate the model: conduct kMeans clustering on the validation and report NMI
            test_nmi = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes, save_path_i, args,
                                False)
            all_test_nmi.append(test_nmi)
            # end one epoch




            # Save model (fine-tuned from the above continue training process)
            model_path = save_path_i + '/models'
            os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)
            print('Model saved.')

        # Save all test nmi
        np.save(save_path_i + '/all_test_nmi.npy', np.asarray(all_test_nmi))
        print('Saved all test nmi.')
        # Save time spent on epochs
        np.save(save_path_i + '/mins_infer_epochs.npy', np.asarray(mins_infer_epochs))
        print('Saved mins_infer_epochs.')
        # Save time spent on batches
        np.save(save_path_i + '/seconds_infer_batches.npy', np.asarray(seconds_infer_batches))
        print('Saved seconds_infer_batches.')

        return model


    # Train on initial/maintenance graphs, t == 0 or t % window_size == 0 in this paper
    def initial_maintain(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model=None, loss_fn_dgi=None):
        args = args_define.args
        # make dir for graph i
        save_path_i = embedding_save_path + '/block_' + str(i)
        if not os.path.isdir(save_path_i):
            os.mkdir(save_path_i)

        # load data
        # 加载数据包括（同构图、特征、标签）自己写的一个获取这些数据的函数在utils.py中
        data = SocialDataset(args.data_path, i)

        # 存储特征
        features = torch.FloatTensor(data.features)
        print("原始特征的形状：",features.shape)
        # node_features = torch.FloatTensor(data.node_features)
        # print("异构图特征的形状：",node_features.shape)
        # combined_features = np.hstack((features, node_features))
        features = torch.FloatTensor(features)
        # 存储标签
        labels = torch.LongTensor(data.labels)
        # 计算了输入特征的列数，并将结果存储在变量 in_feats 中。存储了特征维度，这个长度或维度通常用于定义神经网络的输入层的大小，以确保网络结构的正确匹配
        in_feats = features.shape[1]  # feature dimension 300维获取的是维度数量，用于定义神经网络的输入层的大小
        print(in_feats)

        # Construct graph that contains message blocks 0, ..., i if remove_obsolete = 0 or 1; graph that only contains message block i if remove_obsolete = 2
        g = dgl.DGLGraph(data.matrix)    # 将邻接矩阵转换为图对象，然后存储在g中

        # 统计图g中的孤立节点数量，save_path_i用于指定结果保存的路径或文件名
        num_isolated_nodes = graph_statistics(g, save_path_i)

        # if remove_obsolete is mode 1, resume or generate indices_to_remove, then remove obsolete nodes from the graph
        if args.remove_obsolete == 1: # 如果是1那么就采用最新消息策略，删除过时节点。（孤立节点）
            
            # Resume indices_to_remove from the current block 从当前块中恢复索引
            if (args.resume_path is not None) and args.resume_current and (i == args.resume_point) and (i != 0):
                # 将需要移除的索引从文件中读出然后存储在变量indices_to_remove中转换成列表的形式
                indices_to_remove = np.load(save_path_i + '/indices_to_remove.npy').tolist()

            elif i == 0:  # generate empty indices_to_remove for initial block
                indices_to_remove = []
                # save indices_to_remove
                np.save(save_path_i + '/indices_to_remove.npy', np.asarray(indices_to_remove))

            #  update graph
            else:  # generate indices_to_remove for maintenance block
                # get the indices of all training nodes
                # 遍历所有的训练节点统计当前训练节点数量
                num_all_train_nodes = np.sum(data_split[:i + 1])
                # 创建了一个包含当前所有训练数据的一个索引，也就是从零开始一直到节点数量-1这么多个索引
                all_train_indices = np.arange(0, num_all_train_nodes).tolist()
                # get the number of old training nodes added before this maintenance
                num_old_train_nodes = np.sum(data_split[:i + 1 - args.window_size])
                # indices_to_keep: indices of nodes that are connected to the new training nodes added at this maintenance
                # (include the indices of the new training nodes)
                # 提取了 data.matrix 中所有行号大于等于 num_old_train_nodes 的行所对应的非零元素的列索引。set是为了去除重复索引
                indices_to_keep = list(set(data.matrix.indices[data.matrix.indptr[num_old_train_nodes]:]))
                # indices_to_remove is the difference between the indices of all training nodes and indices_to_keep
                # 找到需要去除的索引
                indices_to_remove = list(set(all_train_indices) - set(indices_to_keep))
                # save indices_to_remove
                np.save(save_path_i + '/indices_to_remove.npy', np.asarray(indices_to_remove))

            if indices_to_remove != []:
                # remove obsolete nodes from the graph
                data.remove_obsolete_nodes(indices_to_remove)
                features = torch.FloatTensor(data.features)
                labels = torch.LongTensor(data.labels)
                # Reconstruct graph
                g = dgl.DGLGraph(data.matrix)  # graph that contains tweet blocks 0, ..., i
                num_isolated_nodes = graph_statistics(g, save_path_i)
        # 采用最新消息策略
        else:

            indices_to_remove = []

        # generate or load training/validate/test masks
        if (args.resume_path is not None) and args.resume_current and (
                i == args.resume_point):  # Resume masks from the current block

            train_indices = torch.load(save_path_i + '/masks/train_indices.pt')
            validation_indices = torch.load(save_path_i + '/masks/validation_indices.pt')
        if args.mask_path is None:

            mask_path = save_path_i + '/masks'
            if not os.path.isdir(mask_path):
                os.mkdir(mask_path)
            # 生成了训练索引还有验证索引，用于训练模型
            train_indices, validation_indices = generateMasks(len(labels), data_split, train_i, i, args.validation_percent,
                                                            mask_path, len(indices_to_remove))

        else:
            train_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/train_indices.pt')
            validation_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/validation_indices.pt')

        # Suppress warning
        # 对tug设置节点初始化器
        g.set_n_initializer(dgl.init.zero_initializer)
        # 只读模式
        g.readonly()

        device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
        if args.use_cuda:
            # 将特征和标签转移到gpu上运行
            g = g.to(device)
            features, labels = features.cuda(), labels.cuda()
            train_indices, validation_indices = train_indices.cuda(), validation_indices.cuda()
        # 将变量 features 赋值给了图 g 中的节点属性 'features'。
        g.ndata['features'] = features
        g.ndata['labels'] = labels

        # 如果是继续在对应的block进行操作
        # 如果resume_path不空就进行模型维护操作
        if (args.resume_path is not None) and args.resume_current and (
                i == args.resume_point):  # Resume model from the current block

            # Declare model
            if args.use_dgi:
                model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
            else:
                model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

            if args.use_cuda:
                model.cuda()

            # Load model from resume_point
            model_path = embedding_save_path + '/block_' + str(args.resume_point) + '/models/best.pt'
            model.load_state_dict(torch.load(model_path))
            print("Resumed model from the current block.")

            # Use resume_path as a flag
            args.resume_path = None

        elif model is None:  # Construct the initial model
            # Declare model
            if args.use_dgi:
                model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
            else:
                # in_feats：嵌入维度（300）； hidden_dim:隐藏维度（8）； out_dim：输出维度（32）； num_heads：头数量（4）； use_residual：是否使用残差连接（yes）
                model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

            if args.use_cuda:
                model.cuda()

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Start training
        message = "\n------------ Start initial training / maintaining using blocks 0 to " + str(i) + " ------------\n"
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        # record the highest validation nmi ever got for early stopping
        best_vali_nmi = 1e-9
        best_epoch = 0
        wait = 0
        # record validation nmi of all epochs before early stop
        all_vali_nmi = []
        # record the time spent in seconds on each batch of all training/maintaining epochs
        seconds_train_batches = []
        # record the time spent in mins on each epoch
        mins_train_epochs = []
        for epoch in range(args.n_epochs):
            start_epoch = time.time()
            losses = []
            total_loss = 0
            if args.use_dgi:
                losses_triplet = []
                losses_dgi = []
            for metric in metrics:
                metric.reset()


            #   添加代码在这里
            # extract_features, extract_labels = extract_embeddings(g, model, len(labels), args, labels)
            # print(extract_features)
            # label_center = {}
            # for l in set(extract_labels):
            #     l_indices = np.where(extract_labels==l)[0]
            #     l_feas = extract_features[l_indices]
            #     l_cen = np.mean(l_feas,0)
            #     label_center[l] = l_cen


            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, train_indices, sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                )
            

            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
                blocks = [b.to(device) for b in blocks]
                batch_labels = blocks[-1].dstdata['labels']

                start_batch = time.time()
                model.train()
                # forward
                pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).


                #修改
                

                # loss_outputs = loss_fn(pred, batch_labels)
                # loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                if args.add_pair:
                    batch_labels_set = set(batch_labels)
                    label_center, cen_keys, cen_metrics, cen_values = get_cen_metric(batch_labels_set,batch_labels,pred)
                    cen_pred_key = np.array([cen_keys[l] for l in batch_labels.cpu().data.numpy()])
                    cen_metrics = cen_values.detach().cpu().numpy()[cen_pred_key]  # 从label_center获取cen_values

                    # 获取任意两个向量
                    cen_pred_metrics = cen_metrics - pred.detach().cpu().numpy()
                    pre_pre_metric = (pred.unsqueeze(1) - pred.unsqueeze(0)).detach().cpu().numpy()
                    cen_pred_metrics = np.tile(np.expand_dims(cen_pred_metrics, axis=1), (1, len(batch_labels), 1))
                    pre_pre_dot_metric = np.sum(
                        (cen_pred_metrics / ((np.linalg.norm(cen_pred_metrics, ord=2, axis=-1, keepdims=True)) + 1e-2)) * (
                            pre_pre_metric / (((np.linalg.norm(pre_pre_metric, ord=2, axis=-1, keepdims=True))) + 1e-2)),
                        axis=-1)
                    pairs, pair_labels, pair_matrix = pairwise_sample(pre_pre_dot_metric, batch_labels)

                    pairs = pairs.cuda()
                    pair_matrix = pair_matrix.cuda()
                    pos_indices = torch.where(pair_labels > 0)[0].cuda()
                            
                    print("正样本数组大小为：",len(pos_indices))
                    neg_indices = torch.where(pair_labels < 0)[0].cuda()
                    print("负样本数组大小为：",len(neg_indices))

                    max_val = neg_indices.shape[0]

                    if max_val < 100:
                        dis = torch.empty([0, 1]).cuda()
                        for l in set(batch_labels.cpu().data.numpy()):
                            label_indices = torch.where(batch_labels == l)
                            l_center = label_center[l]
                            dis_l = (pred[label_indices] - l_center).pow(2).sum(1).unsqueeze(-1)
                            dis = torch.cat([dis, dis_l], 0)
                        # 确保 cen_metrics 是 PyTorch 张量并移动到 GPU
                        cen_metrics_tensor = torch.tensor(cen_metrics).cuda()
                        loss = torch.clamp(dis.mean() - cen_metrics_tensor.mean() + 16, min=0.0)
                    else:
                        pairs = pairs.cuda()
                        max_sample = max_val if max_val > pos_indices.shape[0] else pos_indices.shape[0]
                        neg_ind = torch.randint(0, neg_indices.shape[0], [max_sample]).cuda()
                        pos_ind = torch.randint(0, pos_indices.shape[0], [max_sample]).cuda()


                        neg_dis = (pred[pairs[neg_indices[neg_ind], 0]] - pred[pairs[neg_indices[neg_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
                        pos_dis = (pred[pairs[pos_indices[pos_ind], 0]] - pred[pairs[pos_indices[pos_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
                        pairs_indices = torch.where(torch.clamp(pos_dis + 16 - neg_dis, min=0.0) > 0)
                        loss = torch.mean(torch.clamp(pos_dis + 16 - neg_dis, min=0.0)[pairs_indices[0]])

                    pred = F.normalize(pred, 2, 1)
                    pair_out = torch.mm(pred, pred.t())
                    if args.add_ort:
                        pair_loss = (pair_matrix - pair_out).pow(2).mean()
                        print("pair loss:", loss.item(), "pair orthogonal loss:  ", 100 * pair_loss.item())
                        loss += 100 * pair_loss

                losses.append(loss.item())
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_seconds_spent = time.time() - start_batch
                seconds_train_batches.append(batch_seconds_spent)
                #修改完毕


                #region
                # loss_outputs = loss_fn(pred, batch_labels)
                # loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                # dis = torch.empty([0, 1]).cuda()
                # for l in set(batch_labels.cpu().data.numpy()):
                #     label_indices = torch.where(batch_labels==l)
                #     l_center = torch.FloatTensor(label_center[l]).cuda()
                #     dis_l = (pred[label_indices] - l_center).pow(2).sum(1).unsqueeze(-1)
                #     dis = torch.cat([dis,dis_l],0)

                # if args.add_pair:
                #     pairs, pair_labels, pair_matrix = pairwise_sample(pred, batch_labels)
                #     if args.use_cuda:
                #         pairs = pairs.cuda()
                #         pair_matrix = pair_matrix.cuda()
                #         pair_labels = pair_labels.unsqueeze(-1).cuda()

                #     pos_indices = torch.where(pair_labels > 0)
                #     neg_indices = torch.where(pair_labels < 0)
                #     neg_ind = torch.randint(0, neg_indices[0].shape[0], [5*pos_indices[0].shape[0]]).cuda()
                #     neg_dis = (pred[pairs[neg_indices[0][neg_ind], 0]] - pred[pairs[neg_indices[0][neg_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
                #     pos_dis = (pred[pairs[pos_indices[0], 0]] - pred[pairs[pos_indices[0], 1]]).pow(2).sum(1).unsqueeze(-1)
                #     pos_dis = torch.cat([pos_dis]*5,0)
                #     pairs_indices = torch.where(torch.clamp(pos_dis + args.a - neg_dis, min=0.0)>0)
                #     loss = torch.mean(torch.clamp(pos_dis + args.a - neg_dis, min=0.0)[pairs_indices[0]]) 

                #     label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda()
                #     pred = F.normalize(pred, 2, 1)
                #     pair_out = torch.mm(pred,pred.t())
                #     if args.add_ort:
                #         pair_loss = (pair_matrix - pair_out).pow(2).mean()
                #         print("pair loss:",loss,"pair orthogonal loss:  ",100*pair_loss)
                #         loss += 100 * pair_loss
                # losses.append(loss.item())
                # total_loss += loss.item()
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()


                # batch_seconds_spent = time.time() - start_batch
                # seconds_train_batches.append(batch_seconds_spent)
                #endregion


            # 添加代码完毕
            #region
            # 这个循环使用DGL库的NeighborSampler来采样邻居节点，并进行批处理。g是图数据，
            # args.batch_size是批大小，
            # args.n_neighbors是采样的邻居数，
            # neighbor_type='in'表示采样的邻居类型，
            # shuffle=True表示是否打乱采样顺序，
            # num_workers=32表示多线程数，
            # num_hops=2表示采样的跳数，
            # seed_nodes是起始节点。
            # for batch_id, nf in enumerate(dgl.contrib.sampling.NeighborSampler(g,
            #                                                                 args.batch_size,
            #                                                                 args.n_neighbors,
            #                                                                 neighbor_type='in',
            #                                                                 shuffle=True,
            #                                                                 num_workers=24,
            #                                                                 num_hops=2,
            #                                                                 seed_nodes=train_indices)):
            #     start_batch = time.time()
            #     # 这些行将节点流（NodeFlow）的数据从父图复制过来，并设置模型为训练模式。
            #     nf.copy_from_parent()
            #     model.train()
            #     # forward
            #     if args.use_dgi:
            #         pred, ret = model(nf)  # pred: representations of the sampled nodes (in the last layer of the NodeFlow), ret: discriminator results
            #     else:
            #         # nf是节点特征
            #         pred = model(nf)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
            #     batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
            #     batch_labels = labels[batch_nids]
            #     loss_outputs = loss_fn(pred, batch_labels)
            #     loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            #     if args.use_dgi:
            #         n_samples = nf.layer_nid(-1).size()[0]
            #         lbl_1 = torch.ones(n_samples)
            #         lbl_2 = torch.zeros(n_samples)
            #         lbl = torch.cat((lbl_1, lbl_2), 0)
            #         if args.use_cuda:
            #             lbl = lbl.cuda()
            #         losses_triplet.append(loss.item())
            #         loss_dgi = loss_fn_dgi(ret, lbl)
            #         losses_dgi.append(loss_dgi.item())
            #         loss += loss_dgi
            #         losses.append(loss.item())
            #     else:
            #         losses.append(loss.item())
            #     total_loss += loss.item()

            #     for metric in metrics:
            #         metric(pred, batch_labels, loss_outputs)

            #     if batch_id % args.log_interval == 0:
            #         message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #             batch_id * args.batch_size, train_indices.shape[0],
            #             100. * batch_id / ((train_indices.shape[0] // args.batch_size) + 1), np.mean(losses))
            #         if args.use_dgi:
            #             message += '\tLoss_triplet: {:.6f}'.format(np.mean(losses_triplet))
            #             message += '\tLoss_dgi: {:.6f}'.format(np.mean(losses_dgi))
            #         for metric in metrics:
            #             message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            #         print(message)
            #         with open(save_path_i + '/log.txt', 'a') as f:
            #             f.write(message)
            #         losses = []

            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            #     batch_seconds_spent = time.time() - start_batch
            #     seconds_train_batches.append(batch_seconds_spent)
            #     # end one batch
            #endregion
            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            mins_spent = (time.time() - start_epoch) / 60
            message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
            message += '\n'
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            model.eval()  # Switch to evaluation mode
            with torch.no_grad():
                validation_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
                validation_dataloader = dgl.dataloading.NodeDataLoader(
                    g, validation_indices, validation_sampler,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                )
                total_validation_loss = 0  # 初始化验证损失
                validation_losses = []
                for batch_id, (input_nodes, output_nodes, blocks) in enumerate(validation_dataloader):
                    blocks = [b.to(device) for b in blocks]
                    batch_labels = blocks[-1].dstdata['labels']

                    pred = model(blocks)  # Forward pass
                    
                    if args.add_pair:
                        batch_labels_set = set(batch_labels)
                        label_center, cen_keys, cen_metrics, cen_values = get_cen_metric(batch_labels_set, batch_labels, pred)
                        cen_pred_key = np.array([cen_keys[l] for l in batch_labels.cpu().data.numpy()])
                        cen_metrics = cen_values.detach().cpu().numpy()[cen_pred_key]

                        cen_pred_metrics = cen_metrics - pred.detach().cpu().numpy()
                        pre_pre_metric = (pred.unsqueeze(1) - pred.unsqueeze(0)).detach().cpu().numpy()
                        cen_pred_metrics = np.tile(np.expand_dims(cen_pred_metrics, axis=1), (1, len(batch_labels), 1))
                        pre_pre_dot_metric = np.sum(
                            (cen_pred_metrics / ((np.linalg.norm(cen_pred_metrics, ord=2, axis=-1, keepdims=True)) + 1e-2)) * (
                                pre_pre_metric / (((np.linalg.norm(pre_pre_metric, ord=2, axis=-1, keepdims=True))) + 1e-2)),
                            axis=-1)
                        pairs, pair_labels, pair_matrix = pairwise_sample(pre_pre_dot_metric, batch_labels)

                        pairs = pairs.cuda()
                        pair_matrix = pair_matrix.cuda()
                        pos_indices = torch.where(pair_labels > 0)[0].cuda()
                        neg_indices = torch.where(pair_labels < 0)[0].cuda()

                        max_val = neg_indices.shape[0]

                        if max_val < 100:
                            dis = torch.empty([0, 1]).cuda()
                            for l in set(batch_labels.cpu().data.numpy()):
                                label_indices = torch.where(batch_labels == l)
                                l_center = label_center[l]
                                dis_l = (pred[label_indices] - l_center).pow(2).sum(1).unsqueeze(-1)
                                dis = torch.cat([dis, dis_l], 0)
                            cen_metrics_tensor = torch.tensor(cen_metrics).cuda()
                            validation_loss = torch.clamp(dis.mean() - cen_metrics_tensor.mean() + 16, min=0.0)
                        else:
                            max_sample = max_val if max_val > pos_indices.shape[0] else pos_indices.shape[0]
                            neg_ind = torch.randint(0, neg_indices.shape[0], [max_sample]).cuda()
                            pos_ind = torch.randint(0, pos_indices.shape[0], [max_sample]).cuda()

                            neg_dis = (pred[pairs[neg_indices[neg_ind], 0]] - pred[pairs[neg_indices[neg_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
                            pos_dis = (pred[pairs[pos_indices[pos_ind], 0]] - pred[pairs[pos_indices[pos_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
                            pairs_indices = torch.where(torch.clamp(pos_dis + 16 - neg_dis, min=0.0) > 0)
                            validation_loss = torch.mean(torch.clamp(pos_dis + 16 - neg_dis, min=0.0)[pairs_indices[0]])
                    pred = F.normalize(pred, 2, 1)
                    pair_out = torch.mm(pred, pred.t())
                    if args.add_ort:
                        pair_loss = (pair_matrix - pair_out).pow(2).mean()
                        print("pair loss:", validation_loss.item(), "pair orthogonal loss:  ", 100 * pair_loss.item())
                        validation_loss += 100 * pair_loss
                validation_losses.append(validation_loss.item())
                total_validation_loss += validation_loss.item()
            total_validation_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average validation loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_validation_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            message += '\n'
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            mins_train_epochs.append(mins_spent)

            # Validation
            # Infer the representations of all tweets
            extract_features, extract_labels = extract_embeddings(g, model, len(labels), args, labels)
            # Save the representations of all tweets
            # 添加的代码用于保存提取的特征和标签
            np.save(save_path_i + '/features_' + str(epoch) + '.npy', extract_features)
            np.save(save_path_i + '/labels_' + str(epoch) + '.npy', extract_labels)
            #添加代码完毕
            # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
            # Evaluate the model: conduct kMeans clustering on the validation and report NMI
            validation_nmi = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                    save_path_i, args, True)
            all_vali_nmi.append(validation_nmi)

            # Early stop
            if validation_nmi > best_vali_nmi:
                best_vali_nmi = validation_nmi
                best_epoch = epoch
                wait = 0
                # Save model
                model_path = save_path_i + '/models'
                if (epoch == 0) and (not os.path.isdir(model_path)):
                    os.mkdir(model_path)
                p = model_path + '/best.pt'
                torch.save(model.state_dict(), p)
                print('Best model saved after epoch ', str(epoch))
            else:
                wait += 1
            if wait == args.patience:
                print('Saved all_mins_spent')
                print('Early stopping at epoch ', str(epoch))
                print('Best model was at epoch ', str(best_epoch))
                break
            # end one epoch

        # Save all validation nmi
        np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
        # Save time spent on epochs
        np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
        print('Saved mins_train_epochs.')
        # Save time spent on batches
        np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
        print('Saved seconds_train_batches.')

        # Load the best model of the current block
        best_model_path = save_path_i + '/models/best.pt'
        model.load_state_dict(torch.load(best_model_path))
        print("Best model loaded.")



        if args.remove_obsolete == 2:
            return None, indices_to_remove, model
        return train_indices, indices_to_remove, model

