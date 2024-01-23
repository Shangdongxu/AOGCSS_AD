from __future__ import print_function, division
import time
import random
import argparse
import torch.optim as optim
from process import *
from utils import *
from model import *
import uuid
import itertools
from utils_Tadpole import TadpoleGarphDataset
from utils import calc_chebynet_gso
from sklearn.metrics import  recall_score, f1_score
from tqdm import trange
from sklearn.metrics import roc_auc_score
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=512, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=7, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=256, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='Tadpole', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--layer_norm',type=int, default=1, help='layer norm')
parser.add_argument('--w_fc2',type=float, default=0.0001, help='Weight decay layer-2')
parser.add_argument('--w_fc1',type=float, default=0.0001, help='Weight decay layer-1')
parser.add_argument('--lr_fc',type=float, default=0.01, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_att',type=float, default=0.01, help='Learning rate Scalar')
parser.add_argument('--lr_sel',type=float, default=0.01, help='Learning rate for selector')
parser.add_argument('--wd_sel',type=float,default=8e-06,help='weight decay selector layer')
parser.add_argument('--step1_iter',type=int, default=4000, help='Step-1 iterations')
parser.add_argument('--step2_iter',type=int, default=2, help='Step-2 iterations')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
num_layer = 15
feat_select =15
layer_norm = bool(int(args.layer_norm))
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

def Get_new_graph(graph):
    row, col = graph.shape
    new_graph2 = np.zeros_like(graph)
    new_graph3 = np.zeros_like(graph)
    for row_node in range(row):
        adj = []
        for col_node in range(col):
            if graph[row_node, col_node] and col_node not in adj:
                adj.append(col_node)
        if len(adj):
            adj0 = []
            weight = []
            for f_node2 in adj:
                for col_node in range(col):
                    if graph[f_node2, col_node] and col_node not in adj and col_node != row_node:
                        if graph[row_node, f_node2] + graph[f_node2, col_node] > new_graph2[row_node, col_node]:
                            new_graph2[row_node, col_node] = graph[row_node, f_node2] + graph[f_node2, col_node]
                        adj0.append(col_node)
                        weight.append(graph[f_node2, col_node])

            adj.append(row_node)
            if len(adj0):
                adj1 = []
                for f_node3 in adj0:
                    for col_node in range(col):
                        if graph[f_node3, col_node] and col_node not in adj and col_node not in adj0:
                            if new_graph2[row_node, f_node3] + graph[f_node3, col_node] > new_graph3[row_node, col_node]:
                                new_graph3[row_node, col_node] = new_graph2[row_node, f_node3] + graph[f_node3, col_node]
                            adj1.append(col_node)
    return new_graph2 / 2, new_graph3 / 3
def predict(model_sel,mask):
    output = model_sel(mask,None)
    return output
def train_step(model,optimizer,labels,list_mat,idx_train,list_ind):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat, layer_norm,list_ind)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()
def validate_step(model,labels,list_mat,idx_val,list_ind):
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm,list_ind)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        return loss_val.item(),acc_val.item()
def calculate_rmse(actual_values, predicted_values):
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    mse = np.mean((predicted_values - actual_values) ** 2)
    rmse = np.sqrt(mse)
    return rmse
def test_step(model,labels,list_mat,idx_test,list_ind):
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm,list_ind)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        probabilities = F.softmax(output[idx_test], dim=1)
        probabilities_numpy = probabilities.numpy()
        labels_numpy = labels[idx_test].numpy()
        recall = recall_score(labels[idx_test],output[idx_test].max(1)[1].type_as(labels),average='macro')
        f_measure = f1_score(labels[idx_test],output[idx_test].max(1)[1].type_as(labels),average='macro')
        rmse = calculate_rmse(labels[idx_test], output[idx_test].max(1)[1].type_as(labels))
        one_hot_labels = np.zeros((len(labels_numpy), np.max(labels_numpy) + 1))
        one_hot_labels[np.arange(len(labels_numpy)), labels_numpy] = 1
        auc = roc_auc_score(one_hot_labels, probabilities_numpy, multi_class='ovr')
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return loss_test.item(),acc_test.item(),auc,recall,f_measure,rmse
def selector_step(model,optimizer_sel,mask,o_loss):
    model.train()
    optimizer_sel.zero_grad()
    mask.requires_grad = True
    output = model(mask,o_loss)
    selector_loss = 10*F.mse_loss(output,o_loss)
    selector_loss.backward()
    input_grad = mask.grad.data
    optimizer_sel.step()
    return selector_loss.item(), input_grad,model
def train(datastr):
    if datastr == 'Tadpole':
        thread = 0.5
        _,features, labels, idx_train, idx_val, idx_test, num_features, num_labels = TadpoleGarphDataset(thread)
    else:
        adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = NACCDataset(datastr)
    features = features
    list_mat = []
    list_mat.append(features)
    supports = []
    for i in range(1,16):
        mat = np.load('G' + str(i) + '.npy')
        mat = sp.csr_matrix(mat)
        mat = sys_normalized_adjacency(mat)
        support = calc_chebynet_gso(mat)
        support = sparse_mx_to_torch_sparse_tensor(support)
        supports.append(support)
    for i in range(len(supports)):
        supports[i] = supports[i]
    for ii in range(len(supports)):
        loop_mat = torch.spmm(supports[ii], features)
        list_mat.append(loop_mat)
    model = Classifier(nfeat=num_features,
                nlayers=num_layer,
                nhidden=args.hidden,
                nclass=num_labels,
                dropout=args.dropout)
    optimizer_sett_classifier = [
        {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
        {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
    ]
    optimizer = optim.Adam(optimizer_sett_classifier)
    model_sel = Selector(num_layer,args.hidden)
    optimizer_select = [
        {'params':model_sel.fc1.parameters(), 'weight_decay':args.wd_sel, 'lr':args.lr_sel},
        {'params':model_sel.fc2.parameters(), 'weight_decay':args.wd_sel, 'lr':args.lr_sel}
    ]
    optimizer_sel = optim.Adam(optimizer_select)
    bad_counter = 0
    best = 999999999
    best_sub = []
    best_acc = 0
    combinations = list()
    for nn in range(1,feat_select):
        combinations.extend(list(itertools.combinations(range(num_layer),nn)))
    dict_comb = dict()
    for kk,cc in enumerate(combinations):
        dict_comb[cc] = kk
    for epoch in range(args.step1_iter):
        rand_ind = random.choice(combinations)
        input_mat = [list_mat[ww] for ww in rand_ind]
        loss_tra,acc_tra = train_step(model,optimizer,labels,input_mat,idx_train,rand_ind)
        loss_val,acc_val = validate_step(model,labels,input_mat,idx_val,rand_ind)
        input_mask = torch.zeros(num_layer).float()
        input_mask[list(rand_ind)] = 1.0
        input_loss = torch.FloatTensor([loss_tra])
        eval_loss = torch.FloatTensor([loss_val])
        loss_select, input_grad,model_sel= selector_step(model_sel,optimizer_sel,input_mask,input_loss)
    dict_check_loss = dict()
    best_sub = []
    best_sub_temp = []
    best_loss =9999999999
    best_loss_temp = 9999999999
    best = 999999999
    dict_check_loss = dict()
    for epoch in range(args.step2_iter):
        for i in combinations:
            ergodic = i
            ergodic_mask = torch.zeros(num_layer).float()
            ergodic_mask[list(ergodic)] = 1.0
            predict_loss = predict(model_sel, ergodic_mask)
            if (predict_loss < best_loss_temp and len(ergodic)<6):
                best_sub_temp = ergodic_mask
                best_loss_temp = predict_loss
        if(best_loss_temp<best_loss ):
            best_sub = best_sub_temp
            best_loss = best_loss_temp
    train_mask = best_sub.tolist()
    train_mask = [index for index, value in enumerate(train_mask) if value == 1.0]
    input_mat = [list_mat[ww] for ww in train_mask]
    dict_check_loss[loss_val] = train_mask
    input_mask = torch.zeros(num_layer).float()
    input_mask[list(train_mask)] = 1.0
    for epoch in range(10000):
        if (epoch + 1) % 1 == 0:
            print('Epoch:{:04d}'.format(epoch + 1),
                  'train',
                  'loss:{:.3f}'.format(loss_tra),
                  'acc:{:.2f}'.format(acc_tra * 100),
                  '| val',
                  'loss:{:.3f}'.format(loss_val),
                  'acc:{:.2f}'.format(acc_val * 100))
        loss_tra, acc_tra = train_step(model, optimizer, labels, input_mat, idx_train, train_mask)
        loss_val, acc_val = validate_step(model, labels, input_mat, idx_val, train_mask)
        if loss_val < best:
            best = loss_val
            bad_counter = 0
            best_sub = train_mask
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break
    select_ind = best_sub
    supportss = []
    for i in select_ind:
        j = i + 1
        mat = np.load('G' + str(j) + '.npy')
        thred = 0.9
        thred2 = 0.9
        mat_orgin = mat
        mat[mat < thred] = 0
        G2, G3 = Get_new_graph(mat)
        G2[G2 < thred2] = 0
        G3[G3 < thred2] = 0
        mat_orgin = sp.csr_matrix(mat_orgin)
        G2 = sp.csr_matrix(G2)
        G3 = sp.csr_matrix(G3)
        mat_orgin = sys_normalized_adjacency(mat_orgin)
        G2 = sys_normalized_adjacency(G2)
        G3 = sys_normalized_adjacency(G3)
        support = calc_chebynet_gso(mat_orgin)
        support_2 = calc_chebynet_gso(G2)
        support_3 = calc_chebynet_gso(G3)
        support = sparse_mx_to_torch_sparse_tensor(support)
        support_2 = sparse_mx_to_torch_sparse_tensor(support_2)
        support_3 = sparse_mx_to_torch_sparse_tensor(support_3)
        supportss.append(support)
        supportss.append(support_2)
        supportss.append(support_3)
    for i in range(len(supportss)):
        supportss[i] = supportss[i]
    for ii in range(len(supportss)):
        loop_mat = torch.spmm(supportss[ii], features)
        list_mat.append(loop_mat)
    combinations = list()
    for nn in range(1, len(supportss) + 1):
        combinations.extend(list(itertools.combinations(range(len(supportss)), nn)))
    dict_comb = dict()
    for kk, cc in enumerate(combinations):
        dict_comb[cc] = kk
    for epoch in range(args.step1_iter):
        rand_ind = random.choice(combinations)
        input_mat = [list_mat[ww] for ww in rand_ind]
        loss_tra,acc_tra = train_step(model,optimizer,labels,input_mat,idx_train,rand_ind)
        loss_val,acc_val = validate_step(model,labels,input_mat,idx_val,rand_ind)
        dict_check_loss = dict()
        best_sub = []
        best_sub_temp = []
        best_loss = 9999999999
        best_loss_temp = 9999999999
        best = 999999999
        dict_check_loss = dict()
        for epoch in range(args.step2_iter):
            for i in combinations:
                ergodic = i
                ergodic_mask = torch.zeros(num_layer).float()
                ergodic_mask[list(ergodic)] = 1.0
                predict_loss = predict(model_sel, ergodic_mask)
                if (predict_loss < best_loss_temp):
                    best_sub_temp = ergodic_mask
                    best_loss_temp = predict_loss
            if (best_loss_temp < best_loss):
                best_sub = best_sub_temp
                best_loss = best_loss_temp
        train_mask = best_sub.tolist()
        train_mask = [index for index, value in enumerate(train_mask) if value == 1.0]
        input_mat = [list_mat[ww] for ww in train_mask]
        dict_check_loss[loss_val] = train_mask
        input_mask = torch.zeros(num_layer).float()
        input_mask[list(train_mask)] = 1.0
        for epoch in range(10000):
            if (epoch + 1) % 1 == 0:
                print('Epoch:{:04d}'.format(epoch + 1),
                      'train',
                      'loss:{:.3f}'.format(loss_tra),
                      'acc:{:.2f}'.format(acc_tra * 100),
                      '| val',
                      'loss:{:.3f}'.format(loss_val),
                      'acc:{:.2f}'.format(acc_val * 100))
            loss_tra, acc_tra = train_step(model, optimizer, labels, input_mat, idx_train, train_mask)
            loss_val, acc_val = validate_step(model, labels, input_mat, idx_val, train_mask)
            if loss_val < best:
                best = loss_val
                bad_counter = 0
                best_sub = train_mask
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
        select_ind = best_sub
        input_mat = [list_mat[ww] for ww in select_ind]
        test_out = test_step(model, labels, input_mat, idx_test, select_ind)
        acc = test_out[1]
        auc = test_out[2]
        recall = test_out[3]
        f_measure = test_out[4]
        rmse = test_out[5]
        return acc * 100, auc * 100,recall*100,f_measure*100,rmse

t_total = time.time()
acc_list = []
auc_list = []
recall_list = []
f_measure_list = []
rmse_list = []
for i in trange(10):
    datastr = args.data
    accuracy_data,auc_data,recall_data,f_measure_data,rmse_data = train(datastr)
    auc_list.append(auc_data)
    acc_list.append(accuracy_data)
    recall_list.append(recall_data)
    f_measure_list.append(f_measure_data)
    rmse_list.append(rmse_data)
print("Train time: {:.4f}s".format(time.time() - t_total))
print(f"Test accuracy: {np.mean(acc_list):.2f}, {np.round(np.std(acc_list),2)}")
print(acc_list)
print(f"Area Under the Curve: {np.mean(auc_list):.2f}, {np.round(np.std(auc_list),2)}")
print(auc_list)


