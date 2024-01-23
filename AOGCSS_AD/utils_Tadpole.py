import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from scipy.spatial import distance
import csv
from sklearn import svm
from utils import sys_normalized_adjacency
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
import numpy as np


c1_ind = [95, 37, 64, 127, 156, 146, 46, 118, 32, 88, 143, 60, 27, 125, 53, 129, 110, 21,
          101, 93, 16, 115, 8, 140, 102, 28, 72, 100, 137, 155, 31, 114, 158, 29, 44, 39,
          10, 116, 132, 133, 157, 109, 62, 144, 2, 66, 122, 3, 74, 47, 108, 111, 94, 30,
          147, 124, 26, 63, 22, 123, 61, 13, 55, 99, 54, 136, 36, 84, 65, 23, 96, 40,
          14, 69, 139, 98, 15, 107, 85, 5, 75, 11, 68, 49, 82, 149, 19, 138, 104, 131,
          33, 25, 79, 77, 135, 121, 105, 71, 12, 0, 34, 70, 89, 41, 4, 97, 134, 150,
          20, 90, 51, 42, 142, 87, 67, 7, 148, 57, 50, 38, 80, 76, 120, 1, 113, 52,
          86, 130, 18, 92, 141, 112, 154, 45, 128, 59, 117, 48, 6, 126, 35, 151, 73, 81,
          58, 145, 83, 152, 17, 103, 43, 56, 78, 153, 24, 106, 91, 119, 9]
c2_ind = [93, 231, 9, 21, 64, 149, 254, 291, 221, 304, 263, 140, 165, 253, 211, 160, 132, 48,
          268, 292, 280, 186, 216, 287, 67, 109, 31, 71, 258, 37, 227, 294, 224, 78, 158, 184,
          204, 10, 146, 176, 119, 39, 207, 127, 261, 297, 79, 122, 203, 225, 202, 218, 96, 266,
          58, 82, 135, 153, 59, 92, 123, 311, 17, 166, 296, 138, 85, 50, 312, 246, 233, 139, 281,
          25, 111, 95, 275, 313, 262, 2, 161, 70, 16, 106, 47, 271, 156, 198, 205, 201, 49, 38,
          4, 197, 101, 274, 32, 3, 267, 126, 44, 217, 53, 20, 114, 177, 257, 316, 52, 164, 30,
          120, 110, 249, 15, 270, 236, 256, 214, 273, 174, 234, 63, 121, 240, 157, 272, 192, 0,
          75, 22, 286, 307, 210, 7, 238, 171, 76, 74, 212, 88, 183, 187, 100, 251, 200, 73, 290,
          317, 116, 112, 230, 241, 195, 301, 152, 209, 314, 222, 61, 191, 155, 172, 179, 151, 42,
          83, 108, 81, 180, 300, 97, 259, 302, 141, 55, 264, 45, 36, 277, 298, 154, 315, 163, 33,
          299, 305, 84, 56, 208, 170, 6, 150, 124, 77, 118, 228, 148, 136, 173, 282, 60, 295, 46,
          57, 131, 279, 196, 104, 133, 289, 190, 193, 232, 54, 243, 89, 99, 242, 219, 169, 113,
          189, 162, 178, 43, 143, 80, 260, 248, 115, 34, 66, 24, 94, 91, 199, 8, 62, 252, 276,
          147, 245, 247, 40, 269, 220, 168, 283, 285, 14, 68, 223, 175, 284, 308, 128, 98, 125,
          194, 12, 65, 239, 265, 306, 26, 235, 117, 11, 19, 137, 144, 27, 237, 134, 29, 35, 103,
          102, 244, 142, 86, 41, 188, 213, 1, 105, 51, 310, 250, 167, 90, 215, 309, 130, 278, 69,
          87, 288, 185, 13, 182, 293, 28, 107, 145, 18, 159, 23, 303, 229, 255, 226, 72, 181, 206,
          129, 5]
c3_ind = [37, 74, 4, 69, 50, 14, 10, 22, 76, 53, 68, 64, 57, 45, 52, 12, 8, 65, 70, 18, 72, 11, 58,
          61, 26, 59, 51, 33, 73, 77, 42, 44, 21, 54, 27, 34, 79, 66, 6, 63, 9, 46, 60, 24, 78, 5,
          2, 32, 67, 38, 43, 36, 41, 1, 40, 30, 16, 47, 15, 20, 62, 19, 35, 7, 13, 75, 55, 71, 3, 17,
          31, 0, 56, 23, 49, 39, 25, 28, 48, 29]
id_ind = [524, 285, 210, 341, 191, 168, 92, 281, 122, 124, 205, 98, 1, 490, 10, 436, 496, 277, 71, 360,
          441, 250, 543, 115, 391, 397, 547, 195, 532, 232, 258, 276, 428, 329, 133, 252, 160, 119, 91,
          248, 65, 352, 35, 367, 396, 555, 108, 413, 11, 353, 520, 67, 500, 186, 334, 261, 203, 442, 481,
          178, 23, 303, 525, 126, 123, 42, 376, 5, 217, 3, 489, 128, 136, 209, 287, 30, 70, 173, 310, 112,
          542, 235, 263, 154, 147, 529, 511, 80, 486, 221, 57, 349, 172, 134, 280, 523, 141, 517, 405, 536,
          348, 503, 86, 6, 343, 135, 85, 58, 236, 127, 479, 132, 510, 457, 179, 275, 452, 300, 253, 495, 34,
          468, 7, 382, 199, 471, 364, 414, 501, 316, 162, 111, 66, 372, 410, 270, 24, 220, 84, 291, 395, 439,
          39, 82, 27, 219, 302, 100, 539, 512, 224, 139, 68, 321, 504, 394, 359, 116, 292, 249, 526, 369, 425,
          62, 102, 327, 215, 474, 171, 319, 440, 407, 22, 230, 182, 237, 470, 314, 540, 51, 473, 59, 298, 159,
          48, 415, 447, 332, 421, 138, 475, 18, 163, 399, 361, 223, 502, 326, 505, 521, 164, 117, 380, 330, 477,
          469, 279, 533, 266, 16, 274, 336, 392, 363, 328, 125, 416, 278, 33, 368, 355, 204, 385, 358, 216, 97, 77,
          238, 74, 393, 286, 72, 242, 322, 430, 515, 260, 130, 509, 389, 401, 99, 315, 63, 507, 482, 158, 131, 29,
          513, 295, 150, 460, 273, 243, 81, 412, 55, 554, 480, 381, 438, 129, 69, 60, 454, 420, 268, 366, 426, 356,
          388, 41, 247, 228, 229, 15, 551, 357, 386, 109, 448, 105, 152, 20, 214, 251, 375, 296, 347, 293, 79, 143,
          183, 21, 184, 174, 121, 190, 2, 402, 535, 350, 225, 202, 373, 419, 36, 101, 218, 244, 54, 148, 255, 272,
          107, 96, 294, 26, 553, 188, 444, 307, 95, 32, 257, 196, 193, 181, 254, 46, 549, 4, 506, 478, 331, 404, 61,
          354, 156, 289, 301, 38, 403, 137, 12, 282, 449, 463, 313, 176, 290, 340, 518, 508, 318, 431, 64, 445, 552,
          87, 466, 493, 239, 312, 451, 103, 75, 9, 231, 544, 213, 240, 462, 25, 387, 140, 259, 43, 149, 379, 398, 446,
          528, 56, 104, 13, 550, 233, 464, 498, 185, 49, 212, 378, 409, 390, 383, 351, 333, 317, 201, 484, 151, 262,
          339,
          189, 207, 73, 265, 146, 522, 423, 206, 465, 200, 297, 406, 530, 94, 177, 155, 308, 443, 422, 345, 226, 45,
          245,
          17, 455, 40, 417, 114, 344, 534, 264, 435, 374, 418, 161, 19, 271, 305, 78, 456, 541, 467, 499, 170, 432, 113,
          8,
          145, 384, 198, 437, 427, 556, 90, 365, 342, 299, 411, 485, 494, 157, 256, 0, 514, 50, 335, 28, 44, 208, 323,
          548,
          76, 346, 267, 89, 142, 338, 371, 306, 429, 370, 377, 227, 144, 546, 246, 284, 37, 450, 362, 31, 47, 192, 488,
          110,
          545, 538, 241, 476, 453, 53, 519, 165, 458, 187, 83, 461, 434, 106, 194, 516, 472, 433, 324, 169, 311, 492,
          320,
          487, 288, 175, 491, 180, 88, 269, 93, 222, 325, 400, 459, 483, 309, 304, 166, 120, 424, 408, 118, 211, 167,
          153, 497,
          283, 537, 14, 52, 337, 531, 197, 234, 527]


def Random_shuffle(x, indx):
    y = []
    for i in indx:
        y.append(x[i])
    return y

def preprocess_features(features):
    rowsum = np.asarray(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense())


def TadpoleGarphDataset(sparsity_threshold):
    with open('tadpole_dataset/tadpole_2.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        apoe = []
        ages = []
        gender = []
        fdg = []
        features = []
        labels = []
        cnt = 0
        apoe_col_num = 0
        age_col_num = 0
        gender_col_num = 0
        fdg_col_num = 0
        label_col_num = 0
        for row in rows:
            if cnt != 0:
                row_features = row[fdg_col_num + 1:]
                if row_features.count('') == 0 and row[apoe_col_num] != '':
                    apoe.append(int(row[apoe_col_num]))
                    ages.append(float(row[age_col_num]))
                    gender.append(row[gender_col_num])
                    fdg.append(float(row[fdg_col_num]))
                    labels.append(int(row[label_col_num]) - 1)
                    features.append([float(item) for item in row_features])
                    cnt += 1
            else:
                apoe_col_num = row.index('APOE4')
                age_col_num = row.index('AGE')
                gender_col_num = row.index('PTGENDER')
                fdg_col_num = row.index('FDG')
                label_col_num = row.index('DXCHANGE')
                cnt += 1

        num_nodes = len(labels)
        apoe_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if apoe[i] == apoe[j]:
                    apoe_affinity[i, j] = apoe_affinity[j, i] = 1
        age_threshold = 2
        age_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.abs(ages[i] - ages[j]) <= age_threshold:
                    age_affinity[i, j] = age_affinity[j, i] = 1

        gender_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if gender[i] == gender[j]:
                    gender_affinity[i, j] = gender_affinity[j, i] = 1
        reshaped_fdg = np.reshape(np.asarray(fdg), newshape=[-1, 1])
        svc = svm.SVC(kernel='linear').fit(reshaped_fdg, labels)
        prediction = svc.predict(reshaped_fdg)
        fdg_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if prediction[i] == prediction[j]:
                    fdg_affinity[i, j] = fdg_affinity[j, i] = 1

        features = np.asarray(features)
        column_sum = np.array(features.sum(0))
        r_inv = np.power(column_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        features = features.dot(r_mat_inv)
        dist = distance.pdist(features, metric='euclidean')
        dist = distance.squareform(dist)
        sigma = np.mean(dist)
        w = np.exp(- dist ** 2 / (2 * sigma ** 2))
        w[w < sparsity_threshold] = 0
        apoe_affinity *= w
        age_affinity *= w
        gender_affinity *= w
        fdg_affinity *= w
        mixed_affinity = (apoe_affinity+gender_affinity+fdg_affinity+age_affinity)/4#最终的权重矩阵
        edge = np.array(np.nonzero(mixed_affinity))
        c_1 = [i for i in range(num_nodes) if labels[i] == 0]
        c_2 = [i for i in range(num_nodes) if labels[i] == 1]
        c_3 = [i for i in range(num_nodes) if labels[i] == 2]
        c_1_num = len(c_1)
        c_2_num = len(c_2)
        c_3_num = len(c_3)
        num_nodes = c_1_num + c_2_num + c_3_num
        c_1 = Random_shuffle(c_1, c1_ind)
        c_2 = Random_shuffle(c_2, c2_ind)
        c_3 = Random_shuffle(c_3, c3_ind)
        selection_c_1 = c_1[:c_1_num]
        selection_c_2 = c_2[:c_2_num]
        selection_c_3 = c_3[:c_3_num]
        idx = np.concatenate((selection_c_1, selection_c_2, selection_c_3), axis=0)
        node_weights = np.zeros((num_nodes,))
        node_weights[selection_c_1] = 1 - c_1_num / float(num_nodes)
        node_weights[selection_c_2] = 1 - c_2_num / float(num_nodes)
        node_weights[selection_c_3] = 1 - c_3_num / float(num_nodes)
        idx = Random_shuffle(idx, id_ind)
        features = features[idx, :]
        labels = [labels[item] for item in idx]
        mixed_affinity = mixed_affinity[idx,:]
        mixed_affinity = mixed_affinity[:,idx]
        train_proportion = 0.55
        val_proportion = 0.25
        train_mask = np.zeros((num_nodes,), dtype=np.bool_)
        val_mask = np.zeros((num_nodes,), dtype=np.bool_)
        test_mask = np.zeros((num_nodes,), dtype=np.bool_)
        train_mask[:int(train_proportion * num_nodes)] = 1
        val_mask[int(train_proportion * num_nodes): int((train_proportion + val_proportion) * num_nodes)] = 1
        test_mask[int((train_proportion + val_proportion) * num_nodes):] = 1
        num_labels = 3
        one_hot_labels = np.zeros((num_nodes, num_labels))
        one_hot_labels[np.arange(num_nodes), labels] = 1
        train_label = np.zeros(one_hot_labels.shape)
        val_label = np.zeros(one_hot_labels.shape)
        test_label = np.zeros(one_hot_labels.shape)
        train_label[train_mask, :] = one_hot_labels[train_mask,:]
        val_label[val_mask, :] = one_hot_labels[val_mask, :]
        test_label[test_mask, :] = one_hot_labels[test_mask, :]
        mixed_affinity = sp.csr_matrix(mixed_affinity)
        num_features = features.shape[1]
        labels = np.array(labels)
        features = np.mat(features)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        train_mask = torch.BoolTensor(train_mask)
        val_mask = torch.BoolTensor(val_mask)
        test_mask = torch.BoolTensor(test_mask)
        adj = sys_normalized_adjacency(mixed_affinity)

        return adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
