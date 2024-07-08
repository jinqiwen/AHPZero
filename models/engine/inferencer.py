import torch
import numpy as np
from sklearn.metrics import accuracy_score
import time
def cal_accuracy(model, dataloadr, att, test_id, device,att_all, bias=None):

    scores = []
    labels = []
    cpu = torch.device('cpu')

    for iteration, (img, label) in enumerate(dataloadr):
        img = img.to(device)
        score = model(img, seen_att=att, att_all=att_all)
        scores.append(score)
        labels.append(label)

    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)

    if bias is not None:
        scores = scores-bias

    _,pred = scores.max(dim=1)
    pred = pred.view(-1).to(cpu)

    outpred_0 = test_id[pred]
    outpred = np.array(outpred_0, dtype='int')
    labels = labels.numpy()
    unique_labels = np.unique(labels)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(labels == l)[0]
        acc += accuracy_score(labels[idx], outpred[idx])
    acc = acc / unique_labels.shape[0]
    # 计算并保存每个类别的acc
    compute_per_class_acc_gzsl(
        torch.from_numpy(labels).to(device), torch.from_numpy(outpred).to(device), torch.from_numpy(unique_labels).to(device))
    return acc

def eval(
        tu_loader,
        ts_loader,
        att_unseen,
        att_seen,
        cls_unseen_num,
        cls_seen_num,
        test_id,
        train_test_id,
        model,
        test_gamma,
        device
):
    att = torch.cat((att_seen, att_unseen), dim=0)
    acc_zsl = cal_accuracy(model=model, dataloadr=tu_loader, att=att_unseen, test_id=test_id, device=device, att_all=att, bias=None)

    bias_s = torch.zeros((1, cls_seen_num)).fill_(test_gamma).to(device)
    bias_u = torch.zeros((1, cls_unseen_num)).to(device)
    bias = torch.cat([bias_s, bias_u], dim=1)


    acc_gzsl_unseen = cal_accuracy(model=model, dataloadr=tu_loader, att=att,
                                   test_id=train_test_id, device=device, att_all=att, bias=bias)
    acc_gzsl_seen = cal_accuracy(model=model, dataloadr=ts_loader, att=att,
                                   test_id=train_test_id, device=device, att_all=att,bias=bias)
    H = 2 * acc_gzsl_seen * acc_gzsl_unseen / (acc_gzsl_seen + acc_gzsl_unseen)

    return acc_zsl, acc_gzsl_unseen, acc_gzsl_seen, H


def eval_zs_gzsl(
        tu_loader,
        ts_loader,
        res,
        model,
        test_gamma,
        device
):
    model.eval()
    att_unseen = res['att_unseen'].to(device)
    att_seen = res['att_seen'].to(device)

    test_id = res['test_id']
    train_test_id = res['train_test_id']

    cls_seen_num = att_seen.shape[0]
    cls_unseen_num = att_unseen.shape[0]

    with torch.no_grad():
        acc_zsl, acc_gzsl_unseen, acc_gzsl_seen, H = eval(
            tu_loader,
            ts_loader,
            att_unseen,
            att_seen,
            cls_unseen_num,
            cls_seen_num,
            test_id,
            train_test_id,
            model,
            test_gamma,
            device
        )

    model.train()

    return acc_gzsl_seen, acc_gzsl_unseen, H, acc_zsl

# 计算并保存每个类别的acc
def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, save_ic_img = False):
    device = test_label.device
    per_class_accuracies = torch.zeros(
        target_classes.size()[0]).float().to(device).detach()
    predicted_label = predicted_label.to(device)
    for i in range(target_classes.size()[0]):
        is_class = test_label == target_classes[i]
        per_class_accuracies[i] = torch.div(
            (predicted_label[is_class] == test_label[is_class]).sum().float(),
            is_class.sum().float())
        # if save_ic_img :

    save_per_class_acc(per_class_accuracies,target_classes,"gzsl")
    return per_class_accuracies.mean().item()

def save_per_class_acc(tensor,class_id,prefix):
    species = [str(i) for i in range(1, 103)]
    species_array = np.array(species)
    cl_id = class_id.cpu().numpy().tolist()
    species_array = species_array[cl_id]
    cpu_tensor = tensor.cpu()
    # 获取张量的索引和值
    indices = [i for i in range(len(cpu_tensor))]
    values = [cpu_tensor[i].item() for i in range(len(cpu_tensor))]
    # 记录当前时间
    current_time = time.strftime("%Y-%m-%d %H", time.localtime())
    with open('checkpoints/results.log', 'a') as file:
        file.write(f"\n{prefix}_save_time:{current_time}\n")
        for index, value in zip(indices, values):
            file.write(f"{species_array[index]}, acc:{value}\n")