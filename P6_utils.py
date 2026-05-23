from sklearn.metrics import roc_curve, auc, accuracy_score
import torch
from torch import nn
import torch.nn.functional as F


def calculate_auc(predictRow_epoch, nextLabelRow_epoch):
    nextLabelRow_epoch = nextLabelRow_epoch.detach().cpu()
    predictRow_epoch = predictRow_epoch.detach().cpu()
    fpr, tpr, thresholds = roc_curve(nextLabelRow_epoch, predictRow_epoch, pos_label=1)
    auc_val = auc(fpr, tpr)
    return auc_val


def calculate_acc(predictRow_epoch, nextLabelRow_epoch):
    nextLabelRow_epoch = nextLabelRow_epoch.detach().cpu()
    predictRow_epoch = predictRow_epoch.detach().cpu()
    predict_bool = predictRow_epoch >= 0.5
    next_label_bool = nextLabelRow_epoch == 1
    acc = accuracy_score(y_true=next_label_bool, y_pred=predict_bool)
    return acc


def selecting_mask(effLen, maxLen, opt):
    bsz = effLen.size(0)
    mask = torch.arange(end=maxLen, device=opt.DEVICE).repeat(repeats=(bsz, 1)) < effLen.unsqueeze(1)
    return mask


def calculate_lossGlobal(predict, label, maskEffLen):
    predict = predict.masked_select(maskEffLen)
    label = label.masked_select(maskEffLen)
    predict = torch.nan_to_num(predict, nan=0.5, posinf=1.0, neginf=0.0)
    predict = torch.clamp(predict, min=1e-6, max=1.0 - 1e-6)
    label = torch.nan_to_num(label, nan=0.0, posinf=1.0, neginf=0.0)
    label = torch.clamp(label, min=0.0, max=1.0)

    loss_fn = nn.BCELoss(reduction='sum')
    bce_loss = loss_fn(predict, label)
    return bce_loss


def calculate_lossParams(params, nextSkill_oneHot, effLens, opt):
    L, G, S = params
    lossList = []
    mask_L_forward = L[:, 1:, :] >= L[:, :-1, :]
    mask_L_forwardThres = (L[:, 1:, :] - L[:, :-1, :]) > opt.L_forwardPunishThreshold
    lossL_forward = mask_L_forward.float() * mask_L_forwardThres.float() * (
            L[:, 1:, :] - L[:, :-1, :]) ** 2 * opt.LForwardPunish
    mask_L_backwardThres = (L[:, :-1, :] - L[:, 1:, :]) > opt.L_backwardPunishThreshold
    lossL_backward = (~mask_L_forward).float() * mask_L_backwardThres.float() * (
            L[:, :-1, :] - L[:, 1:, :]) ** 2 * opt.LBackPunish
    lossL = torch.sum(lossL_forward + lossL_backward)
    lossList.append(lossL)

    G = G * nextSkill_oneHot
    S = S * nextSkill_oneHot
    G_punishThreshold = effLens.type(torch.float) * opt.G_punishThresholdCoef
    S_punishThreshold = effLens.type(torch.float) * opt.S_punishThresholdCoef
    G_effSkillSum = torch.sum(G, dim=[1, 2])
    G_maskGt0 = (G_effSkillSum - G_punishThreshold) > 0
    G_effSkillSumPunish = (G_effSkillSum - G_punishThreshold) * G_maskGt0.float()
    lossG = torch.sum(G_effSkillSumPunish)
    lossList.append(lossG)
    S_effSkillSum = torch.sum(S, dim=[1, 2])
    S_maskGt0 = (S_effSkillSum - S_punishThreshold) > 0
    S_effSkillSumPunish = (S_effSkillSum - S_punishThreshold) * S_maskGt0.float()
    lossS = torch.sum(S_effSkillSumPunish)
    lossList.append(lossS)

    sumParamsLoss = 0
    for los in lossList:
        sumParamsLoss += los

    return sumParamsLoss, lossList


def dual_stream_contrast_loss(predict, cf_predict, L_skill, cf_L_skill, nextLabel, maskEffLen, opt,
                              margin, lambda_positive, lambda_negative):
    if predict is None or cf_predict is None:
        return torch.tensor(0.0, device=predict.device if predict is not None else opt.DEVICE)

    predict = torch.nan_to_num(predict, nan=0.5, posinf=1.0, neginf=0.0)
    cf_predict = torch.nan_to_num(cf_predict, nan=0.5, posinf=1.0, neginf=0.0)
    predict = torch.clamp(predict, min=1e-6, max=1.0 - 1e-6)
    cf_predict = torch.clamp(cf_predict, min=1e-6, max=1.0 - 1e-6)
    nextLabel = torch.nan_to_num(nextLabel, nan=0.0, posinf=1.0, neginf=0.0)
    nextLabel = torch.clamp(nextLabel, min=0.0, max=1.0)

    predict_flat = predict.reshape(-1)
    cf_predict_flat = cf_predict.reshape(-1)
    nextLabel_flat = nextLabel.reshape(-1)
    mask_flat = maskEffLen.reshape(-1)

    valid_indices = mask_flat.nonzero().squeeze()
    if valid_indices.numel() == 0:
        return torch.tensor(0.0, device=predict.device)

    if valid_indices.dim() == 0:
        valid_indices = valid_indices.unsqueeze(0)

    predict_valid = predict_flat[valid_indices]
    cf_predict_valid = cf_predict_flat[valid_indices]
    nextLabel_valid = nextLabel_flat[valid_indices]

    correct_indices = (nextLabel_valid == 1).nonzero().squeeze()
    incorrect_indices = (nextLabel_valid == 0).nonzero().squeeze()

    cf_loss = torch.tensor(0.0, device=predict.device)

    if correct_indices.numel() > 0:
        if correct_indices.dim() == 0:
            correct_indices = correct_indices.unsqueeze(0)

        pred_diff_correct = predict_valid[correct_indices] - cf_predict_valid[correct_indices] + 1e-6
        pred_loss_correct = torch.mean(torch.log1p(torch.exp(-pred_diff_correct + margin)))
        cf_loss += pred_loss_correct * lambda_positive

    if incorrect_indices.numel() > 0:
        if incorrect_indices.dim() == 0:
            incorrect_indices = incorrect_indices.unsqueeze(0)

        pred_diff_incorrect = cf_predict_valid[incorrect_indices] - predict_valid[incorrect_indices] + 1e-6
        pred_loss_incorrect = torch.mean(torch.log1p(torch.exp(-pred_diff_incorrect + margin)))
        cf_loss += pred_loss_incorrect * lambda_negative

    try:
        if L_skill is not None and cf_L_skill is not None:
            epsilon_min = getattr(opt, 'epsilon_min', 0.1)
            epsilon_max = getattr(opt, 'epsilon_max', 0.5)
            valid_mask = maskEffLen.unsqueeze(-1).expand_as(L_skill)
            ell_t = L_skill - cf_L_skill

            consistency_loss = torch.zeros_like(ell_t)
            consistency_loss = torch.where(
                ell_t < epsilon_min,
                epsilon_min - ell_t,
                consistency_loss
            )
            consistency_loss = torch.where(
                ell_t > epsilon_max,
                ell_t - epsilon_max,
                consistency_loss
            )

            valid_consistency_loss = consistency_loss.masked_select(valid_mask)
            if valid_consistency_loss.numel() > 0:
                cf_loss += valid_consistency_loss.mean()
    except Exception as e:
        print(f"知识状态一致性约束计算失败: {str(e)}")
        pass

    return cf_loss
