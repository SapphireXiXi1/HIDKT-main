import torch
from torch.nn.utils import clip_grad_norm_
from P6_utils import calculate_auc, calculate_acc, selecting_mask, calculate_lossGlobal, calculate_lossParams, \
    dual_stream_contrast_loss


def train(model, dataloaders, optimizer, opt):
    if not hasattr(opt, 'cf_soft_topk_temp'):
        opt.cf_soft_topk_temp = 0.5
    if not hasattr(opt, 'cf_soft_topk_k'):
        opt.cf_soft_topk_k = 2.0

    def run_epoch(train_eval_test):
        predictRow_epoch_cpu = []
        nextLabelRow_epoch_cpu = []
        n_batch = 0

        for n_batch, batch in enumerate(dataloaders[train_eval_test], start=1):
            effLens_batch, \
            currQuestionAddLabel_batch, currQuestionID_batch, \
            currSkillAddLabel_batch, currSkillID_batch, currSkill_oneHot_batch, \
            currLabel_batch, \
            nextQuestionID_batch, \
            nextSkillID_batch, nextSkill_oneHot_batch, \
            nextLabel_batch = batch

            device = opt.DEVICE
            effLens_batch = effLens_batch.to(device)
            currQuestionAddLabel_batch = currQuestionAddLabel_batch.to(device)
            currSkillAddLabel_batch = currSkillAddLabel_batch.to(device)
            nextQuestionID_batch = nextQuestionID_batch.to(device)
            nextSkillID_batch = nextSkillID_batch.to(device)
            nextSkill_oneHot_batch = nextSkill_oneHot_batch.to(device)
            nextLabel_batch = nextLabel_batch.to(device)

            bsz, maxLen_batch = currSkillAddLabel_batch.size()
            maskEffLen_batch = selecting_mask(effLen=effLens_batch, maxLen=maxLen_batch, opt=opt)

            nextLabelRow_batch = nextLabel_batch.masked_select(maskEffLen_batch)
            if nextLabelRow_batch.numel() > 0:
                nextLabelRow_epoch_cpu.append(nextLabelRow_batch.detach().cpu())

            if train_eval_test == 'train':
                model.train()
            else:
                model.eval()

            with torch.set_grad_enabled(train_eval_test == 'train'):
                seq_lens_batch = effLens_batch.view(-1)

                cf_emb_skill, cf_emb_ques = model.generate_counterfactual_soft_embeddings(
                    currSkillAddLabel_batch,
                    currQuestionAddLabel_batch,
                    nextSkillID_batch,
                    seq_lens=seq_lens_batch,
                    temperature=opt.cf_soft_topk_temp,
                    k_soft=opt.cf_soft_topk_k
                )

                factual_output, _, counterfactual_output, _ = model.dual_encoder(
                    (currSkillAddLabel_batch, currQuestionAddLabel_batch),
                    counterfactual_input=None,
                    nextSkillID=nextSkillID_batch,
                    counterfactual_embedded_input=(cf_emb_skill, cf_emb_ques),
                    seq_lens=seq_lens_batch,
                )

                predict_batch, predictAllSkill_batch, params = model.predict_from_output(
                    factual_output, nextSkillID_batch, nextQuestionID_batch, nextSkill_oneHot_batch
                )
                L_skill, Diff, G, S = params

                cf_predict_batch, cf_predictAllSkill_batch, cf_params = model.predict_from_output(
                    counterfactual_output, nextSkillID_batch, nextQuestionID_batch, nextSkill_oneHot_batch
                )
                cf_L_skill, cf_Diff, cf_G, cf_S = cf_params

            predictRow_batch = predict_batch.masked_select(maskEffLen_batch)
            if predictRow_batch.numel() > 0:
                predictRow_epoch_cpu.append(predictRow_batch.detach().cpu())

            if train_eval_test == 'train':
                optimizer.zero_grad()

                lossD_batch = calculate_lossGlobal(predict_batch, nextLabel_batch, maskEffLen_batch)
                lossParam_batch, lossList = calculate_lossParams([L_skill, G, S],
                                                                 nextSkill_oneHot_batch, effLens_batch, opt)

                margin, lambda_positive, lambda_negative = model.get_contrast_params()
                cf_loss = dual_stream_contrast_loss(
                    predict_batch, cf_predict_batch,
                    L_skill, cf_L_skill,
                    nextLabel_batch, maskEffLen_batch,
                    opt,
                    margin, lambda_positive, lambda_negative
                )

                lambda1 = model.get_lambda1()
                total_loss = lossD_batch * (1.0 - lambda1) + cf_loss * lambda1

                if not torch.isfinite(total_loss):
                    print("[Warn] non-finite total_loss detected, skip this batch")
                    continue

                total_loss.backward()

                for p in model.parameters():
                    if p.grad is not None:
                        non_finite = ~torch.isfinite(p.grad)
                        if non_finite.any():
                            p.grad[non_finite] = 0.0

                clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        if n_batch == 0:
            print(f"Warning: No batches processed for {train_eval_test}")
            return 0.5, 0.5

        if len(predictRow_epoch_cpu) == 0 or len(nextLabelRow_epoch_cpu) == 0:
            print(f"Warning: No predictions or labels collected for {train_eval_test}")
            return 0.5, 0.5

        predictRow_epoch = torch.cat(predictRow_epoch_cpu)
        nextLabelRow_epoch = torch.cat(nextLabelRow_epoch_cpu)

        auc_epoch = calculate_auc(predictRow_epoch, nextLabelRow_epoch)
        acc_epoch = calculate_acc(predictRow_epoch, nextLabelRow_epoch)

        print('epoch %3d %5s || acc=%.4f  auc=%.4f' % (epoch, train_eval_test, acc_epoch, auc_epoch))

        return auc_epoch, acc_epoch

    bestEpoch = 1
    bestAucEval = -1.0
    bestAccEval = -1.0
    bestAucTest = -1.0
    bestAccTest = -1.0
    early_stop_patience = getattr(opt, 'early_stop_patience', 10)
    aucTestList = []
    accTestList = []

    for epoch in range(1, opt.n_epoch + 1):
        print('epoch %3d' % epoch)
        aucTrain, accTrain = run_epoch('train')
        aucEval, accEval = run_epoch('eval')
        aucTest, accTest = run_epoch('test')

        aucTestList.append(aucTest)
        accTestList.append(accTest)

        if aucEval > bestAucEval:
            bestAucEval = aucEval
            bestAccEval = accEval
            bestEpoch = epoch
            bestAucTest = aucTest
            bestAccTest = accTest
            print(f'✓ 验证集新最佳 (epoch {epoch}, eval AUC: {bestAucEval:.6f}, '
                  f'test AUC: {bestAucTest:.6f}, test Acc: {bestAccTest:.6f})')

        epochs_no_improve = epoch - bestEpoch
        if epochs_no_improve > 0:
            if epochs_no_improve == 1:
                print(f'早停监控: 验证集未创新高，开始计数 (1/{early_stop_patience})')
            else:
                print(f'早停监控: {epochs_no_improve}/{early_stop_patience}')

        if epochs_no_improve >= early_stop_patience:
            print(f'早停触发: 当前epoch {epoch}, 最佳验证集epoch {bestEpoch}, patience={early_stop_patience}')
            break

    bestAucTest_final = aucTestList[bestEpoch - 1] if len(aucTestList) >= bestEpoch else bestAucTest
    bestAccTest_final = accTestList[bestEpoch - 1] if len(accTestList) >= bestEpoch else bestAccTest
    print('\n' + '=' * 60)
    print('训练完成！')
    print('=' * 60)
    print(f'最佳验证集epoch: {bestEpoch} (AUC: {bestAucEval:.6f}, Acc: {bestAccEval:.6f})')
    print(f'验证集最佳epoch对应测试集: AUC: {bestAucTest_final:.6f}, Acc: {bestAccTest_final:.6f}')
    print('=' * 60)
