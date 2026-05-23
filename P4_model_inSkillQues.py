import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamEncoder(nn.Module):

    def __init__(self, nSkill, nQues, szRnnIn, szRnnOut, nRnnLayer, dropout, opt):
        super(DualStreamEncoder, self).__init__()
        self.encoderSkillLabel = nn.Embedding(num_embeddings=2 * nSkill + 10, embedding_dim=szRnnIn, padding_idx=0)
        self.encoderQuesLabel = nn.Embedding(num_embeddings=2 * nQues + 10, embedding_dim=szRnnIn, padding_idx=0)
        self.nSkill = nSkill
        self.nQues = nQues

        self.shared_rnn = nn.LSTM(input_size=szRnnIn, hidden_size=szRnnOut,
                                  num_layers=nRnnLayer, batch_first=True, dropout=dropout)

        self.attn_dim = szRnnOut
        self.query_proj = nn.Linear(szRnnOut, self.attn_dim)
        self.key_proj = nn.Linear(szRnnOut, self.attn_dim)
        self.value_proj = nn.Linear(szRnnOut, szRnnOut)
        self.attn_combine = nn.Linear(2 * szRnnOut, szRnnOut)

        self.stream_interaction = nn.Linear(2 * szRnnOut, szRnnOut)
        self.gamma_int_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(0.1))))

        self.opt = opt
        self.nRnnLayer = nRnnLayer
        self.szRnnOut = szRnnOut

        self.time_decay = nn.Parameter(torch.tensor(0.01))

        self.decay_factor = nn.Parameter(torch.tensor(0.01))

        self.relevance_weights = nn.Parameter(torch.FloatTensor([1.0, 0.7, 0.3, 0.1]))
        self.time_decay_adjustments = nn.Parameter(torch.FloatTensor([1.2, 1.0, 0.8, 0.6]))
        self.time_decay_base = nn.Parameter(torch.tensor(0.5))
        self.relevance_importance = nn.Parameter(torch.tensor(0.6))
        self.time_importance = nn.Parameter(torch.tensor(0.4))

    @staticmethod
    def encode_attention_query_skill(nextSkillID, response_label=0):
        return 2 * nextSkillID + response_label

    @staticmethod
    def make_sequence_valid_mask(seq_lens, max_len, device=None):
        seq_lens = seq_lens.view(-1).long()
        if device is None:
            device = seq_lens.device
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        return positions < seq_lens.unsqueeze(1)

    @staticmethod
    def resolve_seq_lens(seq_lens, batch_size, max_len, device):
        if seq_lens is None:
            return None
        seq_lens = torch.as_tensor(seq_lens, dtype=torch.long, device=device).view(-1)
        if seq_lens.numel() == 1 and batch_size > 1:
            seq_lens = seq_lens.expand(batch_size)
        if seq_lens.numel() != batch_size:
            raise ValueError(
                f'seq_lens batch size {seq_lens.numel()} does not match batch_size {batch_size}'
            )
        return seq_lens.clamp(min=0, max=max_len)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.nRnnLayer, bsz, self.szRnnOut).zero_(),
                weight.new(self.nRnnLayer, bsz, self.szRnnOut).zero_())

    def apply_attention(self, rnn_output, query, mask, seq_positions, return_attention=False):
        key = self.key_proj(rnn_output)
        value = self.value_proj(rnn_output)

        B, T, _ = rnn_output.size()
        scores = torch.bmm(query, key.transpose(1, 2)) / (self.attn_dim ** 0.5)

        seq_positions = seq_positions.float()
        relative_positions = seq_positions.unsqueeze(1) / seq_positions.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        lambda_time = F.softplus(self.time_decay)
        decay = torch.exp(-lambda_time * (1 - relative_positions))
        decayed_scores = scores * decay

        if mask is not None:
            final_scores = decayed_scores.masked_fill(~mask, -1e9)
        else:
            final_scores = decayed_scores

        attn_weights = F.softmax(final_scores, dim=2)
        context = torch.bmm(attn_weights, value)
        output = self.attn_combine(torch.cat([rnn_output, context], dim=2))

        if return_attention:
            return output, attn_weights
        return output

    def forward(
            self,
            factual_input,
            counterfactual_input=None,
            nextSkillID=None,
            return_attention=False,
            counterfactual_embedded_input=None,
            seq_lens=None):
        currSkillAddLabel, currQuesAddLabel = factual_input
        bsz, maxLen = currSkillAddLabel.size()
        device = currSkillAddLabel.device

        factual_hidden = self.init_hidden(bsz)
        embSkillLabel = self.encoderSkillLabel(currSkillAddLabel)
        embQuesLabel = self.encoderQuesLabel(currQuesAddLabel)

        difficultySkillLabel = torch.sigmoid(embQuesLabel)

        factual_rnn_in = embSkillLabel * (1 + difficultySkillLabel)

        factual_output, factual_hidden = self.shared_rnn(factual_rnn_in, factual_hidden)

        seq_lens = self.resolve_seq_lens(seq_lens, bsz, maxLen, device)
        if seq_lens is None:
            valid_mask_1d = (currSkillAddLabel != 0)
        else:
            valid_mask_1d = self.make_sequence_valid_mask(seq_lens, maxLen, device)
        mask = valid_mask_1d.unsqueeze(1).expand(bsz, maxLen, maxLen)

        factual_attn_weights = None
        counterfactual_attn_weights = None

        if nextSkillID is not None:
            nextSkillAddLabel = self.encode_attention_query_skill(nextSkillID)
            nextSkillEmb = self.encoderSkillLabel(nextSkillAddLabel)
            query = self.query_proj(nextSkillEmb)
            seq_positions = torch.arange(maxLen, device=device).unsqueeze(0).repeat(bsz, 1)
            if return_attention:
                factual_output, factual_attn_weights = self.apply_attention(
                    factual_output, query, mask, seq_positions, return_attention=True
                )
            else:
                factual_output = self.apply_attention(factual_output, query, mask, seq_positions)

        if counterfactual_input is not None or counterfactual_embedded_input is not None:
            if counterfactual_embedded_input is not None:
                cf_embSkillLabel, cf_embQuesLabel = counterfactual_embedded_input
            else:
                cf_currSkillAddLabel, cf_currQuesAddLabel = counterfactual_input
                cf_embSkillLabel = self.encoderSkillLabel(cf_currSkillAddLabel)
                cf_embQuesLabel = self.encoderQuesLabel(cf_currQuesAddLabel)

            cf_difficultySkillLabel = torch.sigmoid(cf_embQuesLabel)

            cf_rnn_in = cf_embSkillLabel * (1 + cf_difficultySkillLabel)

            counterfactual_hidden = self.init_hidden(bsz)
            counterfactual_output, counterfactual_hidden = self.shared_rnn(cf_rnn_in, counterfactual_hidden)

            if nextSkillID is not None:
                seq_positions = torch.arange(maxLen, device=device).unsqueeze(0).repeat(bsz, 1)
                if return_attention:
                    counterfactual_output, counterfactual_attn_weights = self.apply_attention(
                        counterfactual_output, query, mask, seq_positions, return_attention=True
                    )
                else:
                    counterfactual_output = self.apply_attention(counterfactual_output, query, mask, seq_positions)

            factual_interaction = self.stream_interaction(
                torch.cat([factual_output, counterfactual_output], dim=2)
            )
            counterfactual_interaction = self.stream_interaction(
                torch.cat([counterfactual_output, factual_output], dim=2)
            )

            gamma_int = F.softplus(self.gamma_int_raw)
            factual_output = factual_output + gamma_int * factual_interaction
            counterfactual_output = counterfactual_output + gamma_int * counterfactual_interaction

            if return_attention:
                return (factual_output, factual_hidden, counterfactual_output, counterfactual_hidden,
                        (factual_attn_weights, counterfactual_attn_weights))
            return factual_output, factual_hidden, counterfactual_output, counterfactual_hidden

        if return_attention:
            return factual_output, factual_hidden, None, None, (factual_attn_weights, None)
        return factual_output, factual_hidden, None, None

    def calculate_importance_weights(self):
        rel_weights = torch.sigmoid(self.relevance_weights)
        time_adjustments = torch.sigmoid(self.time_decay_adjustments)
        time_decay_lambda = F.softplus(self.time_decay_base)

        weights_sum = self.relevance_importance + self.time_importance
        rel_importance = self.relevance_importance / weights_sum
        time_importance = self.time_importance / weights_sum

        return rel_weights, time_adjustments, time_decay_lambda, rel_importance, time_importance

    def compute_importance_scores(self, currSkillAddLabel, currQuesAddLabel, nextSkillID, seq_lens=None):
        del currQuesAddLabel
        device = currSkillAddLabel.device
        bsz, maxLen = currSkillAddLabel.size()

        rel_weights, time_adjustments, time_decay_lambda, _, _ = self.calculate_importance_weights()

        seq_lens = self.resolve_seq_lens(seq_lens, bsz, maxLen, device)
        if seq_lens is None:
            mask = (currSkillAddLabel != 0)
            seq_lens = mask.sum(dim=1).clamp(min=1)
        else:
            mask = self.make_sequence_valid_mask(seq_lens, maxLen, device)
            seq_lens = seq_lens.clamp(min=1)

        skill_ids = currSkillAddLabel // 2
        is_correct = (currSkillAddLabel % 2).float()
        target_skill_ids = nextSkillID

        positions = torch.arange(maxLen, device=device).unsqueeze(0).expand(bsz, -1).float()
        seq_lens_f = seq_lens.unsqueeze(1).float()
        base_decay = torch.exp(-time_decay_lambda * (seq_lens_f - positions) / seq_lens_f)

        related = (skill_ids == target_skill_ids) & mask
        correct_mask = (is_correct == 1.0) & mask

        case00 = related & correct_mask
        case01 = related & (~correct_mask)
        case10 = (~related) & correct_mask
        case11 = (~related) & (~correct_mask)

        rel_w = rel_weights.to(device)
        time_adj = time_adjustments.to(device)

        skill_relevance = (
            rel_w[0] * case00.float()
            + rel_w[1] * case01.float()
            + rel_w[2] * case10.float()
            + rel_w[3] * case11.float()
        )
        ta = (
            time_adj[0] * case00.float()
            + time_adj[1] * case01.float()
            + time_adj[2] * case10.float()
            + time_adj[3] * case11.float()
        )
        importance = (skill_relevance * (base_decay * ta)).masked_fill(~mask, float('-inf'))
        return importance, mask, seq_lens


class DKTInSkillQues(nn.Module):
    def __init__(self, nSkill, nQues, szRnnIn, szRnnOut, nRnnLayer, szOut, dropout, opt):
        super(DKTInSkillQues, self).__init__()
        self.dual_encoder = DualStreamEncoder(
            nSkill, nQues, szRnnIn, szRnnOut, nRnnLayer, dropout, opt
        )

        self.encoderNextSkill = nn.Embedding(num_embeddings=nSkill + 1, embedding_dim=szRnnOut, padding_idx=0)
        self.encoderNextQues = nn.Embedding(num_embeddings=nQues + 1, embedding_dim=szRnnOut, padding_idx=0)

        self.transL = nn.Linear(2 * szRnnOut, szOut)
        self.transDiff = nn.Linear(szRnnOut, szOut)
        self.transAlpha = nn.Linear(szRnnOut, szOut)
        self.transK = nn.Linear(2 * szRnnOut, szOut)
        self.transG = nn.Linear(2 * szRnnOut, szOut)
        self.transS = nn.Linear(2 * szRnnOut, szOut)

        self.sigmoid = nn.Sigmoid()
        self.opt = opt
        self.lambda1_logit = nn.Parameter(torch.tensor(0.0))
        self.contrast_margin_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(0.2))))
        self.contrast_positive_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(1.0))))
        self.contrast_negative_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(1.0))))

    def init_hidden(self, bsz):
        return self.dual_encoder.init_hidden(bsz)

    def get_lambda1(self):
        return torch.sigmoid(self.lambda1_logit)

    def get_contrast_params(self):
        margin = F.softplus(self.contrast_margin_raw)
        lambda_positive = F.softplus(self.contrast_positive_raw)
        lambda_negative = F.softplus(self.contrast_negative_raw)
        return margin, lambda_positive, lambda_negative

    @staticmethod
    def _split_add_label(add_label, num_id):
        del num_id
        raw_id = add_label // 2
        label = add_label % 2
        return raw_id, label

    @staticmethod
    def _compose_add_label(raw_id, label, num_id):
        del num_id
        return 2 * raw_id + label

    def predict_from_output(self, rnn_output, nextSkillID, nextQuesID, nextSkill_oneHot):
        embNextSkill = self.encoderNextSkill(nextSkillID)
        embNextQues = self.encoderNextQues(nextQuesID)
        difficultyNextSkill = self.sigmoid(embNextQues)
        nextInput = embNextSkill * (1 + difficultyNextSkill)
        nextFullInfo = torch.cat([rnn_output, nextInput], dim=2)

        L_skill = self.sigmoid(self.transL(nextFullInfo))
        Diff = self.sigmoid(self.transDiff(embNextQues))
        q_alpha = self.sigmoid(self.transAlpha(embNextQues))
        G = self.sigmoid(self.transG(nextFullInfo))
        S = self.sigmoid(self.transS(nextFullInfo))

        x = 4 * q_alpha * (L_skill - Diff)
        L = torch.sigmoid(x)

        c1 = L * (1 - S)
        c2 = (1 - L) * G

        predictAllSkill = c1 + c2
        predict = torch.sum(predictAllSkill * nextSkill_oneHot, dim=2)
        predict = torch.nan_to_num(predict, nan=0.5, posinf=1.0, neginf=0.0)
        predict = torch.clamp(predict, min=1e-6, max=1.0 - 1e-6)

        return predict, predictAllSkill, [L_skill, Diff, G, S]

    def forward(self, currSkillAddLabel, currQuesAddLabel, nextSkill_oneHot, nextSkillID, nextQuesID,
                seq_lens=None):
        factual_output, _, _, _ = self.dual_encoder(
            (currSkillAddLabel, currQuesAddLabel),
            nextSkillID=nextSkillID,
            seq_lens=seq_lens,
        )

        predict, predictAllSkill, params = self.predict_from_output(
            factual_output, nextSkillID, nextQuesID, nextSkill_oneHot
        )

        return predict, predictAllSkill, params

    def forward_counterfactual(self, currSkillAddLabel, currQuesAddLabel, nextSkill_oneHot, nextSkillID, nextQuesID,
                               nextLabel, seq_lens=None):
        cf_currSkillAddLabel, cf_currQuesAddLabel = self.generate_counterfactual_input(
            currSkillAddLabel, currQuesAddLabel, nextSkillID, seq_lens=seq_lens
        )

        factual_output, _, counterfactual_output, _ = self.dual_encoder(
            (currSkillAddLabel, currQuesAddLabel),
            (cf_currSkillAddLabel, cf_currQuesAddLabel),
            nextSkillID,
            seq_lens=seq_lens,
        )

        predict, predictAllSkill, params = self.predict_from_output(
            factual_output, nextSkillID, nextQuesID, nextSkill_oneHot
        )

        cf_predict, cf_predictAllSkill, cf_params = self.predict_from_output(
            counterfactual_output, nextSkillID, nextQuesID, nextSkill_oneHot
        )

        return cf_predict, cf_predictAllSkill, cf_params

    def init_prefix_cache(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return {
            'currSkillAddLabel': torch.empty(batch_size, 0, dtype=torch.long, device=device),
            'currQuesAddLabel': torch.empty(batch_size, 0, dtype=torch.long, device=device),
            'seq_lens': torch.zeros(batch_size, dtype=torch.long, device=device),
            'importance_scores': None,
            'topk_indices': None,
            'topk_scores': None,
        }

    def build_prefix_cache(self, currSkillAddLabel, currQuesAddLabel, seq_lens=None):
        bsz, max_len = currSkillAddLabel.size()
        device = currSkillAddLabel.device
        if seq_lens is None:
            seq_lens = torch.full((bsz,), max_len, dtype=torch.long, device=device)
        else:
            seq_lens = self.dual_encoder.resolve_seq_lens(seq_lens, bsz, max_len, device)
        return {
            'currSkillAddLabel': currSkillAddLabel,
            'currQuesAddLabel': currQuesAddLabel,
            'seq_lens': seq_lens,
            'importance_scores': None,
            'topk_indices': None,
            'topk_scores': None,
        }

    def append_prefix_cache(self, cache, currSkillID, currQuesID, currLabel):
        device = cache['currSkillAddLabel'].device
        currSkillID = torch.as_tensor(currSkillID, dtype=torch.long, device=device).view(-1, 1)
        currQuesID = torch.as_tensor(currQuesID, dtype=torch.long, device=device).view(-1, 1)
        currLabel = torch.as_tensor(currLabel, dtype=torch.long, device=device).view(-1, 1)

        currSkillAddLabel = self._compose_add_label(currSkillID, currLabel, self.dual_encoder.nSkill)
        currQuesAddLabel = self._compose_add_label(currQuesID, currLabel, self.dual_encoder.nQues)

        cache['currSkillAddLabel'] = torch.cat([cache['currSkillAddLabel'], currSkillAddLabel], dim=1)
        cache['currQuesAddLabel'] = torch.cat([cache['currQuesAddLabel'], currQuesAddLabel], dim=1)
        cache['seq_lens'] = cache['seq_lens'] + 1
        cache['importance_scores'] = None
        cache['topk_indices'] = None
        cache['topk_scores'] = None
        return cache

    def cached_step_predict(self, cache, nextSkillID, nextQuesID, return_cache=False):
        currSkillAddLabel = cache['currSkillAddLabel']
        currQuesAddLabel = cache['currQuesAddLabel']
        bsz, prefix_len = currSkillAddLabel.size()
        if prefix_len == 0:
            raise ValueError("cached_step_predict requires at least one cached historical interaction.")

        device = currSkillAddLabel.device
        nextSkillID = torch.as_tensor(nextSkillID, dtype=torch.long, device=device).view(bsz)
        nextQuesID = torch.as_tensor(nextQuesID, dtype=torch.long, device=device).view(bsz)

        target_skill_seq = nextSkillID.unsqueeze(1).expand(-1, prefix_len)
        target_ques_seq = nextQuesID.unsqueeze(1).expand(-1, prefix_len)
        nextSkill_oneHot = F.one_hot(
            nextSkillID, num_classes=self.dual_encoder.nSkill + 1
        ).float().unsqueeze(1).expand(-1, prefix_len, -1)

        importance, _, seq_lens = self.dual_encoder.compute_importance_scores(
            currSkillAddLabel, currQuesAddLabel, target_skill_seq, seq_lens=cache['seq_lens']
        )
        k = min(2, prefix_len)
        topk_scores, topk_indices = torch.topk(importance, k=k, dim=1)
        cache['importance_scores'] = importance.detach()
        cache['topk_scores'] = topk_scores.detach()
        cache['topk_indices'] = topk_indices.detach()
        cache['seq_lens'] = seq_lens.detach()

        cf_currSkillAddLabel, cf_currQuesAddLabel = self.generate_counterfactual_input(
            currSkillAddLabel,
            currQuesAddLabel,
            target_skill_seq,
            seq_lens=cache['seq_lens'],
            precomputed_importance=importance,
            precomputed_seq_lens=seq_lens
        )
        factual_output, _, counterfactual_output, _ = self.dual_encoder(
            (currSkillAddLabel, currQuesAddLabel),
            (cf_currSkillAddLabel, cf_currQuesAddLabel),
            nextSkillID=target_skill_seq,
            seq_lens=cache['seq_lens'],
        )

        predict, predictAllSkill, params = self.predict_from_output(
            factual_output, target_skill_seq, target_ques_seq, nextSkill_oneHot
        )
        cf_predict, cf_predictAllSkill, cf_params = self.predict_from_output(
            counterfactual_output, target_skill_seq, target_ques_seq, nextSkill_oneHot
        )

        last_idx = (seq_lens - 1).clamp(min=0)
        batch_idx = torch.arange(bsz, device=device)
        result = {
            'predict': predict[batch_idx, last_idx],
            'cf_predict': cf_predict[batch_idx, last_idx],
            'predict_all_skill': predictAllSkill[batch_idx, last_idx],
            'cf_predict_all_skill': cf_predictAllSkill[batch_idx, last_idx],
            'params': params,
            'cf_params': cf_params,
            'topk_indices': topk_indices,
            'topk_scores': topk_scores,
        }
        if return_cache:
            result['cache'] = cache
        return result

    def generate_counterfactual_input(self, currSkillAddLabel, currQuesAddLabel, nextSkillID,
                                      seq_lens=None, precomputed_importance=None, precomputed_seq_lens=None):
        device = currSkillAddLabel.device
        bsz, maxLen = currSkillAddLabel.size()
        if precomputed_importance is None or precomputed_seq_lens is None:
            importance, mask, seq_lens = self.dual_encoder.compute_importance_scores(
                currSkillAddLabel, currQuesAddLabel, nextSkillID, seq_lens=seq_lens
            )
        else:
            importance = precomputed_importance
            seq_lens = precomputed_seq_lens

        k = min(2, maxLen)
        topv, topidx = torch.topk(importance, k=k, dim=1)

        to_modify = torch.zeros_like(currSkillAddLabel, dtype=torch.bool, device=device)
        for i in range(k):
            idx_i = topidx[:, i]
            batch_idx = torch.arange(bsz, device=device)
            valid_pos = idx_i < seq_lens.unsqueeze(1).squeeze(1)
            to_modify[batch_idx[valid_pos], idx_i[valid_pos]] = True

        cf_currSkillAddLabel = currSkillAddLabel.clone()
        cf_currQuesAddLabel = currQuesAddLabel.clone()

        modify_indices = to_modify.nonzero(as_tuple=False)
        if modify_indices.size(0) > 0:
            batch_indices = modify_indices[:, 0]
            seq_indices = modify_indices[:, 1]

            skill_ids_to_modify, orig_labels = self._split_add_label(
                cf_currSkillAddLabel[batch_indices, seq_indices], self.dual_encoder.nSkill
            )
            ques_ids_to_modify, _ = self._split_add_label(
                cf_currQuesAddLabel[batch_indices, seq_indices], self.dual_encoder.nQues
            )
            cf_labels = 1 - orig_labels

            cf_currSkillAddLabel[batch_indices, seq_indices] = self._compose_add_label(
                skill_ids_to_modify, cf_labels, self.dual_encoder.nSkill
            )
            cf_currQuesAddLabel[batch_indices, seq_indices] = self._compose_add_label(
                ques_ids_to_modify, cf_labels, self.dual_encoder.nQues
            )

        return cf_currSkillAddLabel, cf_currQuesAddLabel

    def generate_counterfactual_soft_embeddings(
            self,
            currSkillAddLabel,
            currQuesAddLabel,
            nextSkillID,
            seq_lens=None,
            temperature=0.5,
            k_soft=2.0,
            return_gate=False):
        device = currSkillAddLabel.device

        importance, mask, _ = self.dual_encoder.compute_importance_scores(
            currSkillAddLabel, currQuesAddLabel, nextSkillID, seq_lens=seq_lens
        )
        safe_importance = importance.masked_fill(~mask, -1e9)

        tau = max(float(temperature), 1e-4)
        logits = safe_importance / tau
        probs = torch.softmax(logits, dim=1)
        probs = torch.where(mask, probs, torch.zeros_like(probs))
        probs_sum = probs.sum(dim=1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, torch.zeros_like(probs))
        gate = probs * float(k_soft)
        gate = gate.clamp(min=0.0, max=1.0)
        gate_exp = gate.unsqueeze(-1)

        mask_long = mask.long()
        skill_ids, orig_label = self._split_add_label(currSkillAddLabel, self.dual_encoder.nSkill)
        ques_ids, _ = self._split_add_label(currQuesAddLabel, self.dual_encoder.nQues)
        flip_label = 1 - orig_label
        flipped_skill = self._compose_add_label(skill_ids, flip_label, self.dual_encoder.nSkill) * mask_long
        flipped_ques = self._compose_add_label(ques_ids, flip_label, self.dual_encoder.nQues) * mask_long

        emb_skill_orig = self.dual_encoder.encoderSkillLabel(currSkillAddLabel)
        emb_ques_orig = self.dual_encoder.encoderQuesLabel(currQuesAddLabel)
        emb_skill_flip = self.dual_encoder.encoderSkillLabel(flipped_skill)
        emb_ques_flip = self.dual_encoder.encoderQuesLabel(flipped_ques)

        cf_emb_skill = (1.0 - gate_exp) * emb_skill_orig + gate_exp * emb_skill_flip
        cf_emb_ques = (1.0 - gate_exp) * emb_ques_orig + gate_exp * emb_ques_flip

        if return_gate:
            return cf_emb_skill, cf_emb_ques, gate
        return cf_emb_skill, cf_emb_ques