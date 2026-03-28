"""
仅包含「编码器 → 融合头 → 门控加权求和」路径，与训练模型在 eval() 下
对 fused logits 的计算一致，且不执行队列 / KMeans / 动量更新，便于 ONNX 导出。

说明：与完整 `ImageBaseClusterDistancePlusGatingModel.forward` 相比，去掉了
辅助损失与张量副作用；分类用的 `output` 张量应与原模型在同输入下一致（Dropout 在 eval 关闭）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _generate_attention_sequence(num_experts: int, num_features: int) -> list:
    sequence = []
    sequence.append(num_features - 1)
    if num_experts > 1:
        for i in range(num_experts - 1):
            sequence.append((num_features - 2 - i % (num_features - 1)) % num_features)
    return sequence


class ImageBaseInferenceWrapper(nn.Module):
    def __init__(self, source: nn.Module):
        super().__init__()
        self.encoder = source.encoder
        self.proj = source.proj
        self.attention = source.attention
        self.cross_attention = source.cross_attention
        self.fusion_fc_list = source.fusion_fc_list
        self.gating = source.gating
        self.num_experts = source.num_experts

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_feature_list = self.encoder(image)
        feat_last = F.avg_pool2d(
            img_feature_list[-1], img_feature_list[-1].size()[3]
        ).view(img_feature_list[-1].size(0), -1)
        features_flatten = F.normalize(self.proj(feat_last), dim=1)

        index = _generate_attention_sequence(self.num_experts, len(img_feature_list))
        attentioned_features = []
        feat_last_map = img_feature_list[-1]
        for idx in range(len(index)):
            att_feat = self.attention[idx](img_feature_list[index[idx]], feat_last_map)
            attentioned_features.append(att_feat)
        img_features_list = [
            F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in attentioned_features
        ]
        aligned_features = []
        for idx in range(self.num_experts):
            aligned_feat = self.cross_attention[idx](img_features_list[idx], img_features_list[idx])
            aligned_features.append(aligned_feat)

        outputs = [self.fusion_fc_list[idx](aligned_features[idx]) for idx in range(self.num_experts)]
        outputs_tensor = torch.stack(outputs, dim=1)
        adaptive_weight = self.gating(features_flatten)
        final_outputs = torch.einsum("bc,bcn->bcn", adaptive_weight, outputs_tensor)
        final_outputs_summed = final_outputs.sum(dim=1)
        return final_outputs_summed, outputs_tensor
