import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_head
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class CondInst(SingleStageDetector):
    """Implementation of `CondInst <https://arxiv.org/abs/2003.05664>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_branch,
                 mask_head,
                 segm_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CondInst, self).__init__(backbone, neck, bbox_head, train_cfg,
                                       test_cfg, pretrained, init_cfg)
        self.mask_branch = build_head(mask_branch)
        self.mask_head = build_head(mask_head)
        self.segm_head = None if segm_head is None else \
                build_head(segm_head)

    def forward_dummy(self, img):
        raise NotImplementedError

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        if gt_masks is not None:
            H, W = img.size(2), img.size(3)
            tensor_masks = []
            for masks in gt_masks:
                masks = masks.expand(H, W, 0, 0)
                tensor_masks.append(
                    masks.to_tensor(dtype=torch.uint8, device=img.device))              # gt转tensor
            gt_masks = tensor_masks

        # 特征图x，有多层(FPN)，可以直接用
        x = self.extract_feat(img)                                                      # 特征提取网络(backbone)
        
        # FCOS输出，param_pred对每一个位置都有输出，没有经过筛选
        cls_score, bbox_pred, centerness, param_pred = \
                self.bbox_head(x, self.mask_head.param_conv)                            # 分类回归、中心度(FCOS)
                                                                                        # param_conv输入为box_head输出的mask_head的参数，写在mask_head里
        bbox_head_loss_inputs = (cls_score, bbox_pred, centerness) + (
            gt_bboxes, gt_labels, img_metas)                                            # 打包predict和gt
        
        # FCOS损失，根据GT筛选出进行损失计算的位置索引。这些索引长度(size(0))都是inst_num, 与每层的特征图大小、层数、batch_size有关，
        # 如果直接用CondInst的结构则可以直接用
        losses, coors, level_inds, img_inds, gt_inds = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)                  # bbox_head输出：损失、坐标、金字塔层索引、图像索引、gt索引(哪些位置被GT标为前景)
        
        # mask_branch输出，使用P3\P4\P5三层，可以直接用
        mask_feat = self.mask_branch(x)                                                 # x输入mask_branch, 得到输出
        if self.segm_head is not None:
            segm_pred = self.segm_head(x[0])
            loss_segm = self.segm_head.loss(segm_pred, gt_masks, gt_labels)
            losses.update(loss_segm)                                                    # 这个是什么, 原文里没提过, 说不定是之后能够添加的另外的head。另提一句，loss一般和model放在一块
        
        # 用于采样输入
        inputs = (cls_score, centerness, param_pred, coors, level_inds, img_inds, gt_inds)
        # 采样，若gt_ind超过阈值则随机取其中一部分
        param_pred, coors, level_inds, img_inds, gt_inds = self.mask_head.training_sample(*inputs)
        
        # 要改应该就只在下面两个部分改
        # 使用mask_feat作为输入以输出预测结果，包括(位置编码、mask_head两个部分)
        mask_pred = self.mask_head(mask_feat, param_pred, coors, level_inds, img_inds)  # mask_feat: mask_branch的输出; praram_pred: bbox_head预测mask_head的参数; coors、level_inds、img_inds: 都是bbox的输出，但是不知道是啥
        # img、img_metas与pairwise_loss的相似度有关，gt_ind采样gt_labels(或者boxInst的gt_bboxes)与mask_pred算损失
        loss_mask = self.mask_head.loss(img, img_metas, mask_pred, gt_inds, gt_bboxes,
                                        gt_masks, gt_labels)
        losses.update(loss_mask)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        outputs = self.bbox_head.simple_test(
            feat, self.mask_head.param_conv, img_metas, rescale=rescale)                # output：tuple(tensor,...)
        det_bboxes, det_labels, det_params, det_coors, det_level_inds = zip(*outputs)
               # 第一个是tensor(batch_size, num_inst,5)，第二个是tensor(batch_size, num_inst,)[包括背景类]，后面是mlvl_param_pred、mlvl_coors、lvl_inds(经过nms后)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]
        # bbox_results: list(list(array)), len(第一个list) = batch_size，len(第二个list) = num_classes，len(ndarray)是这个类别的实例数

        mask_feat = self.mask_branch(feat)
        mask_results = self.mask_head.simple_test(
            mask_feat,
            det_labels,
            det_params,
            det_coors,
            det_level_inds,
            img_metas,
            self.bbox_head.num_classes,
            rescale=rescale)
        return list(zip(bbox_results, mask_results))    # list(tuple(list(ndarray),list(ndarray)))，
                                                        # zip的作用就是把相同维度的list按照相同下标组合在一起形成tuple，最后由这些tuple形成list

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
