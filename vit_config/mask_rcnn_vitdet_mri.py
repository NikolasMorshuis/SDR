from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
import detectron2.data.transforms as T

#from detectron2_github.projects.ViTDet.configs.common.coco_loader_lsj import dataloader
# Data using LSJ

# Dataloader:
image_size = 640#640
dataloader = model_zoo.get_config("common/data/coco.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    #L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
    #T.RandomApply(T.RandomBrightness(0.8, 1.2), 0.5),
    #T.RandomApply(T.RandomContrast(0.8, 1.2), 0.5),
    #T.RandomApply(T.RandomBlur(kernel_size_min=5, kernel_size_max=15), 0.5),
    #T.RandomApply(T.RandomNoise(std=0.2), 0.5),
]

dataloader.train.dataset.names = "fastmri_knee_train"
dataloader.test.dataset.names = "fastmri_knee_val"
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = 8  #TODO: 4 for a100
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = False
dataloader.train.mapper.use_instance_mask = False

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]
model = model_zoo.get_config("common/models/mask_rcnn_vitdet_mri_knee.py").model

model.roi_heads.num_classes = 22
#model.roi_heads.box_predictor.test_score_thresh = 0.02
model.roi_heads.mask_head = None
model.roi_heads.mask_pooler = None
model.roi_heads.mask_in_features = None

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)

train.output_dir = "./output/vitdet_mri_mask_rcnn"

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 100000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[75000, 90000],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
optimizer.lr = 5e-5 #optimizer.lr #* dataloader.train.total_batch_size / 16