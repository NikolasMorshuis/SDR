import os
import json
import functools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path

from physics.mri import MulticoilMRI
from utils import CG, clear

from data.dataset import GetMRI_Fastmri
import ultralytics.utils.ops as ops
from detectron2.config import LazyConfig, instantiate
from PIL import Image
from detectron2.layers import batched_nms
import yaml
import argparse
import time

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes, Instances
from yolo_functions.fastmri_plus import npy_to_png

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def perturb_gt_boxes(gt_boxes):
    gt_boxes = gt_boxes[0][:, 1:]
    # change size between 0.75 and 1.25
    width_change = torch.rand(gt_boxes.shape[0]) * 0.5 + 0.75
    height_change = torch.rand(gt_boxes.shape[0]) * 0.5 + 0.75
    gt_boxes[:, 2] = gt_boxes[:, 2] * width_change
    gt_boxes[:, 3] = gt_boxes[:, 3] * height_change

    # change position between -0.25 and 0.25 width and height
    width_shift = torch.rand(gt_boxes.shape[0]) * 0.25 * gt_boxes[:, 2]
    height_shift = torch.rand(gt_boxes.shape[0]) * 0.25 * gt_boxes[:, 3]
    # plus or minus?
    plus_minus = torch.randint(0, 2, [gt_boxes.shape[0], 2])
    plus_minus[plus_minus == 0] = -1
    gt_boxes[:, 0] = gt_boxes[:, 0] + width_shift * plus_minus[:, 0]
    gt_boxes[:, 1] = gt_boxes[:, 1] + height_shift * plus_minus[:, 1]
    return gt_boxes



def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--sigma_y', type=float, default=0., help='Noise level')
    parser.add_argument('--use_proposed_bbox', action='store_true', help='Use proposed bounding boxes')
    parser.add_argument("--gamma", type=float, default=5.0, help='Gamma for CG')
    parser.add_argument("--debug", action='store_true', help='Debugging mode')
    parser.add_argument('--acc_factor', type=int, default=12, help='Acceleration factor')
    parser.add_argument('--vit_rob', type=str, default='imagenet', help='imagenet or fastmri or none')
    parser.add_argument('--baserec', type=str, default='ddip', help='ddip or csgm or l1 or e2e')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args_and_config()

    debug = args.debug
    n_reps = 1
    enhance_div = True
    epsnorm_pert = 3
    detection_model_type = 'robust'
    acc_factor = args.acc_factor
    img_size = 320
    optimization_steps = 10
    lr = 0.02
    n_recs = 3
    mirroring = False
    perturb_gt_boxes_yes = True
    # set seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    save_root = f'{config["paths"]["output_dir"]}/{args.baserec}/acc{args.acc_factor}'
    print('save_root: ', save_root)
    irl_types = ['input', 'label', 'progress', 'input_png', 'recon', 'label_png', 'recon_png', 'mask', 'images']
    for t in irl_types:
        save_root_f = os.path.join(save_root, t)
        Path(save_root_f).mkdir(parents=True, exist_ok=True)

    init_recon_dir = f'{config["paths"]["initial_recons"]}/{args.baserec}/acc{args.acc_factor}'

    dataset_root = config['paths']['fastmri_data']
    split = 'val'
    data_dir = os.path.join(dataset_root, f"multicoil_{split}")
    map_dir = os.path.join(dataset_root, f'multicoil_{split}_maps')

    dataset = GetMRI_Fastmri(dataset_root=dataset_root, split=split, acc=0.0, normalize=True,
                             annotation_dir='./data/labels_yolo_knee')

    test_patients = json.load(open('./data/test_patients.json'))
    # For demo purposes, we just look at one test-patient:
    test_patients = ['file1000277']
    idx_with_annotation, volume_names, slice_idxs = dataset.get_idx_with_annotations(
        patients=test_patients)

    idx_with_annotation = idx_with_annotation  # Normally we select every second slice with this: [::2]
    print(f"Number of images with annotations selected: {len(idx_with_annotation)}")

    subset = torch.utils.data.Subset(dataset, idx_with_annotation)  # idx_with_annotation

    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    loader = iter(loader)

    mask_dict = pickle.load(open('data/test_masks.pkl', 'rb'))

    # Load the model:
    from detectron2.engine.defaults import create_ddp_model

    cfg = LazyConfig.load('vit_config/mask_rcnn_vitdet_mri.py')
    detection_model = instantiate(cfg.model)
    detection_model.to('cuda')
    detection_model = create_ddp_model(detection_model)
    DetectionCheckpointer(detection_model).load(
        './models/vitdet_clean.pth'
    )
    if args.vit_rob == 'imagenet':
        state_dict = torch.load(
            './models/vitdet_robust_ft_in.pt'
        )
        detection_model.backbone.load_state_dict(state_dict, strict=False)
    elif args.vit_rob == 'fastmri':
        state_dict = torch.load(
            './models/vitdet_robust_ft_fastmri.pt'
        )
        detection_model.backbone.load_state_dict(state_dict, strict=False)
    detection_model.eval()
    with open('data/label_map.yaml', 'r') as f:
        label_map_dict = yaml.load(f, Loader=yaml.FullLoader)
    categories = [label_map_dict[k] for k in label_map_dict.keys()]
    MetadataCatalog.get("fastmri_knee_train").set(thing_classes=categories)
    epsnorm_pert = epsnorm_pert

    proposals_final = None

    for c, idx in enumerate(range(len(subset))):
        outputs_with_annotation = []
        outputs_with_annotation_attacked = []
        outputs_imgs = []
        outputs_imgs_attacked = []
        outputs_npy_attacked = []
        outputs_npy = []
        undersampled_img = None
        # if c < 22:
        #   continue
        proposals_final = None
        x_opt_orig = None
        item = next(loader)
        slice_name = item[7]
        if args.debug:
            if slice_name[0] != 'file1000277_018':
                continue
        volume_name = item[7][0].split('_')[0]
        x_orig = item[0].type(torch.complex64)
        mps = item[1]
        mps_orig = mps
        norm_constant = item[2]
        gt_boxes = item[6]
        tic = time.time()
        gt_image = npy_to_png(x_orig.numpy()[0, 0], to_pil=False).permute(1, 2, 0)
        v = Visualizer(
            gt_image.numpy(),
            metadata=MetadataCatalog.get('fastmri_knee_train'),  # cfg.DATASETS.TRAIN[0]),
            scale=1.0,
        )
        # TODO: add gt boxes
        annotations_list = []
        for box in gt_boxes[0].numpy():
            # TODO: transform box
            gt_boxes_xyxy = ops.xywh2xyxy(box[1:])
            # v.draw_box(gt_boxes_xyxy*gt_image.shape[0])
            annotations_list.append(
                {"bbox": [int(x_coordinate * 320) for x_coordinate in gt_boxes_xyxy], "bbox_mode": 0,
                 "category_id": int(box[0])})

        gt_boxes_xyxy = ops.xywh2xyxy(gt_boxes[0][:, 1:])
        annotations = {
            "annotations": annotations_list
        }
        out = v.draw_dataset_dict(annotations)
        outputs_with_annotation.append(out.get_image())
        outputs_imgs.append(out.img)

        x_orig = x_orig.repeat(n_reps, 1, 1, 1)
        mps = mps.repeat(n_reps, 1, 1, 1)
        # MRI forward operator
        num_successful = 0
        num_tried = 0
        batch_el = 0
        print(f"Slice {slice_name[0]}")

        mask = mask_dict[volume_name][-acc_factor // 2]
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
        mask = mask.repeat(1, 1, img_size, 1)
        A_funcs = MulticoilMRI(mask=mask)

        # Alias
        A = lambda z, mps: A_funcs._A(z, mps)
        AT = lambda z, mps: A_funcs._AT(z, mps)
        Ap = lambda z, mps: A_funcs._Adagger(z, mps)

        A_nomask = lambda z, mps: A_funcs._A(z, mps, use_mask=False)
        AT_nomask = lambda z, mps: A_funcs._AT(z, mps, use_mask=False)

        def Acg(x, mps, gamma):
            return x + gamma * A_funcs._AT(A_funcs._A(x, mps), mps)

        y = torch.zeros_like(mps_orig)
        ATy = torch.zeros_like(x_orig)
        for idx in range(x_orig.shape[0]):
            x_idx = x_orig[idx:idx + 1, ...].cuda()
            mps_idx = mps_orig[idx:idx + 1, ...].cuda()
            y_idx = A(x_idx, mps_idx)
            y += torch.randn_like(y) * args.sigma_y
            ATy_idx = AT(y_idx, mps_idx)
            if undersampled_img == None:
                undersampled_img = npy_to_png(ATy_idx[0, 0].abs().cpu().detach(), to_pil=False).permute(1, 2, 0)
            y[idx, ...] = y_idx
            ATy[idx, ...] = ATy_idx
            input = np.abs(clear(ATy_idx))
            label = np.abs(clear(x_idx))

        # Load reconstructions
        # check if file exists:
        init_recon_filepath = os.path.join(init_recon_dir, f'{slice_name[0]}_00.npy')
        if not os.path.exists(init_recon_filepath):
            assert False, f"Reconstruction file {init_recon_filepath} does not exist. Please Download the file from Nextcloud first."

        x_rec_orig = torch.from_numpy(np.load(os.path.join(init_recon_dir, f'{slice_name[0]}_00.npy')))
        if args.baserec == 'csgm':
            x_rec_orig = x_rec_orig/norm_constant
            x_rec_orig = x_rec_orig.type(torch.complex64)

        eps_adv = torch.randn((n_recs, 1, 320, 320), dtype=torch.complex64, device='cuda')
        eps_adv = eps_adv / (eps_adv.norm(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + 1e-8) * epsnorm_pert
        eps_adv[0, ...] = 0

        # Create slight variations of x_rec_orig
        x_rec_orig = x_rec_orig.unsqueeze(0).cuda()#.repeat(5, 1, 1, 1)

        xs = []
        features_finals = []
        box_features_final = []
        all_features_final = []
        reconstructions = []
        all_distances = []
        num_repeats = 0
        num_attacks = n_recs
        eps_turn = 0

        optimizer = torch.optim.Adam([eps_adv], lr=lr)
        optimizer.zero_grad()
        idx = 0
        ATy_idx = ATy[idx:idx + 1, ...].cuda()
        y_idx = y[idx:idx + 1, ...].cuda()
        mps_idx = mps_orig[idx:idx + 1, ...].cuda()
        Acg_idx = functools.partial(Acg, mps=mps_idx, gamma=args.gamma)
        first_time = True

        for i in range(optimization_steps*num_attacks):
            enable_grad = num_repeats >= num_attacks
            with torch.set_grad_enabled(enable_grad):
                if enable_grad:
                    eps_adv.requires_grad = True
                x_rec = x_rec_orig.detach().clone()

                if first_time:
                    x_rec = x_rec + eps_adv[eps_turn:eps_turn + 1]
                else:
                    x_rec = x_rec + eps_adv[eps_turn+1:eps_turn + 2]

                bcg = x_rec + args.gamma * ATy_idx
                x_rec = CG(Acg_idx, bcg, x_rec, n_inner=5)

                y_pred = A_nomask(x_rec, mps_idx)
                maskbool = mask.bool().repeat(1, y_pred.shape[1], 1, 1)
                y_pred[maskbool] = y_idx[maskbool]
                x_rec = AT_nomask(y_pred, mps_idx)

                if proposals_final is None and not args.use_proposed_bbox:
                    if perturb_gt_boxes_yes:
                        gt_boxes = perturb_gt_boxes(gt_boxes)
                    else:
                        gt_boxes = gt_boxes[0][:, 1:]
                    gt_boxes_xyxy = ops.xywh2xyxy(gt_boxes)
                    proposals_final = Instances(image_size=(320, 320),
                                                proposal_boxes=Boxes(gt_boxes_xyxy.cuda()),
                                                objectness_logits=torch.ones(len(gt_boxes_xyxy)).cuda(),
                                                # pred_boxes=Boxes(gt_boxes_xyxy.cuda()))
                                                )

                x_opt = x_rec.abs().flip(-2)
                x_opt = torch.clamp(x_opt, 0, 1)
                x_opt = x_opt * 255
                x_opt = x_opt.repeat(1, 3, 1, 1)
                x_opt_orig = x_opt.clone().cpu().detach()

                sizes = [640]
                x_opt = [F.interpolate(x_opt, size=(s, s), mode='bilinear')[0] for s in sizes]
                if mirroring:
                    x_opt_mirrored = [torch.flip(x, [-1]) for x in x_opt]
                    x_opt = x_opt + x_opt_mirrored
                x_opt = [torch.clamp(x, 0, 255) for x in x_opt]

                inputs = [{"image": x_opt[kj], "height": 320, "width": 320} for kj in range(len(x_opt))]
                all_box_features = []
                all_features = []
                for input_num, input in enumerate(inputs):
                    images = detection_model.preprocess_image([input])
                    images_input = images.tensor
                    features = detection_model.backbone.net(images_input)
                    feature_list = features['last_feat']
                    features = detection_model.backbone(images_input)

                    all_features.append(feature_list)
                    if proposals_final is None:
                        proposals, _ = detection_model.proposal_generator(images, features, None)
                        new_proposals = Instances(image_size=(320, 320))
                        proposal_boxes = proposals[0]._fields['proposal_boxes'].tensor[:750]
                        keep = batched_nms(proposal_boxes,
                                           torch.ones(len(proposal_boxes), device='cuda') / 2,
                                           torch.ones(len(proposal_boxes)), 0.05)

                        new_boxes = proposals[0]._fields['proposal_boxes'][keep]
                        new_proposals.set('pred_boxes', Boxes(new_boxes.tensor/sizes[0]))
                        new_proposals.set('proposal_boxes', Boxes(new_boxes.tensor/sizes[0]))

                        proposals_final = new_proposals

                    # Create a scaled copy of proposals_final
                    proposals_this_size = Instances(proposals_final.image_size)
                    for field_name, field_value in proposals_final.get_fields().items():
                        proposals_this_size.set(field_name, field_value.clone() if hasattr(field_value,
                                                                                           "clone") else field_value)
                    proposals_this_size.proposal_boxes = Boxes(
                        proposals_final.proposal_boxes.tensor * sizes[0])

                    features_ = [features[f] for f in detection_model.roi_heads.box_in_features]
                    box_features = detection_model.roi_heads.box_pooler(features_,
                                                                        [x.proposal_boxes for x in [
                                                                            proposals_this_size]])  # we can put [:10] at x.proposal_boxes if we want
                    all_box_features.append(box_features)
                # get grad of features:
                if num_repeats >= num_attacks:
                    distance = 0
                    for feature_it in range(len(box_features_final)):
                        if feature_it == eps_turn+1:
                            continue
                        for box_element in range(len(box_features)):
                            # L2 distance
                            distance += torch.norm(
                                box_features_final[feature_it][box_element].unsqueeze(
                                    0).cuda() -
                                box_features[box_element].flatten(0).unsqueeze(0).cuda(), p=2
                            )

                    distance = -distance
                    distance.backward()
                    print(distance.item())
                    all_distances.append(distance.item())
                    optimizer.step()
                    optimizer.zero_grad()
                    with torch.no_grad():
                        perturbation = eps_adv.data
                        perturbation_flat = perturbation.view(*perturbation.shape[:-2], -1)
                        perturbation_flat = perturbation_flat.renorm(p=2, dim=0, maxnorm=epsnorm_pert)
                        perturbation = perturbation_flat.view_as(perturbation)
                        eps_adv.data = perturbation
                    # prepare for next optmization step
                    optimizer.zero_grad()
                    all_features_final[eps_turn+1] = [x.detach().clone() for x in all_features]
                    box_features_final[eps_turn+1] = box_features.flatten(1).detach().clone()

                    num_repeats += 1
                    if num_repeats > optimization_steps*num_attacks - num_attacks:
                        reconstructions.append(x_rec.cpu().detach())  # TODO: check if this is correct
                        proposals, _ = detection_model.proposal_generator(images, features, None)
                        results, _ = detection_model.roi_heads(images, features, proposals, None)
                        postprocessed_results = detection_model._postprocess(results, [inputs[0]],
                                                                             images.image_sizes)

                        # filter postprocessed results for threshold > 0.5:

                        v = Visualizer(
                            np.clip(x_opt_orig[0].cpu().detach().numpy().transpose(1, 2, 0).astype(
                                np.uint8), 0, 255),
                            MetadataCatalog.get('fastmri_knee_train'), scale=1.0)
                        out = v.draw_instance_predictions(
                            postprocessed_results[0]["instances"].to("cpu"))
                        outputs_with_annotation_attacked.append(out.get_image())
                        outputs_imgs_attacked.append(out.img)
                        outputs_npy_attacked.append(clear(x_rec))  # TODO: check if correct
                        if num_repeats >= optimization_steps*num_attacks:
                            break
                        else:
                            eps_turn += 1
                            eps_turn = eps_turn % num_attacks
                            continue
                    else:
                        eps_turn += 1
                        eps_turn = eps_turn % num_attacks
                        continue

                else:
                    proposals, _ = detection_model.proposal_generator(images, features, None)
                    results, _ = detection_model.roi_heads(images, features, proposals, None)
                    postprocessed_results = detection_model._postprocess(results, [inputs[0]],
                                                                         images.image_sizes)

                    v = Visualizer(
                        x_opt_orig[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8),
                        MetadataCatalog.get('fastmri_knee_train'), scale=1.0)
                    out = v.draw_instance_predictions(postprocessed_results[0]["instances"].to("cpu"))
                    outputs_with_annotation.append(out.get_image())
                    outputs_imgs.append(out.img)
                    outputs_npy.append(clear(x_rec))  # TODO: check if correct!

                    num_repeats += 1
                    features_finals.append(features)
                    detached_features = [tensor.detach().cpu() for tensor in all_features]
                    box_features_final.append(box_features.flatten(1).cpu().detach())
                    try:
                        all_features_final.append(detached_features)
                    except TypeError:
                        assert False, "TypeError"

                    reconstructions.append(x_rec.cpu().detach())  # TODO: check if correct!
                    eps_turn += 1
                    if eps_turn == num_attacks:
                        num_attacks = num_attacks - 1
                        eps_turn = 0
                        first_time = False
                        continue
                    eps_turn = eps_turn % num_attacks
                    continue

        toc = time.time()
        print(f"Time for slice {slice_name[0]}: {toc - tic}")

        # Save results
        if args.debug:
            fig, ax = plt.subplots(nrows=3, ncols=num_attacks+2, figsize=(20, 10))
            ax[0, 0].imshow(outputs_with_annotation[0])
            ax[0, 0].set_title(f"Ground Truth \n {slice_name[0]}")
            ax[0, 1].imshow(outputs_with_annotation[1])
            ax[0, 1].set_title("Orig Prediction")
            for i in range(2, num_attacks+2):
                ax[0, i].imshow(outputs_with_annotation_attacked[i - 2])
                ax[0, i].set_title(f"Changed Prediction {i - 1}")
            # ax[0,2].imshow(outputs_with_annotation_attacked[0])
            # ax[0,2].set_title("Changed Prediction")

            ax[1, 0].imshow(outputs_imgs[0])
            ax[1, 1].imshow(outputs_imgs[1])
            for i in range(2, num_attacks+2):
                ax[1, i].imshow(outputs_imgs_attacked[i - 2])
            # ax[1,2].imshow(outputs_imgs_attacked[0])

            # set all axis off:
            for ax_ in ax.flatten():
                ax_.axis('off')

            # plot diffs:
            ax[2, 0].imshow(undersampled_img)
            ax[2, 0].set_title("Undersampled")
            diff1 = np.float32(outputs_imgs[0] / 255) - np.float32(outputs_imgs[1] / 255)
            ax[2, 1].imshow(diff1[:, :, 0], vmin=-0., cmap='hot')
            # diff2 = np.abs(np.float32(outputs_imgs[1]/255) - np.float32(outputs_imgs[2]/255))
            # ax[2,2].imshow(diff2[:,:,0], vmin=-0., cmap='hot')
            for i in range(2, num_attacks+2):
                diff = np.float32(outputs_imgs[1] / 255) - np.float32(outputs_imgs_attacked[i - 2] / 255)
                ax[2, i].imshow(diff[:, :, 0], vmin=-0., cmap='hot')
            plt.tight_layout()
            plt.show()
        else:
            # Save results
            for i_attacked, img in enumerate(outputs_imgs_attacked):
                im = Image.fromarray(img)
                im.save(f"{save_root}/images/{slice_name[0]}_{i_attacked + 1:02d}.png")
                np.save(f"{save_root}/recon/{slice_name[0]}_{i_attacked + 1:02d}.npy", outputs_npy_attacked[i_attacked])
            im = Image.fromarray(outputs_imgs[1])
            im.save(f"{save_root}/images/{slice_name[0]}_00.png")
            np.save(f"{save_root}/recon/{slice_name[0]}_00.npy", outputs_npy[0])

            # save undersampled image:
            im = Image.fromarray(undersampled_img.numpy())
            im.save(f"{save_root}/input_png/{slice_name[0]}.png")

            # save label:
            im = Image.fromarray(outputs_imgs[0])
            im.save(f"{save_root}/label_png/{slice_name[0]}.png")
            np.save(f"{save_root}/label/{slice_name[0]}.npy", label)

            np.save(f"{save_root}/mask/{slice_name[0]}.npy", clear(mask.bool()))#

            print(f"Slice {slice_name[0]} done")

        # free cuda memory:
        torch.cuda.empty_cache()
        del outputs_with_annotation, outputs_with_annotation_attacked, outputs_imgs, outputs_imgs_attacked, outputs_npy_attacked, outputs_npy
    print('ok')
