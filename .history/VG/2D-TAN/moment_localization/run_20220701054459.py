from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import eval
from eval import nms

from lib.datasets.dataset import MomentLocalizationDataset
from lib.core.config import cfg, update_config
from lib.core.utils import AverageMeter, create_logger
import lib.models as models
import lib.models.loss as loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--seed', help='seed', default=0, type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--mode', default='train', help='run test epoch only')
    parser.add_argument('--split', help='test split', type=str)
    parser.add_argument('--no_save', default=True, action="store_true", help='don\'t save checkpoint')
    parser.add_argument('--debug', default=False, type=bool)
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers is not None:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATASET.DATA_DIR = os.path.join(args.dataDir, config.DATASET.DATA_DIR)
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.tag:
        config.TAG = args.tag
    if args.debug:
        config.DEBUG = args.debug


def collate_fn(batch):
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]
    batch_vis_mask = [b['vis_mask'] for b in batch]
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_video_ids = [b['video_id'] for b in batch]
    batch_video_features = [b['video_features'] for b in batch]
    batch_descriptions = [b['description'] for b in batch]

    batch_data = {
        'batch_video_ids': batch_video_ids,
        'batch_anno_idxs': batch_anno_idxs,
        'batch_descriptions': batch_descriptions,
        'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        'batch_map_gt': [nn.utils.rnn.pad_sequence(map_gt, batch_first=True).float()[:, None] for map_gt in
                         zip(*batch_map_gt)],
        'batch_video_features': nn.utils.rnn.pad_sequence(batch_video_features, batch_first=True).float().transpose(1,
                                                                                                                    2),
        'batch_vis_mask': nn.utils.rnn.pad_sequence(batch_vis_mask, batch_first=True).float().transpose(1, 2),
    }

    if cfg.DATASET.SLIDING_WINDOW:
        batch_pos_emb = [b['pos_emb'] for b in batch]
        batch_data.update({
            'batch_pos_emb': [nn.utils.rnn.pad_sequence(pos_emb, batch_first=True).float().permute(0, 3, 1, 2) for
                              pos_emb in zip(*batch_pos_emb)]
        })
    else:
        batch_data.update({
            'batch_duration': [b['duration'] for b in batch]
        })

    return batch_data


def recover_to_single_map(joint_probs):
    batch_size, _, map_size, _ = joint_probs[0].shape
    score_map = torch.zeros(batch_size, 1, map_size, map_size).cuda()
    for prob in joint_probs:
        scale_num_clips, scale_num_anchors = prob.shape[2:]
        dilation = map_size // scale_num_clips
        for i in range(scale_num_anchors):
            score_map[..., :map_size // dilation * dilation:dilation, (i + 1) * dilation - 1] = torch.max(
                score_map[..., :map_size // dilation * dilation:dilation, (i + 1) * dilation - 1].clone(), prob[..., i])
    return score_map


def upsample_to_single_map(joint_probs):
    batch_size, _, map_size, _ = joint_probs[0].shape
    score_map = torch.zeros(batch_size, 1, map_size, map_size).cuda()
    for i, prob in enumerate(joint_probs):
        dilation = 2 ** (i)
        num_clips, num_anchors = prob.shape[-2:]
        score_map[..., :dilation * num_clips, :dilation * num_anchors] = torch.max(
            F.interpolate(prob, scale_factor=dilation, mode='bilinear', align_corners=True),
            score_map[..., :dilation * num_clips, :dilation * num_anchors]
        )
    return score_map


def network(sample, model, optimizer=None, return_map=False):
    import pdb
    pdb.set_trace()
    textual_input = sample['batch_word_vectors']
    textual_mask = sample['batch_txt_mask']
    visual_mask = sample['batch_vis_mask']
    visual_input = sample['batch_video_features']
    map_gts = sample['batch_map_gt']

    predictions, map_masks = model(textual_input, textual_mask, visual_input, visual_mask)

    loss_value = 0
    for prediction, map_mask, map_gt in zip(predictions, map_masks, map_gts):
        scale_loss = getattr(loss, cfg.LOSS.NAME)(prediction, map_mask, map_gt.cuda(), cfg.LOSS.PARAMS)
        loss_value += scale_loss
    joint_prob = recover_to_single_map(predictions)
    mask = recover_to_single_map(map_masks)

    if torch.sum(mask[0] > 0).item() == 0:
        print(sample['batch_anno_idxs'])
    assert torch.sum(mask[0] > 0).item() > 0

    if cfg.DATASET.SLIDING_WINDOW:
        time_unit = cfg.DATASET.TIME_UNIT * cfg.DATASET.INPUT_NUM_CLIPS / cfg.DATASET.OUTPUT_NUM_CLIPS[0]
        sorted_times = get_sw_proposal_results(joint_prob.detach().cpu(), mask, time_unit)
    else:
        sorted_times = get_proposal_results(joint_prob.detach().cpu(), mask, sample['batch_duration'])

    if model.training:
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    # if cfg.DEBUG:
    #     print('visual_input.shape: {}'.format(visual_input.shape))
    #     print('textual_input.shape: {}'.format(textual_input.shape))
    #     return

    if return_map:
        return loss_value, sorted_times, joint_prob.detach().cpu()
    else:
        return loss_value, sorted_times


def get_proposal_results(scores, mask, durations):
    # assume all valid scores are larger than one
    out_sorted_times = []
    batch_size, _, num_clips, num_anchors = scores.shape
    scores, indexes = torch.topk(scores.view(batch_size, -1), torch.sum(mask[0] > 0).item(), dim=1)
    t_starts = (indexes // num_anchors).float() / num_clips * torch.tensor(durations).view(batch_size, 1)
    t_ends = t_starts + (indexes % num_anchors + 1).float() / num_clips * torch.tensor(durations).view(batch_size, 1)

    for t_start, t_end in zip(t_starts, t_ends):
        t_start, t_end = t_start[t_start < t_end], t_end[t_start < t_end]
        dets = nms(torch.stack([t_start, t_end], dim=1).tolist(), thresh=cfg.TEST.NMS_THRESH,
                   top_k=max(cfg.TEST.RECALL))
        out_sorted_times.append(dets)
    return out_sorted_times


def get_sw_proposal_results(scores, mask, time_unit):
    # assume all valid scores are larger than one
    out_sorted_times = []
    batch_size, _, num_clips, num_anchors = scores.shape
    scores, indexes = torch.topk(scores.view(batch_size, -1), torch.sum(mask[0] > 0).item(), dim=1)
    t_starts = (indexes // num_anchors).float() * time_unit
    t_ends = t_starts + (indexes % num_anchors + 1).float() * time_unit

    for t_start, t_end in zip(t_starts, t_ends):
        t_start, t_end = t_start[t_start < t_end], t_end[t_start < t_end]
        dets = nms(torch.stack([t_start, t_end], dim=1).tolist(), thresh=cfg.TEST.NMS_THRESH,
                   top_k=max(cfg.TEST.RECALL))
        out_sorted_times.append(dets)
    return out_sorted_times


def train_epoch(train_loader, model, optimizer, verbose=False):
    model.train()

    loss_meter = AverageMeter()
    sorted_segments_dict = {}
    if verbose:
        pbar = tqdm(total=len(train_loader), dynamic_ncols=True)

    for cur_iter, sample in enumerate(train_loader):
        loss_value, sorted_times = network(sample, model, optimizer)
        loss_meter.update(loss_value.item(), 1)
        sorted_segments_dict.update({idx: timestamp for idx, timestamp in zip(sample['batch_anno_idxs'], sorted_times)})
        if verbose:
            pbar.update(1)
        if args.debug:
            return

    if verbose:
        pbar.close()

    annotations = train_loader.dataset.annotations
    annotations = [annotations[key] for key in sorted(sorted_segments_dict.keys())]
    sorted_segments = [sorted_segments_dict[key] for key in sorted(sorted_segments_dict.keys())]
    result = eval.evaluate(sorted_segments, annotations)

    return loss_meter.avg, result


@torch.no_grad()
def test_epoch(test_loader, model, verbose=False, save_results=False):
    model.eval()

    loss_meter = AverageMeter()
    sorted_segments_dict = {}
    saved_dict = {}

    if verbose:
        pbar = tqdm(total=len(test_loader), dynamic_ncols=True)

    for cur_iter, sample in enumerate(test_loader):
        loss_value, sorted_times, score_maps = network(sample, model, return_map=True)
        loss_meter.update(loss_value.item(), 1)
        sorted_segments_dict.update({idx: timestamp for idx, timestamp in zip(sample['batch_anno_idxs'], sorted_times)})
        saved_dict.update({idx: {'vid': vid, 'timestamps': timestamp, 'description': description}
                           for idx, vid, timestamp, description in zip(sample['batch_anno_idxs'],
                                                                       sample['batch_video_ids'],
                                                                       sorted_times,
                                                                       sample['batch_descriptions'])})
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
    annotations = test_loader.dataset.annotations
    annotations = [annotations[key] for key in sorted(sorted_segments_dict.keys())]
    sorted_segments = [sorted_segments_dict[key] for key in sorted(sorted_segments_dict.keys())]
    saved_dict = [saved_dict[key] for key in sorted(saved_dict.keys())]
    if save_results:
        if not os.path.exists('results/{}'.format(cfg.DATASET.NAME)):
            os.makedirs('results/{}'.format(cfg.DATASET.NAME))
        torch.save(saved_dict,
                   'results/{}/{}-{}.pkl'.format(cfg.DATASET.NAME, os.path.basename(args.cfg).split('.yaml')[0],
                                                 test_loader.dataset.split))
    result = eval.evaluate(sorted_segments, annotations)
    return loss_meter.avg, result


def train(cfg, verbose):
    logger, final_output_dir, tensorboard_dir = create_logger(cfg, args.cfg, cfg.TAG)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + pprint.pformat(cfg))

    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    init_epoch = 0
    model = getattr(models, cfg.MODEL.NAME)(cfg.MODEL)
    if cfg.MODEL.CHECKPOINT and cfg.TRAIN.CONTINUE:
        init_epoch = int(os.path.basename(cfg.MODEL.CHECKPOINT)[5:9]) + 1
        model_checkpoint = torch.load(cfg.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
        print(f"loading checkpoint: {cfg.MODEL.CHECKPOINT}")
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # print FLOPs and Parameters
    # if True:
    #     video_feature = torch.zeros([1, 4096, 384], device=device)
    #     video_mask = torch.zeros([1, 1, 384], device=device)
    #     text_feature = torch.zeros([1, 25, 300], device=device)
    #     text_mask = torch.zeros([1, 25, 1], device=device)
    #
    #     count_dict, *_ = flop_count(model, (text_feature, text_mask, video_feature, video_mask))
    #     count = sum(count_dict.values())
    #     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    #     logger.info(flop_count_str(FlopCountAnalysis(model, (text_feature, text_mask, video_feature, video_mask))))
    #     logger.info('{:<30}  {:.1f} GFlops'.format('number of FLOPs: ', count))
    #     logger.info('{:<30}  {:.1f} MB'.format('number of params: ', n_parameters / 1000 ** 2))

    if cfg.OPTIM.NAME == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)
    elif cfg.OPTIM.NAME == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)
    elif cfg.OPTIM.NAME == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)
    elif cfg.OPTIM.NAME == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)
    else:
        raise NotImplementedError

    train_dataset = MomentLocalizationDataset(cfg.DATASET, 'train')
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=cfg.TRAIN.SHUFFLE,
                              num_workers=cfg.WORKERS,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)

    if not cfg.DATASET.NO_VAL:
        val_dataset = MomentLocalizationDataset(cfg.DATASET, 'val')
        val_loader = DataLoader(val_dataset,
                                batch_size=cfg.TEST.BATCH_SIZE,
                                shuffle=False,
                                num_workers=cfg.WORKERS,
                                pin_memory=True,
                                collate_fn=collate_fn)

    test_dataset = MomentLocalizationDataset(cfg.DATASET, 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.TEST.BATCH_SIZE,
                             shuffle=False,
                             num_workers=cfg.WORKERS,
                             pin_memory=True,
                             collate_fn=collate_fn)

    # [[score1, score2], [epoch1, epoch2]]
    max_metric = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    writer = SummaryWriter(tensorboard_dir)

    for cur_epoch in range(init_epoch, cfg.TRAIN.MAX_EPOCH):
        train_avg_loss, train_result = train_epoch(train_loader, model, optimizer, verbose)

        loss_message = '\n' + 'epoch: {}; train loss {:.4f};'.format(cur_epoch, train_avg_loss)
        table_message = '\n' + eval.display_results(train_result, 'performance on training set')

        if not cfg.DATASET.NO_VAL:
            val_avg_loss, val_result = test_epoch(val_loader, model, verbose)
            loss_message += ' val loss: {:.4f};'.format(val_avg_loss)
            table_message += '\n' + eval.display_results(val_result, 'performance on validation set')
            writer.add_scalar('val_avg_loss', val_avg_loss, cur_epoch)

        # test_result: {'ranks': [[R1@0.1, R5@0.1], [R1@0.3, R5@0.3], [R1@0.5, R5@0.5], [R1@0.7, R5@0.7]], 'mIoU': [mean IoU]}
        test_avg_loss, test_result = test_epoch(test_loader, model, verbose)
        loss_message += ' test loss: {:.4f}'.format(test_avg_loss)
        table_message += '\n' + eval.display_results(test_result, 'performance on testing set')

        message = loss_message + table_message
        logger.info(message)

        # save max metrics and tensorboard
        if True:
            tious = cfg.TEST.TIOU
            recalls = cfg.TEST.RECALL

            for i in range(len(tious)):
                for j in range(len(recalls)):
                    if test_result['ranks'][i, j] > max_metric[i][j][0]:
                        max_metric[i][j][0], max_metric[i][j][1] = test_result['ranks'][i, j], cur_epoch

            test_max_result = eval.display_max_results(max_metric, 'max score and epoch')
            logger.info(test_max_result)

            writer.add_scalar('train_avg_loss', train_avg_loss, cur_epoch)
            writer.add_scalar('test_avg_loss', test_avg_loss, cur_epoch)
            writer.add_scalar('mIoU', test_result['mIoU'], cur_epoch)

            for i in range(len(tious)):
                for j in range(len(recalls)):
                    writer.add_scalar(f'R{recalls[j]}@{tious[i]}', test_result['ranks'][i, j], cur_epoch)

        # save model
        if not args.no_save:
            # test_result['ranks'][0] == [R1@0.1, R5@0.1]
            saved_model_filename = os.path.join(cfg.MODEL_DIR, '{}/{}/epoch{:04d}-{:.4f}-{:.4f}.pkl'.format(
                cfg.DATASET.NAME, os.path.basename(args.cfg).split('.yaml')[0],
                cur_epoch, test_result['ranks'][0, 0], test_result['ranks'][0, 1]))

            # os.path.dirname(path), 去掉文件名，返回目录
            root_folder1 = os.path.dirname(saved_model_filename)
            root_folder2 = os.path.dirname(root_folder1)
            root_folder3 = os.path.dirname(root_folder2)
            if not os.path.exists(root_folder3):
                print('Make directory %s ...' % root_folder3)
                os.mkdir(root_folder3)
            if not os.path.exists(root_folder2):
                print('Make directory %s ...' % root_folder2)
                os.mkdir(root_folder2)
            if not os.path.exists(root_folder1):
                print('Make directory %s ...' % root_folder1)
                os.mkdir(root_folder1)

            torch.save(model.module.state_dict(), saved_model_filename)


def test(cfg, split):
    # cudnn related setting, ignore it
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    model = getattr(models, cfg.MODEL.NAME)(cfg.MODEL)

    if os.path.exists(cfg.MODEL.CHECKPOINT):
        model_checkpoint = torch.load(cfg.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    else:
        raise ("checkpoint not exists")

    model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = MomentLocalizationDataset(cfg.DATASET, split)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.WORKERS,
                            pin_memory=True,
                            collate_fn=collate_fn)
    avg_loss, result = test_epoch(dataloader, model, True, save_results=True)

    print(' val loss {:.4f}'.format(avg_loss))
    print(eval.display_results(result, 'performance on {} set'.format(split)))


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    reset_config(cfg, args)
    if args.mode == 'train':
        train(cfg, args.verbose)
    else:
        test(cfg, args.split)
