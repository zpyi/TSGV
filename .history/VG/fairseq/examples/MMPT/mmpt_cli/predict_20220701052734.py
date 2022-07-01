# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import glob
import argparse
import pprint
import omegaconf

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from mmpt.utils import load_config, set_seed
from mmpt.evaluators import Evaluator
from mmpt.evaluators import predictor as predictor_path
from mmpt.tasks import Task
from mmpt import processors
from mmpt.datasets import MMDataset


def get_dataloader(config):
    meta_processor_cls = getattr(processors, config.dataset.meta_processor)
    video_processor_cls = getattr(processors, config.dataset.video_processor)
    text_processor_cls = getattr(processors, config.dataset.text_processor)
    aligner_cls = getattr(processors, config.dataset.aligner)

    meta_processor = meta_processor_cls(config.dataset)


    video_processor = video_processor_cls(config.dataset)
    text_processor = text_processor_cls(config.dataset)
    aligner = aligner_cls(config.dataset)

    test_data = MMDataset(
        meta_processor,
        video_processor,
        text_processor,
        aligner,
    )
    print("test_len", len(test_data)) # test_len 9790
    output = test_data[0]
    test_data.print_example(output)  # 打印一个样本

    test_dataloader = DataLoader(
        test_data,
        batch_size=config.fairseq.dataset.batch_size, # 256
        shuffle=False,
        num_workers=6,
        collate_fn=test_data.collater,
    )

    return test_dataloader


def main(args):
    config = load_config(args)
    if isinstance(config, omegaconf.dictconfig.DictConfig):
        print(OmegaConf.to_yaml(config))  # 打印config文件
    else:
        pp = pprint.PrettyPrinter(indent=4)
        pp.print(config)

    mmtask = Task.config_task(config)
    mmtask.build_model() 

    test_dataloader = get_dataloader(config) # len=39
    checkpoint_search_path = os.path.dirname(config.eval.save_path) # 'runs/retri/videoclip/tacos_zs'
    results = []

    prefix = os.path.basename(args.taskconfig)  # 'test_tacos_zs.yaml'
    if prefix.startswith("test"):
        # loop all checkpoint for datasets without validation set.
        if "best" not in config.fairseq.common_eval.path:
            print("eval each epoch.")
            for checkpoint in glob.glob(checkpoint_search_path + "/checkpoint*"):
                model = mmtask.load_checkpoint(checkpoint)
                ckpt = os.path.basename(checkpoint)
                evaluator = Evaluator(config)
                output = evaluator.evaluate(
                    model, test_dataloader, ckpt + "_merged")
                results.append((checkpoint, output))
        # use the one specified by the config lastly.
        model = mmtask.load_checkpoint(config.fairseq.common_eval.path)
        evaluator = Evaluator(config)
        output = evaluator.evaluate(model, test_dataloader)
        results.append((config.fairseq.common_eval.path, output))

        best_result = None
        best_metric = 0.
        for checkpoint, result in results:
            # runs/retri/videoclip/checkpoint_best.pt, {'R1': 0.00020833845739840246, 'R5': 0.0010416922869920123, 'R10': 0.0020833845739840246, 'MR': 3936.0}
            print(checkpoint)
            evaluator.metric.print_computed_metrics(result)
            best_score = evaluator.metric.best_metric(result)
            if best_score > best_metric:
                best_result = (checkpoint, result)
                best_metric = best_score
        print("best results:")
        print(best_result[0])
        evaluator.metric.print_computed_metrics(best_result[1])

    elif prefix.startswith("vis"):
        model = mmtask.load_checkpoint(config.fairseq.common_eval.path)
        predictor_cls = getattr(predictor_path, config.predictor)
        predictor = predictor_cls(config)
        predictor.predict_loop(model, test_dataloader, mmtask, None)
    else:
        raise ValueError("unknown prefix of the config file", args.taskconfig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("taskconfig", type=str)
    args = parser.parse_args()
    main(args)
