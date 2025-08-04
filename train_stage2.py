from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from solver.make_optimizer_prompt import make_optimizer_2stage
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage2 import do_train_stage2
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/cnn_clipreid.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.STAGE2_OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pth_directory = '/media/hasil/Etc_HDD/Re-id/CLIP-ReID_ver3_2_2/result'
    pth_files = [f for f in os.listdir(pth_directory) if f.endswith('.pth')]
    pth_files.sort()
    for pth_file in pth_files:

        model_output_dir = os.path.join(output_dir, os.path.splitext(pth_file)[0])
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        file_name = 'train_log_' + pth_file.split('_')[-1].split('.')[0] + '.txt'
        logger = setup_logger("transreid", model_output_dir, file_name, if_train=True)
        logger.info("Saving model in the path :{}".format(model_output_dir))
        logger.info(args)

        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))
            with open(args.config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        if cfg.MODEL.DIST_TRAIN:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

        train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

        loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

        model.load_param(os.path.join(cfg.STAGE1_OUTPUT_DIR, pth_file))
        optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
        scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA, cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                      cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

        do_train_stage2(
            cfg,
            model_output_dir,
            model,
            center_criterion,
            train_loader_stage2,
            val_loader,
            optimizer_2stage,
            optimizer_center_2stage,
            scheduler_2stage,
            loss_func,
            num_query, args.local_rank
        )