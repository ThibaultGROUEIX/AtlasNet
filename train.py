import sys
import auxiliary.argument_parser as argument_parser
import auxiliary.my_utils as my_utils
import time
import torch
from auxiliary.my_utils import yellow_print

"""
Main training script.
author : Thibault Groueix 01.11.2019
"""

opt = argument_parser.parser()
torch.cuda.set_device(opt.multi_gpu[0])
my_utils.plant_seeds(random_seed=opt.random_seed)
import training.trainer as trainer

trainer = trainer.Trainer(opt)
trainer.build_dataset()
trainer.build_network()
trainer.build_optimizer()
trainer.build_losses()
trainer.start_train_time = time.time()

if opt.demo:
    with torch.no_grad():
        trainer.demo(opt.demo_input_path)
    sys.exit(0)

if opt.run_single_eval:
    with torch.no_grad():
        trainer.test_epoch()
    sys.exit(0)

for epoch in range(trainer.epoch, opt.nepoch):
    trainer.train_epoch()
    with torch.no_grad():
        trainer.test_epoch()
    trainer.dump_stats()
    trainer.increment_epoch()
    trainer.save_network()

yellow_print(f"Visdom url http://localhost:{trainer.opt.visdom_port}/")
yellow_print(f"Netvision report url http://localhost:{trainer.opt.http_port}/{trainer.opt.dir_name}/index.html")
yellow_print(f"Training time {(time.time() - trainer.start_time)//60} minutes.")
