import auxiliary.argument_parser as argument_parser
import auxiliary.my_utils as my_utils
import time
import os
import torch

opt = argument_parser.parser()
my_utils.plant_seeds(randomized_seed=opt.randomize)

import training.trainer as trainer

trainer = trainer.Trainer(opt)
trainer.build_dataset()
trainer.build_network()
trainer.build_optimizer()
trainer.build_losses()
trainer.start_train_time = time.time()

if opt.run_single_eval:
    with torch.no_grad():
        trainer.test_epoch()
    os.exit()

for epoch in range(trainer.epoch, opt.nepoch):
    trainer.train_epoch()
    with torch.no_grad():
        trainer.test_epoch()
    trainer.dump_stats()
    trainer.increment_epoch()

trainer.save_network()
print("end script!")
