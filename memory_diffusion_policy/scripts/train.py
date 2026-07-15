import hydra
from eval_envs.trainer.trainer import Trainer
import os
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
@hydra.main(config_path=os.path.join(main_dir, "eval_envs/config"), config_name="train_dp_unet.yaml", version_base=None)
def main(config):
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()