from configparser import ConfigParser
import argparse
#from dbm.kld_test import *
from dbm.gan_rec2 import *

def main():

    #torch device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU will be used!!!")
    else:
        device = torch.device("cpu")
        print("CPU will be used!")
    torch.set_num_threads(12)

    # ## Read config

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    config_file = args.config

    cfg = ConfigParser(inline_comment_prefixes="#")
    cfg.read(config_file)

    #set up model
    model = GAN(device=device, cfg=cfg)

    #with open('./' + name + '/config.ini', 'a') as f:
    #    cfg.write(f)

    #train
    model.val(dir='val')


if __name__ == "__main__":
    main()
