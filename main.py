# global imports 
import os
import glob
import json
import matplotlib.pyplot as plt # debug
import argparse

# custom imports
from src.volumeregistration2dbased import VolumeRegistration2Dbased as vol2d


def main():
    ### CONFIG AND ARG-PARSING ###
    parser = argparse.ArgumentParser(description='OCT Whole Volume Registration (OVolReg).')
    parser.add_argument("-n", "--name", type=str, help="Experiment name.", default='test')
    parser.add_argument("-c", "--config", default='config.json', help="Path to config file with default experiment parameters.")
    
    # get the arguments
    args = parser.parse_args() # access via args.argument
    args_dict = vars(args) # access via args_dict["argument"]

    # load config
    with open(args.config, 'r') as f:
        parameters = json.load(f)
    # parameters['parsed_args'] = args_dict
    # with open(os.path.join(experiment_path, 'config.json'), 'w+') as f:
    #     json.dump(parameters, f, sort_keys=True, indent=0)
    
    # volume_dirs = glob.glob(parameters["path_loading"] + "/*")
    # volume_dirs = (
    #     r"C:\Users\PhilippsLabLaptop\Desktop\Data_volumeAveraging\recon_rasterVol01_4000x512x511.bin",
    #     r"C:\Users\PhilippsLabLaptop\Desktop\Data_volumeAveraging\recon_rasterVol02_4000x512x511.bin",
    #     r"C:\Users\PhilippsLabLaptop\Desktop\Data_volumeAveraging\recon_rasterVol03_4000x512x511.bin" 
    #     )

    VolReg = vol2d(parameters)
    v = VolReg._rigid_registration(method="crosscorrelation")
    VolReg._save_volume_to_disk(v)
    
    # # create experiment 
    # experiment_path = os.path.join(os.getcwd(), 'experiments', args.name)
    # try:
    #     os.mkdir(experiment_path)
    # except FileExistsError:
    #     # TODO just pass if you don't want this safety prompt
    #     if args.name != "test":
    #         print(pcolors.RED + '\n"{}" already exists! Continuing will overwrite previous results!'.format(args.name), end='\t')
    #         decision = str.lower(input("Continue anyway? [y/N]  "  + pcolors.END)) or "n"
    #         print('\n')
    #         if decision != 'y':
    #             print('Aborting!')
    #             return -1
    
    print('\nDone with registration.\n')
    

if __name__ == '__main__':
    main()