import os
import numpy as np
import torch
import torchvision
import socket

def detuple(x):
    if isinstance(x, tuple):
        x = x[0]
    return x

class LogWriter():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def log_args_and_backup_code(args, file_path):
    file_name = os.path.basename(file_path)
    logfile = LogWriter(os.path.join(args.output_dir, f"args_{file_name.split('.')[0]}.txt"))
    for k, v in args.__dict__.items():
        logfile.cprint(f"{k} : {v}")
    logfile.cprint("Numpy: {}".format(np.__version__))
    logfile.cprint("Pytorch: {}".format(torch.__version__))
    logfile.cprint("torchvision: {}".format(torchvision.__version__))
    logfile.cprint("Cuda: {}".format(torch.version.cuda))
    logfile.cprint("hostname: {}".format(socket.gethostname()))
    logfile.cprint("="*30) # separator
    logfile.close()

    os.system(f'cp {file_path} {args.output_dir}/{file_name}.backup')