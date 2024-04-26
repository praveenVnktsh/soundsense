import shutil
import os
import glob



for path in os.listdir('lightning_logs/'):
    if os.path.isdir('lightning_logs/' + path):
        contains_chkpt = False
        for file in list(glob.glob('lightning_logs/' + path + '/*')):
            if '.ckpt' in file:
                contains_chkpt = True
                break
        if not contains_chkpt:
            shutil.rmtree('lightning_logs/' + path)
            print('lightning_logs/' + path)