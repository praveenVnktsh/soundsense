import os
import glob

os.makedirs('../data/processed', exist_ok=True)

for path in glob.glob('../data/videos/*'):
    filename = os.path.basename(path)
    basename = filename.split('.')[0]
    print(basename)
    os.system(f'python estimate_pf.py {filename} ../data/processed/{basename}.json')
    os.system(f'python estimate_gripper.py {filename} ../data/processed/{basename}.json')