'''
FBP mayo script
'''

# %%
import sys
import os
import copy
sys.path.append('..')
from helper.scripts_generation import multi_thread_script

# %%
job_name = 'fbp_mayo'
devices = ['2', '3']
nprocesses = len(devices)
args = {
    'input_dir': '/home/dwu/data/lowdoseCTsets/',
    'geometry': '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg',
    'N0': '1e5',
    'imgNorm': '0.019'
}
output_dir = '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/'
slurm_header = """#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name={0}
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0
""".format(job_name)

# %%
# identify the input directories
names = [
    'L067_full_sino', 'L096_full_sino', 'L109_full_sino', 'L143_full_sino', 'L192_full_sino',
    'L286_full_sino', 'L291_full_sino', 'L310_full_sino', 'L333_full_sino', 'L506_full_sino'
]
# dose_rates = range(1,17,1)
# dose_rates = [1,2,4,8,12,16]
dose_rates = [1, 4]
cmds = []
for name in names:
    for dose_rate in dose_rates:
        cmd = copy.deepcopy(args)
        cmd['output_dir'] = os.path.join(output_dir, 'dose_rate_{0}'.format(dose_rate))
        cmd['name'] = name
        cmd['dose_rate'] = dose_rate
        cmd['device'] = devices[len(cmds) % len(devices)]
        cmds.append(cmd)

# %%
multi_thread_script(
    job_name + '.sh', 'fbp_mayo.py', job_name, cmds, nprocesses, sh_header=slurm_header
)
