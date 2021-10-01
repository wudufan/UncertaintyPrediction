# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# generate the scripts to process the datasets


# %%
import glob
import os
import copy
import pathlib
from datetime import datetime


# %%
job_name = 'fbp_mayo'
devices = ['0', '1', '2', '3']
nprocesses = 4
args = {'input_dir': '/home/dwu/data/lowdoseCTsets/', 
        'geometry': '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg',
        'N0': '1e5',
        'imgNorm': '0.019'
        }
output_dir = '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/'
slurm_header = """#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=%s
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0
"""%job_name


# %%
# identify the input directories
names = ['L067_full_sino', 'L096_full_sino', 'L109_full_sino', 'L143_full_sino', 'L192_full_sino', 
         'L286_full_sino', 'L291_full_sino', 'L310_full_sino', 'L333_full_sino', 'L506_full_sino']
# dose_rates = range(1,17,1)
# dose_rates = [1,2,4,8,12,16]
dose_rates = [6]
cmds = []
for name in names:
    for dose_rate in dose_rates:
        cmd = copy.deepcopy(args)
        cmd['output_dir'] = os.path.join(output_dir, 'dose_rate_%d'%dose_rate)
        cmd['name'] = name
        cmd['dose_rate'] = dose_rate
        cmd['device'] = devices[len(cmds) % len(devices)]
        cmds.append(cmd)

# %%
# generate scripts
logfile = '%s_%s'%(job_name, datetime.now().strftime('%Y%m%d'))
with open('%s.sh'%job_name, 'w') as f:
    # slurm
    f.write(slurm_header + '\n\n')

    for k, cmd in enumerate(cmds):
        argstr = ' '.join(['--%s "%s"'%(name, cmd[name]) for name in cmd])
        logstr = '&>> %s_%d.log'%(logfile, k%nprocesses)
        f.write('echo "%d/%d" %s\n'%(k+1, len(cmds), logstr))
        f.write('python3 %s.py '%job_name + argstr + ' ' + logstr + ' &\n')
        if (k+1)%nprocesses == 0:
            f.write('wait\n')
            f.write('echo "%d/%d"\n'%(k+1, len(cmds)))
    f.write('wait\n')
    # cat logs together
    f.write('cat ' + ' '.join(['%s_%d.log'%(logfile, k) for k in range(nprocesses)]) + ' > ' + logfile+'.log\n')
    for k in range(nprocesses):
        f.write('rm %s_%d.log\n'%(logfile, k))


# %%



