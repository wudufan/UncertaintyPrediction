'''
Helper function for scripts generation.
'''

# %%
from datetime import datetime


# %%
def multi_thread_script(
    sh_name,
    script_name,
    log_name,
    cmd_args,
    nprocess=1,
    exe_name='python3',
    time_stamp='date',
    sh_header='#!/bin/bash'
):
    if time_stamp == 'date':
        time_stamp = datetime.strftime(datetime.now(), '%Y%m%d')
    elif time_stamp == 'now':
        time_stamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    elif time_stamp is not None:
        time_stamp = str(time_stamp)

    if time_stamp is not None:
        log_name = log_name + '_' + time_stamp

    with open(sh_name, 'w') as f:
        f.write(sh_header + '\n\n')
        for k, args in enumerate(cmd_args):
            if k % nprocess == 0:
                if k > 0 and nprocess > 1:
                    f.write('wait\n')
                f.write('echo "{k}/{total}"\n'.format(k=k, total=len(cmd_args)))

            arg_str = []
            for name in args:
                if isinstance(args[name], list):
                    val = ' '.join(['"{0}"'.format(v) for v in args[name]])
                    arg_str.append('--{0}'.format(name) + ' ' + val)
                else:
                    arg_str.append('--{0} "{1}"'.format(name, args[name]))
            arg_str = ' '.join(arg_str)

            log_str = '&>> ' + log_name + '_' + str(k % nprocess) + '.log'
            f.write(' '.join([exe_name, script_name, arg_str, log_str]))
            if nprocess > 1:
                f.write(' &\n')
            else:
                f.write('\n')
        if nprocess > 1:
            f.write('wait\n')

        f.write(
            'cat '
            + ' '.join(['{log}_{k}.log'.format(log=log_name, k=k) for k in range(nprocess)])
            + ' > ' + log_name + '.log\n'
        )
        for k in range(nprocess):
            f.write('rm {log}_{k}.log\n'.format(log=log_name, k=k))
