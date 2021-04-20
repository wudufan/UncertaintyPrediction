import argparse
import configparser

def build_argparser_from_config(parser, config):
    for sec in config:
        for k in config[sec]:
            arg_name = sec + '.' + k
            default_val = config[sec][k]
            
            parser.add_argument('--' + arg_name, default = default_val)
    
    return parser

def update_config_from_args(config, args):
    for k in vars(args):
        tokens = k.split('.')
        if len(tokens) == 2:
            config[tokens[0]][tokens[1]] = getattr(args, k)
    
    return config
            
def get_kwargs(config, additional_type_dict = None, default_type = str):
    '''
    Get kwargs from the configuration. Convert the strings to corresponding types

    The types are specificed by the type_dict. 
    For each section in the config, <section name> gives the default data type in the config. <section name>.<key name> overrides the data type for a specific key. 
    The parameter additional_type_dict will give additional type mapping o override exsiting type mapping.

    If a paremeter has ',' in the value, it will be interpreted as a list. Otherwise is will be interpreted as a single vlue
    '''
    type_dict = {
        'IO': str,
        'Train': int,
        'Train.lr': float,
        'Data': int,
        'Data.zoom': float,
        'Data.offset': float,
        'Data.noise_std': float,
        'Data.noise_prob': float,
        'Loss': float,
        'Loss.name': str,
        'Loss.nhard': int,
        'Loss.gamma': float,
        'Loss.alpha': float,
        'Network': int,
        'Network.conv_drop_rates': float,
        'Network.se_ratios': float,
        'Network.drop_rate': float,
        'Network.negative_slope': float,
        'Network.attention': str,
    }
    if additional_type_dict is not None:
        type_dict.update(additional_type_dict)

    arg_dict = {}
    for sec in config:
        # get default type for the section
        if sec in type_dict:
            sec_type = type_dict[sec]
        else:
            sec_type = default_type
        arg_dict[sec] = {}
        for k in config[sec]:
            # determine the type of the property
            name = sec + '.' + k
            if name in type_dict:
                k_type = type_dict[name]
            else:
                k_type = sec_type
            
            val = config[sec][k].strip(' ')
            if len(val) == 0:
                val = None
            elif ',' in val:
                # check if it is list
                val = [k_type(v) for v in val.split(',') if len(v) > 0]
            else:
                val = k_type(val)
            
            arg_dict[sec][k] = val
    
    return arg_dict