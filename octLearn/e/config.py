import configparser
import os

import octLearn

def reset():
    config_dir = get_config_dir()
    config_file = os.path.join(config_dir, 'default.ini')

    os.truncate(config_file, 0)

def get_config_dir():
    working_dir = os.path.abspath(os.path.curdir)
    project_dir = os.path.dirname(os.path.abspath(octLearn.__file__))

    if working_dir.startswith(project_dir) or os.path.basename(working_dir) == 'bin':
        target_path = os.path.expanduser(os.path.join('~', '.log'))
        os.makedirs(target_path, 0x755, exist_ok=True)
        return target_path
    else:
        return working_dir


def get_config():
    config_dir = get_config_dir()
    config_file = os.path.join(config_dir, 'default.ini')
    config_obj = configparser.ConfigParser()
    project_dir = os.path.dirname(os.path.abspath(octLearn.__file__))
    if os.path.exists(config_file):
        config_obj.read(config_file)

    if not config_obj.has_section('main'):
        config_obj.add_section('main')
    config_obj['main']['working_directory'] = config_dir
    config_obj['main']['project_directory'] = project_dir
    
    return config_obj


def set_config(*, config_obj):
    config_dir = get_config_dir()
    config_file = os.path.join(config_dir, 'default.ini')
    with open(config_file, 'w') as file:
        config_obj.write(file)


def update_config(dict_config):
    dict_dup = dict()
    for k, v in dict_config.items():
        t = type(v)
        if t in [int, float, str, bool]:
            dict_dup[k] = v
        elif isinstance(v, type):
            dict_dup[k] = v.__name__
        elif v is None:
            continue
        else:
            print('[Warning] config value ({}: {}) not being serialized'.format(k, v))

    config_obj = get_config()
    config_obj.read_dict({'misc': dict_dup}, 'User provided')
    set_config(config_obj=config_obj)
