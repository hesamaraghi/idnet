# Read remote params from a config file (YAML or JSON).
# Default config path: ./idn/config/remote.yaml (project convention).
import json
try:
    import yaml
except Exception:
    yaml = None

import socket, subprocess, shutil, os

# Default candidate locations for the remote config
DEFAULT_CONFIG_PATHS = [
    os.path.join(os.getcwd(), 'toy_datasets', 'config', 'remote.yaml'),
    os.path.join(os.getcwd(), 'toy_datasets', 'config', 'remote.yml'),
]


def load_remote_config(config_path=None, verbose=True):
    """Load remote settings from a YAML or JSON config file.
    Returns a dict (possibly empty) and prints where it was loaded from."""
    paths = [config_path] if config_path else []
    for p in DEFAULT_CONFIG_PATHS:
        if p not in paths:
            paths.append(p)
    for p in paths:
        if not p:
            continue
        p_exp = os.path.expanduser(p)
        if not os.path.exists(p_exp):
            continue
        try:
            if yaml and p_exp.lower().endswith(('.yml', '.yaml')):
                with open(p_exp, 'r') as f:
                    conf = yaml.safe_load(f) or {}
            else:
                with open(p_exp, 'r') as f:
                    conf = json.load(f)
            if verbose:
                print(f'Loaded remote config from: {p_exp}')
            return conf
        except Exception as e:
            if verbose:
                print(f'Failed to read config {p_exp}: {e}')
            continue
    if verbose:
        print('No config file found at default locations.')
    return {}


# Auto-detect whether this machine is the HPC (if hostname contains keywords).
def is_hpc_host(hpc_keywords=('hpc','cluster','login')):
    name = socket.gethostname().lower()
    return any(k in name for k in hpc_keywords)


def ensure_local_file(rel_path, remote_server=None, remote_user=None, remote_base=None, config_path=None, verbose=True):
    """Ensure rel_path exists locally. If missing and we are NOT on the HPC host, attempt to rsync from the HPC.
    Remote parameters are read from a config file (preferred) or from explicit args.
    Returns the absolute local path (may still not exist if copy failed)."""
    local_path = os.path.abspath(os.path.join(os.getcwd(), rel_path))
    if os.path.exists(local_path):
        if verbose:
            print(f'Local file exists: {local_path}')
        return local_path

    # If running on the HPC, we expect files to be local on that host; just return path
    if is_hpc_host():
        if verbose:
            print('Detected HPC host: not attempting remote copy; expecting local file to exist on HPC.')
        return local_path

    # Load configuration (YAML/JSON). Explicit args override config values.
    conf = load_remote_config(config_path=config_path, verbose=verbose)
    remote_server = remote_server or conf.get('IDNET_HPC_SERVER') or conf.get('remote_server') or conf.get('server')
    remote_user = remote_user or conf.get('IDNET_HPC_USER') or conf.get('remote_user') or conf.get('user')
    remote_base = remote_base or conf.get('IDNET_HPC_BASE') or conf.get('remote_base') or conf.get('base')

    if not remote_server:
        if verbose:
            print(f'File not found locally: {local_path} and remote server not configured in config file.')
        return local_path

    # Build remote path
    remote_rel = rel_path.lstrip('./')
    if remote_base:
        remote_path = os.path.join(remote_base, remote_rel)
    else:
        remote_path = remote_rel

    user_prefix = f'{remote_user}@' if remote_user else ''
    remote_spec = f'{user_prefix}{remote_server}:{remote_path}'

    # Ensure destination directory exists
    dest_dir = os.path.dirname(local_path)
    os.makedirs(dest_dir, exist_ok=True)

    # Use rsync for robust transfer
    rsync_cmd = shutil.which('rsync')
    if not rsync_cmd:
        if verbose:
            print('rsync is not available on this system. Please install rsync to enable remote copy.')
        return local_path

    cmd = [rsync_cmd, '-avz', '--progress', remote_spec, dest_dir]
    if verbose:
        print('Running:', ' '.join(cmd))
    try:
        res = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            if verbose:
                print('rsync failed. stdout:')
                print(res.stdout)
                print('stderr:')
                print(res.stderr)
            return local_path
    except Exception as e:
        if verbose:
            print(f'Error while running rsync: {e}')
        return local_path

    if os.path.exists(local_path):
        if verbose:
            print(f'Copied successfully: {local_path}')
    else:
        if verbose:
            print('rsync reported success but file missing locally; see rsync output above.')

    return local_path