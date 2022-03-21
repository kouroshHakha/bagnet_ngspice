from typing import Any, Union, Dict, Mapping

from pathlib import Path

from ruamel.yaml import YAML
import pickle
import h5py
import numpy as np
from numbers import Number


PathLike = Union[str, Path]
yaml = YAML(typ='safe')

def read_yaml(fname: Union[str, Path]) -> Any:
    """Read the given file using YAML.

    Parameters
    ----------
    fname : str
        the file name.

    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    with open(fname, 'r') as f:
        content = yaml.load(f)

    return content

def write_yaml(fname: Union[str, Path], obj: object, mkdir: bool = True) -> None:
    """Writes the given object to a file using YAML format.

    Parameters
    ----------
    fname : Union[str, Path]
        the file name.
    obj : object
        the object to write.
    mkdir : bool
        If True, will create parent directories if they don't exist.

    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    if isinstance(fname, str):
        fpath = Path(fname)
    else:
        fpath = fname

    if mkdir:
        fpath.parent.mkdir(parents=True, exist_ok=True)

    with open(fpath, 'w') as f:
        yaml.dump(obj, f)

def get_full_name(name: str, prefix: str = '', suffix: str = ''):
    """Returns a full name given a base name and prefix and suffix extensions

    Parameters
    ----------
    name: str
        the base name.
    prefix: str
        the prefix (default='')
    suffix
        the suffix (default='')

    Returns
    -------
    full_name: str
        the fullname
    """
    if prefix:
        name = f'{prefix}_{name}'
    if suffix:
        name = f'{name}_{suffix}'
    return name

def read_pickle(fname: Union[str, Path]) -> Any:
    """Read the given file using Pickle.

    Parameters
    ----------
    fname : str
        the file name.

    Returns
    -------
    content : Any
        the object returned by pickle.
    """
    with open(fname, 'rb') as f:
        content = pickle.load(f)

    return content

def write_pickle(fname: Union[str, Path], obj: object, mkdir: bool = True) -> None:
    """Writes the given object to a file using pickle format.

    Parameters
    ----------
    fname : Union[str, Path]
        the file name.
    obj : object
        the object to write.
    mkdir : bool
        If True, will create parent directories if they don't exist.

    Returns
    -------
    content : Any
        the object returned by pickle.
    """
    if isinstance(fname, str):
        fpath = Path(fname)
    else:
        fpath = fname

    if mkdir:
        fpath.parent.mkdir(parents=True, exist_ok=True)

    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def read_hdf5(fpath: PathLike) -> Dict[str, Any]:
    def _load_hdf5_helper(root: h5py.Group) -> Dict[str, Any]:
        init_dict = {}
        for k, v in root.items():
            if isinstance(v, h5py.Dataset):
                init_dict[k] = np.array(v)
            elif isinstance(v, h5py.Group):
                init_dict[k] = _load_hdf5_helper(v)
            else:
                raise ValueError(f'Does not support type {type(v)}')
        return init_dict

    with h5py.File(fpath, 'r') as f:
        return _load_hdf5_helper(f)



def write_hdf5(data_dict: Mapping[str, Any], fpath: PathLike) -> None:

    def _save_as_hdf5_helper(obj: Mapping[str, Union[Mapping, np.ndarray]], root: h5py.File):
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                root.create_dataset(name=k, data=v)
            elif isinstance(v, dict):
                grp = root.create_group(name=k)
                _save_as_hdf5_helper(v, grp)
            elif isinstance(v, Number):
                root.create_dataset(name=k, data=v)
            else:
                raise ValueError(f'Does not support type {type(v)}')

    with h5py.File(fpath, 'w') as root:
        _save_as_hdf5_helper(data_dict, root)


class HDF5Error(Exception):
    pass
