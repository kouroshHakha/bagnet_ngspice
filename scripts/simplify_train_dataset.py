

from builtins import breakpoint
from utils.file import read_pickle, write_pickle
from utils.pdb import register_pdb_hook
register_pdb_hook()

from pathlib import Path

def as_float(map):
    for key, value in map.items():
        map[key] = float(value)

    return map

fdir = Path('outputs/log/two_stage/oracle_12-03-2022_01-19-36')

db = read_pickle(fdir / 'db.pkl')

train_dataset = {}
for iter_cnt in range(len(db)):
    train_dataset[iter_cnt] = []
    for dsn in db[iter_cnt]:
        sample = dict(
            params=as_float(dsn.value_dict),
            specs=as_float(dsn.key_specs)
        )
        train_dataset[iter_cnt].append(sample)

write_pickle(fdir / 'train.pkl', train_dataset)
breakpoint()