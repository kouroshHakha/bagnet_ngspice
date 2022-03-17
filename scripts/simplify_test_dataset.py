

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

test_dataset = {}
for file in fdir.iterdir():
    if file.stem.startswith('oracle_qs'):
        idx = int(file.stem.split('_')[-1])
        data = read_pickle(file)
        test_dataset[idx] = []
        for sample_cnt in range(len(data['inputs1'])):
            sample = dict(
                input1={'params': as_float(data['inputs1'][sample_cnt].value_dict),
                        'specs':  as_float(data['inputs1'][sample_cnt].key_specs)},
                input2={'params': as_float(data['inputs2'][sample_cnt].value_dict),
                        'specs':  as_float(data['inputs2'][sample_cnt].key_specs)},
                critical_labels=data['critical_specs'],
            )
            for key in data:
                if key not in ('inputs1', 'inputs2', 'critical_specs'):
                    sample[key] = int(data[key][sample_cnt])
            test_dataset[idx].append(sample)

write_pickle(fdir / 'test.pkl', test_dataset)