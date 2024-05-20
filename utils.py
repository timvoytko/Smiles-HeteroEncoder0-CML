import numpy as np

def replace_atoms(smi, reverse=False):
    if not reverse:
        d = {'Br': 'X', 'Cl': 'Y', 'Se': 'Z', 'br': 'x', 'cl': 'y', 'se': 'z', '-]': 'V]'}
    else:
        d = {'X': 'Br', 'Y': 'Cl', 'Z': 'Se', 'x': 'br', 'y': 'cl', 'z': 'se', 'V]': '-]'}
    new_smi = smi
    for key in d:
        new_smi = new_smi.replace(key, d[key])
    return new_smi

def from_one_hot_to_smiles(curr, int_to_char_smiles):
    return "".join([int_to_char_smiles[idx] for idx in np.argmax(curr, axis=1)])
