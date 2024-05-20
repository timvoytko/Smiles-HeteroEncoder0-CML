import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(full_data):
    full_data['smiles_len'] = full_data['smiles'].apply(len)
    full_data = full_data[(full_data.smiles_len < 75) & (full_data.smiles_len > 35)]
    return full_data

def filter_data(full_data, twoarylaminopyr_m):
    matches = [False] * full_data.shape[0]
    for i, val in enumerate(full_data.kekule_smiles):
        mol = Chem.MolFromSmiles(replace_atoms(val, reverse=True))
        matches[i] = mol.HasSubstructMatch(twoarylaminopyr_m)
    full_data['is_have_2aapd'] = matches
    return full_data[full_data['is_have_2aapd'] == True]

def train_test_split(full_data):
    full_train = full_data.sample(frac=0.8, random_state=42)
    full_test = full_data.drop(full_train.index)
    return full_train, full_test

def preprocess_embeddings(embeds_df_train, embeds_df_test):
    sc = StandardScaler().fit(embeds_df_train)
    embeds_df_train = sc.transform(embeds_df_train)
    embeds_df_test = sc.transform(embeds_df_test)
    return embeds_df_train, embeds_df_test, sc

def preprocess_energy(energy_df_train, energy_df_test):
    sc_energy = StandardScaler().fit(energy_df_train.to_numpy().reshape(-1, 1))
    energy_df_train = sc_energy.transform(energy_df_train.to_numpy().reshape(-1, 1))
    energy_df_test = sc_energy.transform(energy_df_test.to_numpy().reshape(-1, 1))
    return energy_df_train, energy_df_test, sc_energy

def vectorize_smiles(data, charset, char_to_int, embed):
    one_hot = np.zeros((data.shape[0], embed, len(charset)), dtype=bool)
    for i, smile in enumerate(data):
        one_hot[i, 0, char_to_int['!']] = True
        for j, c in enumerate(smile):
            one_hot[i, j+1, char_to_int[c]] = True
        one_hot[i, len(smile)+1:, char_to_int['E']] = True
    return one_hot[:, :-1, :], one_hot[:, 1:, :]
