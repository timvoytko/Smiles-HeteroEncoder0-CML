import numpy as np
from rdkit import Chem

def latent_to_smiles_n_canon(latent, energy, to_smiles_model, smiles_latent_to_states, char_to_int_smiles, int_to_char_smiles, embed_smiles):
    states_smiles = smiles_latent_to_states.predict([latent, energy])
    to_smiles_model.layers[1].reset_states(states=[states_smiles[0], states_smiles[1]])
    startidx = char_to_int_smiles["!"]
    samplevec = np.zeros((1, 1, len(char_to_int_smiles)))
    samplevec[0, 0, startidx] = 1
    smiles = ""
    for i in range(embed_smiles):
        o = to_smiles_model.predict(samplevec)
        sampleidx = np.argmax(o)
        samplechar = int_to_char_smiles[sampleidx]
        if samplechar != "E":
            smiles += samplechar
            samplevec = np.zeros((1, 1, len(char_to_int_smiles)))
            samplevec[0, 0, sampleidx] = 1
        else:
            break
    return smiles
