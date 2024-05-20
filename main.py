import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import datetime
import warnings
from google.colab import drive
import os

# Import custom modules
from utils import replace_atoms, from_one_hot_to_smiles, vectorize
from data_preprocessing import load_and_preprocess_data
from model import build_model, CustomChemLoss
from train import train_model
from generate import generate_molecules

DATASET_PATH = ''
OUTPUT_PATH =''
def main():
    # Set up GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load and preprocess data
    drive.mount('/content/drive')
    full_data = pd.read_csv(DATASET_PATH)
    preprocessed_data = load_and_preprocess_data(full_data)

    smiles_df_train, cannonical_df_train, embeds_df_train, energy_df_train, \
    smiles_df_test, cannonical_df_test, embeds_df_test, energy_df_test, \
    charset_smiles, charset_cannon, n_chars_smiles, n_chars_cannon, \
    char_to_int_smiles, int_to_char_smiles, char_to_int_cannon, int_to_char_cannon, \
    embed_smiles, embed_cannon = preprocessed_data

    # Vectorize data
    X_smiles_train, y_smiles_train = vectorize(smiles_df_train, charset_smiles, char_to_int_smiles, embed_smiles)
    X_smiles_test, y_smiles_test = vectorize(smiles_df_test, charset_smiles, char_to_int_smiles, embed_smiles)
    X_canon_train, y_canon_train = vectorize(cannonical_df_train, charset_cannon, char_to_int_cannon, embed_cannon)
    X_canon_test, y_canon_test = vectorize(cannonical_df_test, charset_cannon, char_to_int_cannon, embed_cannon)

    # Build the model
    model = build_model(X_canon_train, X_smiles_train, embeds_df_train, energy_df_train, y_smiles_train, y_canon_train)

    # Train the model
    history, model_checkpoint_callback = train_model(
        model, 
        [X_smiles_train, X_canon_train, embeds_df_train, energy_df_train], 
        [y_smiles_train, y_canon_train], 
        [X_smiles_test, X_canon_test, embeds_df_test, energy_df_test], 
        [y_smiles_test, y_canon_test]
    )

    # Generate new molecules
    df_gen = full_data.sort_values('energy').head(1000).reset_index(drop=True)
    resultDict = generate_molecules(
        model_checkpoint_callback, df_gen, charset_smiles, char_to_int_smiles, int_to_char_smiles, 
        embed_smiles, charset_cannon, char_to_int_cannon, int_to_char_cannon, embed_cannon
    )

    # Save results

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    

    counter = 1
    for key in resultDict.keys():
        os.mkdir(OUTPUT_PATH + '/' + str(counter))
        with open(OUTPUT_PATH + '/' + str(counter) + '/input.smi', 'w') as f:
            print(key, file=f)
        with open(OUTPUT_PATH + '/' + str(counter) + '/output.smi', 'w') as f:
            for entry in resultDict[key]:
                print(entry, file=f)
        counter += 1

    with open(OUTPUT_PATH + '/total.smi', 'w') as f:
        f.write('TOTAL\n')
    with open(OUTPUT_PATH + '/total.smi', 'a') as f:
        for key in resultDict.keys():
            for entry in resultDict[key]:
                f.write(entry + '\n')

if __name__ == "__main__":
    main()
