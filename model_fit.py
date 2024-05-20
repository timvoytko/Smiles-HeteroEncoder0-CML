import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import History, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import datetime

def build_model(X_smiles_train_shape, X_canon_train_shape, embeds_df_train_shape, energy_df_train_shape, output_dim_decoder1, output_dim_decoder2, latent_dim, lstm_dim, bn_momentum):
    input_canonical = Input(X_canon_train_shape[1:])
    input_smiles = Input(X_smiles_train_shape[1:])
    input_embeddings = Input(embeds_df_train_shape[1:])
    input_energy = Input(energy_df_train_shape[1:])

    # Encoding
    smiles_encoder = LSTM(lstm_dim, return_state=True, return_sequences=True)(input_smiles)
    _, state_h1, state_c1 = LSTM(lstm_dim, return_state=True)(smiles_encoder)
    cannon_encoder = LSTM(lstm_dim, return_state=True, return_sequences=True)(input_canonical)
    _, state_h2, state_c2 = LSTM(lstm_dim, return_state=True)(cannon_encoder)

    encoder_embeds = Dense(16, activation='relu')(BatchNormalization()(Dense(32, activation='relu')(Dense(64, activation='relu')(input_embeddings))))

    states = Concatenate()([state_h1, state_c1, state_h2, state_c2, encoder_embeds])
    states = BatchNormalization(momentum=bn_momentum)(states)
    neck_outputs = Dense(latent_dim, activation='relu')(states)

    # Decoding
    to_decode = Concatenate()([neck_outputs, input_energy])
    decode_h1 = Dense(lstm_dim, activation='relu')(to_decode)
    decode_c1 = Dense(lstm_dim, activation='relu')(to_decode)
    decode_h2 = Dense(lstm_dim, activation='relu')(to_decode)
    decode_c2 = Dense(lstm_dim, activation='relu')(to_decode)

    decoder1_inputs = Input(X_smiles_train_shape[1:])
    decoder2_inputs = Input(X_canon_train_shape[1:])
    decoder1_lstm = LSTM(lstm_dim, return_sequences=True)
    decoder2_lstm = LSTM(lstm_dim, return_sequences=True)
    decoder1_outputs = Dense(output_dim_decoder1, activation='softmax')(decoder1_lstm(decoder1_inputs, initial_state=[decode_h1, decode_c1]))
    decoder2_outputs = Dense(output_dim_decoder2, activation='softmax')(decoder2_lstm(decoder2_inputs, initial_state=[decode_h2, decode_c2]))

    model = Model(inputs=[[input_smiles, input_canonical, input_embeddings, input_energy], [decoder1_inputs, decoder2_inputs]], outputs=[decoder1_outputs, decoder2_outputs])
    return model

def compile_and_train(model, X_train, y_train, X_test, y_test, learning_rate=0.005, epochs=25, batch_size=64):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', run_eagerly=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        History(),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=1, epsilon=1e-4),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        ModelCheckpoint(filepath="model.h5", monitor='val_loss', save_best_only=True)
    ]
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, y_test), callbacks=callbacks)
    return history
