import typer
from loguru import logger
from pathlib import Path
import sys
import pandas as pd
import tensorflow as tf
from utils import patient_dataset_splitter, df_to_dataset, create_tf_categorical_feature_cols, build_vocab_files
from typing import List, Optional
import pickle

app = typer.Typer(add_completion=False)


def build_sequential_model(feature_layer):
    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(175, activation='relu'),
        tf.keras.layers.Dense(75, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model


@app.command()
@logger.catch(onerror=lambda _: sys.exit(1))
def train(input_data: Path = typer.Argument(..., help="Input file path; usually final.csv"),
          output_dir: Path = typer.Argument(...),
          epochs: int = typer.Argument(30, help="Training epochs"),
          patient_key: str = typer.Option("patient_TrustNumber", help="Key column in the INPUT_DATA"),
          predictor_col: str = typer.Option("Event", help="Column for prediction in the INPUT_DATA"),
          vocab_dir: str = typer.Option('survival_vocab', help="Name of vocabulary list folder relative to the OUTPUT_DIR"),
          train_dir: str = typer.Option('training', help="Name of training folder relative to the OUTPUT_DIR"),
          checkpoint: str = typer.Option('ann.h5', help="Checkpoint filename for training"),
          batch_size: int = typer.Option(10, help="Training batch size"),
          lr: float = typer.Option(0.000001, help="Learning rate"),
          loss: str = typer.Option('mse', help="Loss function"),
          metrics: List[str] = typer.Option(['accuracy'], help="Loss metrics"),
          patience: int = typer.Option(5, help="Wait for how many number of epochs before stop the iteration early.")):
    """
    Train a fully convolutional network. The INPUT_DATA is path to the final.csv file.
    """
    if output_dir.is_dir():
        logger.warning(f"Output directory {output_dir} exist. This training may overwrite existing files.")
    else:
        output_dir.mkdir(exist_ok=False)
        logger.info(f"Results will be stored in {output_dir}.")

    # create local dirs
    vocab_dir = output_dir / vocab_dir
    vocab_dir.mkdir(exist_ok=True)

    train_dir = output_dir / train_dir
    train_dir.mkdir(exist_ok=True)

    # read the input data
    survival_df = pd.read_csv(input_data)

    # Define columns
    categorical_col_list = ['p_basal_anterior', 'p_basal_anteroseptum', 'p_mid_anterior', 'p_mid_anteroseptum',
                            'p_apical_anterior',
                            'p_apical_septum', 'p_basal_inferolateral', 'p_basal_anterolateral', 'p_mid_inferolateral',
                            'p_mid_anterolateral',
                            'p_apical_lateral', 'p_basal_inferoseptum', 'p_basal_inferior', 'p_mid_inferoseptum',
                            'p_mid_inferior', 'p_apical_inferior']

    # select only specific volumes
    survival_df = survival_df[categorical_col_list + [predictor_col, patient_key]]
    logger.info(f"{input_data}: {survival_df.shape[0]} rows with {survival_df.shape[1]} columns")

    # fix column types
    for c in categorical_col_list:
        survival_df[c] = survival_df[c].astype(str)

    # split the data
    d_train, d_val, d_test = patient_dataset_splitter(survival_df, patient_key)
    assert len(d_train) + len(d_val) + len(d_test) == len(survival_df)

    logger.info(f"Train data: {d_train.shape[0]}")
    logger.info(f"Validation data: {d_val.shape[0]}")
    logger.info(f"Test data: {d_test.shape[0]}")

    # drop the patient key
    d_train = d_train.drop(columns=[patient_key])
    d_val = d_val.drop(columns=[patient_key])
    d_test = d_test.drop(columns=[patient_key])

    # save the splitting data for later use
    d_train.to_csv(output_dir / "train_data.csv")
    d_val.to_csv(output_dir / "val_data.csv")
    d_test.to_csv(output_dir / "test_data.csv")

    # Create vocabulary list from the training data
    vocab_list = build_vocab_files(d_train, categorical_col_list, vocab_dir=vocab_dir)
    logger.debug(f"Vocabulary list = {vocab_list}")

    # Create categorical features
    tf_cat_col_list = create_tf_categorical_feature_cols(categorical_col_list, vocab_dir=vocab_dir)
    logger.info(f"Created {len(tf_cat_col_list)} category lists.")
    claim_feature_layer = tf.keras.layers.DenseFeatures(tf_cat_col_list)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=train_dir / checkpoint, save_best_only=False, verbose=1)

    # Convert dataset from Pandas dataframes to TF dataset
    survival_train_ds = df_to_dataset(d_train, predictor_col, batch_size=batch_size)
    survival_val_ds = df_to_dataset(d_val, predictor_col, batch_size=batch_size)
    # survival_test_ds = df_to_dataset(d_test, predictor_col, batch_size=batch_size)

    # build the model
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    model = build_sequential_model(claim_feature_layer)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # train
    history = model.fit(survival_train_ds,
                        validation_data=survival_val_ds,
                        callbacks=[cp_callback, early_stop],
                        epochs=epochs)
    pickle.dump(model, open(output_dir / 'model.pkl', 'wb'))
    logger.info(f"Saved model in {output_dir / 'model.pkl'}")

    hist_db = pd.DataFrame(history.history)
    hist_file = train_dir / "train_history.csv"
    hist_db.to_csv(hist_file, index=False)
    logger.info(f"Saved training history in {hist_file}")
