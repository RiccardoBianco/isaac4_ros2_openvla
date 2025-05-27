####################################################################
# ^ This script splits a collected dataset (using Isaac Sim) into
# ^ training and validation sets. It takes a directory of images and
# ^ splits them into two subdirectories: 'train' and 'val'.
# ^ The split is done based on a specified ratio (default is 80% train, 20% val).
#
# ! Each episode (trajectory) has to be a single .npy file, under the 
# ! directory 'episodes'.
########################################################################

import os
import shutil
import math

MULTICUBE = False

def empty_dir(dir_path):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(dir_path)

def split_dataset(input_dir, output_dir, train_ratio=1.0):
    # Lista ordinata dei file
    all_files = sorted(os.listdir(input_dir))
    all_files = [f for f in all_files if os.path.isfile(os.path.join(input_dir, f))]

    # Check if input is empty
    if not all_files:
        print(f"⚠️  Input directory '{input_dir}' is empty. Aborting split to avoid deleting output data.")
        return

    total_files = len(all_files)
    train_count = math.floor(total_files * train_ratio)

    train_files = all_files[:train_count]
    print(f"Number of TRAIN files: {len(train_files)}")
    val_files = all_files[train_count:]
    print(f"Number of VAL files: {len(val_files)}")

    # Percorsi delle cartelle
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    # Svuota (o crea) le cartelle
    empty_dir(train_dir)
    empty_dir(val_dir)
    # Sposta i file nel training set
    for f in train_files:
        shutil.move(os.path.join(input_dir, f), os.path.join(train_dir, f))

    # Sposta i file nel validation set
    for f in val_files:
        shutil.move(os.path.join(input_dir, f), os.path.join(val_dir, f))


    print(f"Totale file: {total_files}")
    print(f"Train: {len(train_files)} file -> {train_dir}")
    print(f"Val: {len(val_files)} file -> {val_dir}")


def split_dataset_taskwise(input_dir, output_dir):
    all_files = sorted(os.listdir(input_dir))
    all_files = [f for f in all_files if f.endswith(".npy") and os.path.isfile(os.path.join(input_dir, f))]

    if not all_files:
        print(f"⚠️  Input directory '{input_dir}' is empty. Aborting.")
        return

    # Percorsi delle cartelle
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    empty_dir(train_dir)
    empty_dir(val_dir)

    block_size = 800
    train_per_block = 750
    val_per_block = 50

    total_blocks = len(all_files) // block_size

    for i in range(total_blocks):
        block_start = i * block_size
        block_train = all_files[block_start:block_start + train_per_block]
        block_val = all_files[block_start + train_per_block:block_start + train_per_block + val_per_block]

        print(f"Block {i+1}: train {len(block_train)}, val {len(block_val)}")

        for f in block_train:
            shutil.move(os.path.join(input_dir, f), os.path.join(train_dir, f))
        for f in block_val:
            shutil.move(os.path.join(input_dir, f), os.path.join(val_dir, f))

    print(f"Split completato: {len(os.listdir(train_dir))} train, {len(os.listdir(val_dir))} val")


# Esempio di utilizzo:
# split_dataset("path/to/input_dir", "path/to/output_dir")
if __name__ == "__main__":
    input_dir = "./isaac_ws/src/output/episodes"  # Sostituisci con il tuo percorso
    output_dir = "./rlds_dataset_builder/sim_data_custom_v0/data"  # Sostituisci con il tuo percorso
    if MULTICUBE:
        # Split dataset into train and val
        split_dataset_taskwise(input_dir, output_dir)
    else:
        # Split dataset into train and val
        split_dataset(input_dir, output_dir)


