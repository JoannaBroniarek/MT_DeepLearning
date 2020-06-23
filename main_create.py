from create_data import *
import numpy as np
import random
import os

def main():
    """
    Before running, specify paths of tf.records file:
    In the last line, in "new create" as a first argument specify the output file. As the second argument specify the type of dataset to be generated (train/test/val).
    """

    print(os.getcwd())
    # Read subject ids with available dwi data
    with open('/myworkdir/data/SubjectsOfInterest.txt', 'r') as f:
        subjects = [line.split()[0] for line in f.readlines()]

    # Shuffle subject ids
    random.shuffle(subjects)

    # Split intro train/test/set
    num_sub = len(subjects)
    train_size = np.floor(0.6*num_sub).astype('int')
    val_size = np.ceil(0.2*num_sub).astype('int')
    test_size = num_sub - train_size - val_size

    # Build a dict of splitted subjects
    subject_dict = {'train_subjects':subjects[:train_size], "val_subjects" : subjects[train_size: train_size+val_size], 'test_subjects' : subjects[train_size+val_size:]}

    # Run generator of tfRecord file
    new_create('/myworkdir/data/small_val.tfrecords',  ids=subject_dict['val_subjects'])



if __name__ == "__main__":
    main()
