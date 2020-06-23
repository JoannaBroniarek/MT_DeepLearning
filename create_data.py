import SimpleITK as sitk
import os
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
from dltk.io.preprocessing import whitening

def read_clinical_data():
    subject_data = pd.read_csv('/myworkdir/data/ds000221-download/participants.tsv', sep='\t')
    
    age_dict = {'20-25':1, '25-30':2, '30-35':3, 
            '35-40':4, '40-45':5, '45-50':6, 
            '50-55':7, '55-60':8, '60-65':9, 
            '65-70':10, '70-75':11, '75-80':12, 
            np.nan:-1}
    
    subject_data['isFemale'] = (subject_data['gender'] == "F")*1
    subject_data['age_code'] = subject_data['age (5-year bins)'].apply(lambda r: age_dict[r]).astype('int')

    return subject_data[['participant_id','isFemale', 'age_code']]


def load_img(file_path):
    """ 
    Loads the training image as numpy arrays.
    
    """
    #  Construct a file path to read an image from.
    img_ = file_path
        
    # Read the .nii.gz image and get the numpy array:
    sitk_img = sitk.ReadImage(img_)
    img_ = sitk.GetArrayFromImage(sitk_img)
    img_ = np.moveaxis(img_, 0, -1)
    return img_


def create_tf_feature(image, bvals, directions, isfemale, age):
    '''
    Create a feature dict for TFRecords    
    '''
    feature = {'dwi': _bytes_feature(tf.io.serialize_tensor(image)),
               'bvalues' : _bytes_feature(tf.io.serialize_tensor(bvals)),
               'directions' : _bytes_feature(tf.io.serialize_tensor(directions)),
               'isfemale' : _int64_feature(isfemale),
               'age' : _int64_feature(age)  }
    return feature


def get_one_pair(image, bvalues, directions, isfemale, age, pair_id):
    """
    Function takes volume images of one patient and selects only the volumes with pair (b_low, b_high).  

    - Image standarization
    - Resize 
    - Create tf.feature object needed to build tr.Records file
    """
    b_5_ids = np.where(bvalues==5)[0]
    high_ids = np.where(bvalues>5)[0]
    pairs_ids = [(b, i) for b in b_5_ids for i in high_ids]
    b5, bhigh = pairs_ids[pair_id]

    # Standardize images and build a "pair"
    new_img = np.empty_like(image[:, :, :, 1:3])
    new_img[:, :, :, 0] = tf.image.per_image_standardization(image[:, :, :, b5])
    new_img[:, :, :, 1] = tf.image.per_image_standardization(image[:, :, :, bhigh])
    # Resize to higher resolution
    new_img = tf.image.resize(new_img, [256, 256], method = 'lanczos5')
    # Create tf.feature
    feature = create_tf_feature(new_img, bvalues[[b5, bhigh]], directions[:, [b5, bhigh]], isfemale, age)
    return feature



def into_spherical(directions):
    """ 
    Convert directions from cartesian into spherical coordinates.
    Constant Radius = 1
    """
    x, y, z = directions
    theta = np.arccos(z)/np.pi
    phi = np.arctan2(y, x)/np.pi
    return np.vstack((theta, phi))



###########  Building the TFRecord file #############

## I copied few functions from the tensoflow tutoral: [https://www.tensorflow.org/tutorials/load_data/tfrecord]

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#########################################################################



def create(filename, ids):
    """
    Function runs all needed steps to create tf.records file.
    ---- deprecated ---> check next function: new_create.
    """

    # read the clinical data : age & gender
    clinical_df = read_clinical_data()
    
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(filename)

    # Iterate through directories of OpenNeuro dataset
    dataset_dir = '/myworkdir/data/ds000221-download'    
    
    for subject_dir in tqdm(ids): #tqdm(os.listdir(dataset_dir)):

        # check if it is the correct 'subject' directory 
        if not subject_dir.startswith('sub'):
            continue

        # find gender and age of the current subject
        clinic_row = clinical_df[clinical_df['participant_id']==subject_dir]
        isfemale = clinic_row['isFemale'].tolist()[0]
        age = clinic_row['age_code'].tolist()[0]
        
        # if there is no information about age, skip this subject 
        if isfemale==-1: continue
        
        #select the session directory which contains the 'dwi' sub-directory
        subject_dir  = os.path.join(dataset_dir,subject_dir)
        sess_with_dwi_dirs = [ses_dir for ses_dir in os.listdir(subject_dir) if 'dwi' in os.listdir(os.path.join(subject_dir, ses_dir))]
        
        for sess_dir in sess_with_dwi_dirs:
            dwi_path = os.path.join(subject_dir, sess_dir, 'dwi')
            
            
            bvalues = np.loadtxt(glob.glob(dwi_path + "/*.bval")[0])       # Read .bval file
            directions = np.loadtxt(glob.glob(dwi_path + "/*.bvec")[0])    # Read .bvec file
            image = load_img(glob.glob(dwi_path + "/*.nii.gz")[0])         # Load .nii.gz file   
            
            # Convert direction into spherical variables
            directions = into_spherical(directions)
            
            # Data Augmentation & Writing to a TFRecords file 
            for feature in augment(image, bvalues, directions, isfemale, age):
                
                #Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

    writer.close()
    
def new_create(filename, ids):
    """
    Function runs all needed steps to create tf.records file.
    Difference with "create" function is here a shuffling of patients is done.
    """
    
    # read the clinical data : age & gender
    clinical_df = read_clinical_data()

    # open the TFRecords file
    writer = tf.io.TFRecordWriter(filename)
    
    # Iterate through directories of OpenNeuro dataset
    dataset_dir = '/myworkdir/data/ds000221-download'
    counter = 0
    for pair_id in tqdm(range(0, 420, 28)):
        if counter > 24: break

        for subject_dir in ids:
            if counter > 24:
                break
    
#             print(".", end='')
            if not subject_dir.startswith('sub'):
                continue

            # find gender and age of the current subject
            clinic_row = clinical_df[clinical_df['participant_id']==subject_dir]
            isfemale = clinic_row['isFemale'].tolist()[0]
            age = clinic_row['age_code'].tolist()[0]

            # if there is no information about age, skip this subject 
            if isfemale==-1: continue

            dwi_path = os.path.join(dataset_dir, subject_dir, 'ses-01', 'dwi')
            if not os.path.isdir(dwi_path): 
                continue
            
            bvalues = np.loadtxt(glob.glob(dwi_path + "/*.bval")[0])       # Read .bval file
            directions = np.loadtxt(glob.glob(dwi_path + "/*.bvec")[0])    # Read .bvec file
            image = load_img(glob.glob(dwi_path + "/*.nii.gz")[0])         # Load .nii.gz file   

            # Convert direction into spherical variables
            directions = into_spherical(directions)

            ####################################################
            # Get one pair of (b_low, b_high)
            one_feature = get_one_pair(image, bvalues, directions, isfemale, age, pair_id)
            
            #Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=one_feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            counter += 1
    writer.close()    

######################################################################    
###############  Read and Parse the TFRecords file ###################
######################################################################



def _decode(example_proto):
    feature_description = {'dwi': tf.io.FixedLenFeature([], tf.string),
                           'bvalues': tf.io.FixedLenFeature([], tf.string),
                           'directions': tf.io.FixedLenFeature([], tf.string),
                           'isfemale': tf.io.FixedLenFeature([], tf.int64),
                           'age': tf.io.FixedLenFeature([], tf.int64)}
    
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    dwi = tf.io.parse_tensor(example['dwi'], out_type=tf.float32)
    bvalues = tf.io.parse_tensor(example['bvalues'], out_type=tf.float64)
    directions = tf.io.parse_tensor(example['directions'], out_type=tf.float64)
    isfemale = example['isfemale']
    age = example['age']
    return dwi, bvalues, directions, isfemale, age


def preprocess_input_fn(img, bvals, directions, isfemale, age):
    """
    Function parses data to be compatible with tf.Model input/output format.
    """
    def _preprocess_input(img, bvals, directions, isfemale, age):
        labels = {'dense_1': isfemale, 'dense': age}
        return (img, labels)    
    return _preprocess_input(img, bvals, directions, isfemale, age)


def parse_dataset(filename):
    """
    Function reads tf.records dataset and parses each example.
    """
    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset =  raw_dataset.map(_decode)
    return parsed_dataset.map(preprocess_input_fn)
