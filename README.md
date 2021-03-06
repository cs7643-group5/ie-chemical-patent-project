# ie-chemical-patent-project

## Model Access
The model files are too large to submit to gradescope for the project
and are too large to push to the repository. As a temporary solution
the models will be publicly available through a google drive link.

For the folder ner_models please follow this link: https://drive.google.com/drive/folders/1CUYnCoA7eHxIzq3Cdput1tBuXoCfbXj3?usp=sharing

For the folder re_models please follow this link: https://drive.google.com/drive/folders/1JiZuBrWVeHyuCCqn5YfZnZjg2gswhkqi?usp=sharing

Both of these folders should be downloaded and placed in the src directory to 
be accessed by the training and evaluation scripts.



## Dataset
The data can be obtained from the ChEMU competition through the following link. http://chemu2020.eng.unimelb.edu.au.
The competition restricted us from sharing it or making it available to any other party. To anyone training or evaluating our models, please download the data from the link above and place the training and development sets in the data folder.

## Data Pre-processing
Loading in the all the data needed for training is found in the preprocessing
folder. Preprocessor corresponds to the pre-processing for named entity recognition, while ee corresponds to the relation extraction models.

Loading in the data for the relation extraction models has two options.
Either directly using run_ee_preprocessor or loading the data from a
pickle file. The later is preferred when training occurs in google Colab.
The function to store the data in the pickle file corresponds to the
store_data() function in utils.py. This saves the file in the folder
data as ee_data.pickle.

The evaluation of task 2 occurs in ee_evaluation.py and task2_evaluate.py.
ee_evaluation.py shares similar functions to preprocess.py
and ee_preprocessor.py but it is tailored to the format
needed for evaluating the model pipeline. task2_evaluate.py
is currently set up to only load data from pickle files.
To save the pickle files run the store_data() function in  ee_evaluation.py
utils.py is also set up to call this function. Two pickle files
are created in the data folder corresponding to ee_eval_data.pickle and
missed_entity_pairs.pickle.

## Training Models

### NER models
The model architecture for the NER model for task 1 and task 2 follows the huggingface model class format and can be loaded in and saved as such.
src/ner_train.py is used to fine-tune the pretrained BioBERT models.

One would make whatever changes in the hyperparameters directly in ner_train.py and then would run
python src/preprocessing/ner_train.py

The model will then be saved in the ner_models directory. 


### relation extraction
The model architecture for the custom relation extraction model
is  seen in the re_models folder as custom_model.py. The r_bert
structure is seen in R_bert.py in the same folder.

re_train.py is set up to take arguments using argparse. For example,
python src/re_train.py -m 'r_bert' -d 1 -t 'colab' -b 32 -lr 1e-5 -e 10

-m corresponds to the relation extraction model that is desired to be trained
if 'r_bert' is passed the r_bert model will be used. If anything else is passed the custom model
will be used. <br>
-d corresponds to the percentage of data to use 1 is 100% 0.2 is 20%. <br>
-t is training type, If 'colab' is specified data will be loaded from
the pickle file ee_data.pickle. <br>
-b corresponds to batch size. <br>
-lr is learning rate. <br>
-e is number of epochs. <br>

the trained model is stored in re_models prefaced with custom_model followed
by the settings for the hyperparameters as the file name.


## Evaluation

### NER models
The saved models for Task 1 and Task 2 are in the ner_models folder. They follow the huggingface model class format and are loaded in as such.
ner_evaluate.py is in the src folder is used to evaluate the models. Only BioBERT models are in this folder as they were the higher performing ones.

One would need to run:
python src/ner_evaluate.py

One would need to modify the load directory to evaluate a newly trained model.

### Task 2
task 2 evaluation is run with task2_evaluate.py with argument <br>
-d corresponding to the percentage of data to be used for evaluation.

Task 2 evaluation sets up a model pipeline using an NER model
and an RE model for event extraction. Please ensure that
the pickle files ee_eval_data.pickle and missed_entity_pairs.pickle
are in the data folder. Also ensure that the models you want
to use are located in the re_models folder.










