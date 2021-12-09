# ie-chemical-patent-project

## Dataset
the data can be obtained from the ChEMU competition through the following link. http://chemu2020.eng.unimelb.edu.au.

## Data Pre-processing
Loading in the all the data needed for training is found in the preprocessing
folder. Preprocessor corresponds to the pre-processing for named entity recognition, while ee corresponds to the relation extraction models.

Loading in the data for the relation extraction models has two options. Either directly using run_ee_preprocessor or loading the data from a pickle file.
The later is preferred when training occurs in google Colab.
The function to store the data in the pickle file corresponds to the
store_data() function in utils.py. This saves the file in the folder
data as ee_data.pickle.

The evaluation of task 2 occurs in ee_evaluation.py and evaluate_task2.py.
ee_evaluation.py shares similar functions to preprocess.py
and ee_preprocessor.py but it is tailored to the format
needed for evaluating the model pipeline. evaluate_task2.py
is currently set up to only load data from pickle files.
To save the pickle files run the store_data() function in  ee_evaluation.py
utils.py is also set up to call this function. Two pickle files
are created in the data folder corresponding to ee_eval_data.pickle and
missed_entity_pairs.pickle.

## Training Models

### NER models
The model architecture for the NER model for task 1 and task 2
is  seen in the re_models folder. They follow the huggingface model class format and can be loaded in as such.
run_preprocessor.py is used to fine-tune the pretrained BioBERT models.

One would make whatever changes in the hyperparameters directly in run_preprocessor.py and then would run
python src/preprocessing/run_preprocessor.py

The model will then be saved in the re_models directory



### relation extraction custom model
The model architecture for the custom relation extraction model
is  seen in the re_models folder as custom_model.py.
train_re.py is set up to use the model class in custom_model.py for
training. 

train_re.py is set up to take arguments using argparse. For example,
python src/train_re.py -d 1 -t 'colab' -b 32 -lr 1e-5 -e 10

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
The model architecture for the NER model for task 1 and task 2
is  seen in the re_models folder. They follow the huggingface model class format and can be loaded in as such.
run_preprocessor.py is used to fine-tune the pretrained BioBERT models.

One would make whatever changes in the hyperparameters directly in run_preprocessor.py and then would run
python src/preprocessing/run_preprocessor.py

The model will then be saved in the re_models directory

### Task 2
task 2 evaluation is run with evaluate_task2.py with argument <br>
-d corresponding to the percentage of data to be used for evaluation.

Task 2 evaluation sets up a model pipeline using an NER model
and an RE model for event extraction. Please ensure that
the pickle files ee_eval_data.pickle and missed_entity_pairs.pickle
are in the data folder. Also ensure that the models you want
to use are located in the re_models folder.










