# # if ie-chemical-patent-project is working directory
# # run following command to test out data loading
# # python src/preprocessing/run_ee_preprocessor.py



import preprocessor

print('loading ee data')
sentences_train, tags_train, ee_sentences_train, ee_labels_train = preprocessor.load_data_ee()
print('ee data load complete')