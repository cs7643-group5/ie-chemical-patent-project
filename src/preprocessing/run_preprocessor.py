import preprocessor

# if ie-chemical-patent-project is working directory
# run following command to test out data loading
# python src/preprocessing/run_preprocessor.py

train_data, val_data = preprocessor.load_data()


print('\nexample training')
print('sentence')
print(train_data[0][-1])
print('sentence tags')
print(train_data[1][-1])

print('\nexample validation')
print('sentence')
print(val_data[0][-1])
print('sentence tags')
print(val_data[1][-1])