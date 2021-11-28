import preprocessor

# if ie-chemical-patent-project is working directory
# run following command to test out data loading
# python src/preprocessing/run_preprocessor.py

train_dataloader, val_dataloader = preprocessor.load_data()

for batch_sents, batch_masks, batch_labels in train_dataloader:
    print(f'\nbatch sents shape: {batch_sents.shape}')
    print(f'batch masks shape: {batch_masks.shape}')
    print(f'batch labels shape: {batch_labels.shape}\n')


# print('\nexample training')
# print('sentence')
# print(train_data[0][-1])
# print('sentence tags')
# print(train_data[1][-1])
#
# print('\nexample validation')
# print('sentence')
# print(val_data[0][-1])
# print('sentence tags')
# print(val_data[1][-1])