# if ie-chemical-patent-project is working directory
# run following command to test out data loading
# python src/preprocessing/test.py

import preprocessor

tag2i_train, i2tag_train, tag2i_val, i2tag_val, train_dataloader, val_dataloader = preprocessor.load_data()

for batch_sents, batch_masks, batch_labels in train_dataloader:
    print(f'\nbatch sents shape: {batch_sents.shape}')
    print(f'batch masks shape: {batch_masks.shape}')
    print(f'batch labels shape: {batch_labels.shape}\n')