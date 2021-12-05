# if ie-chemical-patent-project is working directory
# run following command to test out data loading
# python src/preprocessing/test.py
import pdb
import preprocessor

tag2i_train, i2tag_train, tag2i_val, i2tag_val, train_dataloader, val_dataloader = preprocessor.load_data(ner_task=2)

for batch_sents, batch_masks, batch_labels in train_dataloader:
    print(f'\nbatch sents shape: {batch_sents.shape}')
    print(f'batch masks shape: {batch_masks.shape}')
    print(f'batch labels shape: {batch_labels.shape}\n')

# import ee_preprocessor
#
# train_dataloader = ee_preprocessor.load_data_ee()
#
# # for batch_sents, batch_masks, batch_labels, batch_trig_mask, batch_ent_mask in train_dataloader:
# #     print(f'\nbatch sents shape: {batch_sents.shape}')
# #     print(f'batch masks shape: {batch_masks.shape}')
# #     print(f'batch labels shape: {batch_labels.shape}\n')
# #     print(f'batch labels shape: {len(batch_trig_mask)}\n')
# #     print(f'batch labels shape: {len(batch_ent_mask)}\n')
