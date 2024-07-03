# class LooptuneDataset(Dataset):
#     def split(self):
#         dataset = dataset.train_test_split(train_size=train_size, seed=42, stratify_by_column='label')

#         if val_size is not None:
#             val_ratio = 1 - (val_size/(1 - train_size))
#             dataset2 = dataset['test'].train_test_split(train_size=val_ratio, seed=42)

#             dataset['test'] = dataset2['train']
#             dataset['val'] = dataset2['test']

#         return dataset


# def df_to_ds(df):
#     print(df.groupby('label').count())
#     ds = LooptuneDataset.from_pandas(df)
#     ds = ds.class_encode_column('label')
#     ds.target_map = {i: ds.features["label"].str2int(i) for i in ds.features["label"].names}
#     return ds, target_map


# # class LooptuneDatasetDict(DatasetDict):
# #     def split()

# #     def balance_dataset(self, col_from='train', col_to=False):