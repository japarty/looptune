import pandas as pd

# Dataset specific load to 2-col df ['text', label']

def load_predefined_dataset(data_path):
    if data_path.endswith('polish_pathos_translated.xlsx'):
        return load_polish_pathos_translated(data_path)
    elif data_path.endswith('PolarIs-Pathos.xlsx'):
        return load_PolarIs(data_path)


def load_PolarIs(path):
    df = pd.read_excel(path)
    df['label'] = df[['No_pathos', 'Positive', 'Negative']].idxmax(axis=1)
    df = df.rename(columns={'Sentence': 'text'})
    df = df[['text', 'label']]
    return df


def load_polish_pathos_translated(data_path):
    df = pd.read_excel(data_path)
    df['text'] = df['English']
    df['label'] = df['cleaned_pathos']
    df = df[['text', 'label']]
    return df