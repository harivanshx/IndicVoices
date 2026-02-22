import polars as pl
import sys
from joblib import Parallel, delayed
import tqdm
import os

def find_chunk(filepath, test_df, idx):
    #get the current idx
    base_path, chunk_ext = filepath.split('_chunk_')
    cur_idx, ext = chunk_ext.split('.')
    next_idx = int(cur_idx) + idx

    next_filepath = f'{base_path}_chunk_{next_idx}.{ext}'

    try:
        return full_df.filter(pl.col('audio_filepath') == next_filepath).row(0, named=True)
    except:
        return None

LANGS = ['Assamese','Bengali','Bodo','Dogri','Gujarati','Hindi','Kannada','Kashmiri','Konkani','Maithili','Malayalam','Manipuri','Marathi','Nepali','Odia','Punjabi','Sanskrit','Santali','Sindhi','Tamil','Telugu','Urdu']

if __name__ == "__main__":
    lang = sys.argv[1].capitalize()
    assert lang in LANGS, lang
    valid_path = f'/auto/ASR/shared_ai4b/asrteam/speechteam/speech-datasets/datasets/indicvoices/{lang}/valid_filtered.json'
    train_path = f'/auto/ASR/shared_ai4b/asrteam/speechteam/speech-datasets/datasets/indicvoices/{lang}/train_filtered.json'
    train_df = pl.read_ndjson(train_path, schema={'audio_filepath': pl.String})
    valid_df = pl.read_ndjson(valid_path, schema={'audio_filepath': pl.String})
    full_df = pl.read_ndjson([train_path, valid_path])

    _next = Parallel(n_jobs=-1, backend='threading')(delayed(find_chunk)(x, full_df, 1) for x in tqdm.tqdm(full_df['audio_filepath'].to_list()))
    _prev = Parallel(n_jobs=-1, backend='threading')(delayed(find_chunk)(x, full_df, -1) for x in tqdm.tqdm(full_df['audio_filepath'].to_list()))


    full_df = full_df.with_columns([pl.Series(name='next', values=_next, strict=False), pl.Series(name='prev', values=_prev, strict=False)])

    full_df.filter(pl.col('audio_filepath').is_in(train_df['audio_filepath'].to_list())).write_ndjson(f'/auto/ASR/shared_ai4b/asrteam/speechteam/speech-datasets/datasets/indicvoices/{lang}/train_filtered_with_prev_next.json')
    full_df.filter(pl.col('audio_filepath').is_in(valid_df['audio_filepath'].to_list())).write_ndjson(f'/auto/ASR/shared_ai4b/asrteam/speechteam/speech-datasets/datasets/indicvoices/{lang}/valid_filtered_with_prev_next.json')
