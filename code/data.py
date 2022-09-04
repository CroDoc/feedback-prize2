import codecs

import pandas as pd
from sklearn.model_selection import KFold
from text_unidecode import unidecode


def replace_encoding_with_utf8(error):
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error):
    return error.object[error.start : error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text):
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

score_dict = {'Ineffective': 0, 'Adequate': 1, 'Effective': 2}

def clean_text(text, strip=True):
    text = text.replace(u'\xa0', u' ')
    text = text.replace(u'\x85', u'\n')
    text = text.strip()
    text = resolve_encodings_and_normalize(text)

    return text

def get_train():
    train_df =  pd.read_csv('data/train.csv')
    train_df['discourse_text'] = train_df['discourse_text'].map(clean_text)

    essay_ids = set(train_df['essay_id'])
    essay_texts = {}
    essay_labels = {}

    for essay_id in essay_ids:
        with open('data/train/' + essay_id + '.txt', 'r') as f:
            text = clean_text(f.read(), strip=False)
        essay_texts[essay_id] = text
        essay_labels[essay_id] = []

    for _, row in train_df.iterrows():
        essay_labels[row['essay_id']].append((row['discourse_text'], row['discourse_type'], score_dict[row['discourse_effectiveness']]))

    data = []

    for essay_id in sorted(essay_ids):
        labels = essay_labels[essay_id]
        data.append((essay_id, labels))

    train_df = pd.DataFrame(data, columns=['essay_id', 'labels'])

    return train_df
