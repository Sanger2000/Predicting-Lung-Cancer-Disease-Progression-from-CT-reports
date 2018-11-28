import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import project.data.preprocess_data as preprocess
import torch
from sklearn.preprocessing import LabelEncoder
from project.data import tokenization

def tokenize_input(baseline_text, context_text, split, tokenizer=tokenization.FullTokenizer("cased_bert/vocab.txt"), max_len=509):
    baseline = tokenizer.tokenize(baseline_text)
    context = tokenizer.tokenize(context_text)
    print(len(baseline))
    print(len(context))
    baseline_size = int(split*max_len)
    context_size = max_len - baseline_size

    baseline = preprocess.preprocess_tokens(baseline, baseline_size)
    context = preprocess.preprocess_tokens(context, context_size)

    final_tokens = ["[CLS]"]
    classifications = [0, 0]

    for token in baseline:
        final_tokens.append(token)
        classifications.append(0)

    final_tokens.append("[SEP]")
    classifications.append(1)

    for token in context:
        final_tokens.append(token)
        classifications.append(1)

    final_tokens.append("[SEP]")
    for i in range(max_len-(len(context) + len(baseline))):
        final_tokens.append("[MASK]")
        classifications.append(0)
    return tokenizer.convert_tokens_to_ids(final_tokens), classifications


def one_hot_encode(labels):
    out = np.zeros((labels.shape[0], 4))
    out[np.arange(labels.shape[0]), labels] = 1
    return out

def learn_bow(reports, min_df=1, ngram_range=(1, 3), max_features=5000):
    stopwords = ['mm', 'dd', '2017', '2016', '2015', '2014', '2013', '2012', 'date', 'md']
    countVec = CountVectorizer(min_df = min_df, \
                               ngram_range = ngram_range, \
                               max_features = max_features, \
                               stop_words = stopwords)
    countVec.fit(reports)
    return countVec.transform(reports)

def prepare_y(data_y):
    label_enc = LabelEncoder()
    label_enc_y = label_enc.fit(data_y)

    return label_enc_y.transform(data_y)

def createTextFeatures(reports, max_base_feats, max_prog_feats):
    baseline_text, progress_text, _, __ = reports
    #print(baseline_reports)
    #print(baseline_reports['clean_report_text'])
    #print(type(baseline_reports['clean_report_text']))
    #print(baseline_reports['clean_report_text'].tolist())
    baseline_bow = np.array(learn_bow(baseline_text['clean_report_text'].tolist(), max_features=max_base_feats).todense())
    progress_bow = np.array(learn_bow(progress_text['clean_report_text'].tolist(), max_features=max_prog_feats).todense())
    print(baseline_bow.shape)
    print(progress_bow.shape)
    overallTextFeatures = np.hstack([baseline_bow, progress_bow])
    return overallTextFeatures

def make_id(patient_id):
    if patient_id < 10:
        return "MSK_00" + str(patient_id)
    elif patient_id < 100:
        return "MSK_0" + str(patient_id)
    else:
        return "MSK_" + str(patient_id)

def pad_vectors(feats, max_len, feat_lens):
    for val in (True, False):
        for i in range(len(feats[val])):
            for j in range(max_len):
                if j >= len(feats[val][i]):
                    feats[val][i].append(np.zeros(feat_lens))
    return feats

def setupFeatureVectors(df, desired_features, max_before, max_after):
    FEAT_LENS = len(desired_features) + max_before + max_after

    patients = df.groupby("Patient ID")
    max_len = 0

    train_feats = {True: [], False: []}

    train_labels = []
    id_list = set()

    count = -1

    before_text = np.array(learn_bow(df["before_text"], max_features = max_before).todense())
    after_text = np.array(learn_bow(df["after_text"], max_features = max_after).todense())

    train_features = {True: [], False: []}

    for patient_id in sorted([int(key[-3:]) for key in patients.groups.keys()]):
        count += 1
        patient = make_id(patient_id)
        context = {True: [], False: []}


        checker = {True: False, False: False}
        len_counter = {True: 0, False: 0}

        count2 = -1
        for i in patients.groups[patient]:
            count2 += 1

            checker[df["is_baseline"][i]] = True
            len_counter[df["is_baseline"][i]] += 1

            context[df["is_baseline"][i]].append(np.concatenate((np.array([df[desired_feat][i] for desired_feat in desired_features]), \
                                                            before_text[i], after_text[i])))


        if not(checker[True] or checker[False]):
            continue
        elif not checker[True]:
            context[True].append(np.zeros(FEAT_LENS))
        elif not checker[False]:
            context[False].append(np.zeros(FEAT_LENS))


        max_len = max(max_len, len_counter[True], len_counter[False])

        id_list.add(patient)
        for val in (True, False):
            train_features[val].append(context[val])

        train_labels.append(df["labels"][i])

    train_features  = pad_vectors(train_features, max_len, FEAT_LENS)

    return np.array(train_features[False]), np.array(train_features[True]),  one_hot_encode(prepare_y(train_labels)), id_list


def create_data(max_base, max_prog, max_before, max_after, desired_features):
    df = preprocess.load_reports()
    df_extraction = preprocess.extractFeatures(df)
    baseX, progX, labs, id_list = setupFeatureVectors(df_extraction, desired_features, max_before, max_after)
    reports = preprocess.extractText(df, id_list)
    df_text = createTextFeatures(reports, max_base, max_prog)
    id_vals = torch.tensor(list(map(lambda x: tokenize_input(x[0], x[1], split=0.4), zip(reports[2]['bert_text'], \
                                                                                    reports[3]['bert_text']))))
    id_vals.resize_((2, id_vals.size(0)))

    return torch.from_numpy(baseX), torch.from_numpy(progX), torch.from_numpy(df_text), torch.from_numpy(labs), id_vals[0], id_vals[1]
