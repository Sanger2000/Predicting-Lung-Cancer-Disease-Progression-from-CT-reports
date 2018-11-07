import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import project.data.preprocess_data as preprocess
import torch
from sklearn.preprocessing import LabelEncoder

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
    baseline_reports, progress_reports, _ = reports
    baseline_bow = np.array(learn_bow(baseline_reports['clean_report_text'], max_features=max_base_feats).todense())
    progress_bow = np.array(learn_bow(progress_reports['clean_report_text'], max_features=max_prog_feats).todense())
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

    return np.array(train_features[False]), np.array(train_features[True]),  prepare_y(train_labels), id_list


def create_data(max_base, max_prog, max_before, max_after, desired_features):
    df = preprocess.load_reports()
    df_extraction = preprocess.extractFeatures(df)
    baseX, progX, labs, id_list = setupFeatureVectors(df_extraction, desired_features, max_before, max_after)
    df_text = createTextFeatures(preprocess.extractText(df, id_list), max_base, max_prog)

    return torch.from_numpy(baseX), torch.from_numpy(progX), torch.from_numpy(df_text), torch.from_numpy(labs)
