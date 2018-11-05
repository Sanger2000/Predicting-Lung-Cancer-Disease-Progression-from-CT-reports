import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import data.preprocess_data as preprocess
import torch

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

def createTextFeatures((baseline_reports, progress_report, _), max_base_feats, max_prog_feats):
    baseline_bow = np.array(learn_bow(baseline_reports['clean_report_text'], max_features=max_base_feats).todense())
    progress_bow = np.array(learn_bow(progress_reports['clean_report_text'], max_features=max_prog_feats).todense())
    overallTextFeatures = np.hstack([baseline_bow, progress_bow])
def make_id(num):
    if patient_id < 10:
        return "MSK_00" + str(patient_id)
    elif patient_id < 100:
        return "MSK_0" + str(patient_id)
    else:
        return "MSK_" + str(patient_id)


def setupFeatureVectors(df, desired_features, ):

    patients = df.groupby("Patient ID")
    max_len = 0

    context = {True: ([], []), False: ([], [])}

    train_labels = []
    id_list = set()

    count = -1

    before_text = learn_bow(df_train["before_text"], max_features = 600)
    after_text = learn_bow(df_train["after_text"], max_features = 300)

    for patient_id in sorted([int(key[-3:]) for key in patients.groups.keys()]):
        count += 1
        patient = make_id(patient_id)


        checker = {True: False, False: False}
        len_counter = {True: 0, False: 0}

        count2 = -1
        for i in patients.groups[patient]:
            count2 += 1

            checker[df["is_baseline"][i]] = True
            len_counter[df["is_baseline"][i]] += 1

            for j in range(len(desired_features)):
                context[df["is_baseline"][i]][0].append([count, count2, j])
                context[df["is_baseline"][i]][1].append(df[desired_features[j]][i])

            indices = torch.nonzero(before_text[i])
            for index in indices:
                context[df["is_baseline"][i]][0].append([count, count2, index + len(desired_features))
                context[df["is_baseline"][i]][1].append(before_text[i][index])

            indices = torch.nonzero(after_text[i])
            for index in indices:
                context[df["is_baseline"][i]][0].append([count, count2, index+len(desired_features)+len(before_text[i])])
                context[df["is_baseline"][i]][1].append(after_text[i][index])


        if not(checker[True] or checker[False]):
            continue

        max_len = max(max_len, len_counter[True], len_counter[False])
        id_list.add(patient)

        train_labels.append(df["labels"][i])

    baseline_train_features = torch.sparse.floatTensor(torch.LongTensor(context[True][0]).t().cuda(0), torch.floatTensor(context[True][1]).cuda(0), \
                torch.Size([len(id_list), max_len, len(desired_features)+600+300])).cuda(0)
    progress_train_features = torch.sparse.floatTensor(torch.LongTensor(context[False][0]).t().cuda(0), torch.floatTensor(context[False][1]).cuda(0), \
                torch.Size([len(id_list), max_len, len(desired_features)+600+300]))).cuda(0)

    return baseline_train_features, progress_train_features,  prepare_y(train_labels)

def create_data(max_base=400, max_prog=800, desired_features=("lens", "organs", "date_dist")):
    df = preprocess.load_reports()
    df_extraction = preprocess.extractFeatures(df)
    df_text = createTextFeatures(preprocess.extractText(df), max_base, max_prog)

    baseX, progX, labs = setupFeatureVectors(df, desired_features)

    return baseX, progX, df_text, labs
