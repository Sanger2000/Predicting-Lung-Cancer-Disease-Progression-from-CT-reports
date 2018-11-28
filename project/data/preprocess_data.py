from datetime import datetime
import re
import urllib
import zipfile
import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

EMBEDDING_PATH = "embeddings/GloveEmbeddings."


def preprocess_tokens(tokens, token_length):
    if len(tokens) > token_length:
        tokens = tokens[:token_length]
    return tokens

def bert_text_cleaner(text):
    start = re.search("FINDINGS?:", text)
    if start == None:
        return ""
    end = re.search("IMPRESSIONS?:", text)
    if end == None:
        return text[start.end():]

    return text[start.end():end.start()]

def download_embeddings():
    urllib.urlretrieve('http://nlp.stanford.edu/data/glove.42B.300d.zip', EMBEDDING_PATH + "zip")
    zip_ref = zipfile.ZipFile(EMBEDDING_PATH + 'zip', 'r')
    zip_ref.extractall(EMEDDING_PATH + 'txt')
    zip_ref.close()
    os.remove(EMBEDDING_PATH + "zip")

def save_embeddings():
    embeddings_index = {}
    f = open(EMBEDDING_PATH + "txt")
    count = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        count += 1
        f.close()
        pickle.dump(embeddings_index,  EMBEDDING_PATH + "pkl")
        os.remove(EMBEDDING_PATH + "txt")

def make_POD(curr):
    if curr == "POD/brain":
        return "POD"
    return curr

def load_reports():
    df_train = pd.read_csv('reports/urop_dataset_training.csv')
    df_train = df_train[df_train["Scan included on RECIST form? (y/n)"] == "yes"]
    df_train["Objective Response per RECIST v1.1"] = df_train["Objective Response per RECIST v1.1"].apply(lambda x: make_POD(x.strip()))
    df_train["clean_report_text"] = df_train["Scan report text"].apply(lambda text: re.sub('\W+', ' ', text).lower().strip() + str(' '))
    df_train["bert_text"] = df_train["Scan report text"].apply(bert_text_cleaner)

    return df_train


def findGroups(row):
    '''
    Input text from Scan Text Report
    Output: A dictionary for each organ of the text involved in it
    '''

    text = row["Scan report text"]
    out_dict = {"before_text":[], "after_text":[], "organs":[], "lens":[], "date_dist":[], "timepoint":[], "Patient ID":[], "labels": []}

    first, end = re.search("FINDINGS?:?", text), re.search("IMPRESSIONS?:?", text)

    if first == None or end == None:
        return []

    text = text[first.end() + 1: end.start()]
    group = re.search("(([A-Z]\/?\s?)+):|$", text)
    while group.group(1) != None:
        text = text[group.end():]
        text = text[re.search('\w', text).start():]

        next_group = re.search('\s+(([A-Z]\/?\s?)+):|$', text)
        report = re.sub('\s+', ' ', text[:next_group.start()])
        report, lens = pruneOverall(report)

        if len(lens) != 0:
            count = 0
            context = extractContext(report, "aabbSIZEbbaa")
            while context != None:
                out_dict["before_text"].append(report[context[0]:context[2].start()])
                out_dict["after_text"].append(report[context[2].end():context[1]])
                out_dict["organs"].append(group.group(1))
                out_dict["lens"].append(lens[count])
                out_dict["date_dist"].append(row["date_dist"])
                out_dict["timepoint"].append(row["timepoint"])
                out_dict["Patient ID"].append(row["Patient ID"])
                out_dict["labels"].append(row["Objective Response per RECIST v1.1"])

                count += 1
                report = report[context[2].end():]
                context = extractContext(report, "aabbSIZEbbaa")

        group = next_group
    return out_dict

def pruneVols(text, search_term='(\d\.?\d?) ([CcMm][Mm])? ?x (\d\.?\d?) ([CcMm][Mm])', mult = {"cm": 10, "mm": 1}):
    '''
    Input text of a given organ description
    Output: extracts the smaller axis lengths from a given volume measurement and replaces it with generic aabbSIZEbbaa word
    '''
    vol_text = re.search(search_term, text)
    if vol_text == None:
        return (text, [])

    lens = []
    while vol_text != None:
        if vol_text.group(2) == None:
            options = (float(vol_text.group(1))*(mult[vol_text.group(4).lower()]), float(vol_text.group(3))*(mult[vol_text.group(4).lower()]))
            lens.append(sum(options)/2.0)
        else:
            options = (float(vol_text.group(1))*(mult[vol_text.group(2).lower()]), float(vol_text.group(3))*(mult[vol_text.group(4).lower()]))
            lens.append(sum(options)/2.0)
        text = re.sub(search_term, " aabbSIZEbbaa ", text, count=1)
        vol_text = re.search(search_term, text)
    return (re.sub('\s+', ' ', text), lens)

def pruneLens(text, search_term='(\d\.?\d?) ([CcMm][Mm])', mult = {"cm": 10, "mm": 1}):
    '''
    Input text of a given organ description
    Output: extracts the lengths and replaces it with generic aabbsizebbaa word
    '''
    len_text = re.search(search_term, text)
    if len_text == None:
        return (text, [])

    lens = []
    while len_text != None:
        lens.append(float(len_text.group(1))*mult[len_text.group(2).lower()])
        text = re.sub(search_term, " aabbSIZEbbaa ", text, count=1)
        len_text = re.search(search_term, text)
    return (re.sub('\s+', ' ', text), lens)

def pruneOverall(text):
    '''
    Input: dictionary of organ descriptions
    Output: returns dictionary of organ descriptions where each description has volume or length measurements replaced
            also returns dictionary of organ tumor minor axis measurements
    '''
    text, lens = pruneVols(text)
    text, lens2 = pruneLens(text)
    return (text, lens+lens2)

def extractContext(text, sub):
    first = re.search(sub, text)
    if first == None:
        return None
    start = first.start() - re.search('$|(\s\w+){4}|(\s\w+)* ?\.', text[first.start()::-1]).end()+1
    end = first.end() + re.search('$|(\s\w+){4}|(\s\w+)* ?\.', text[first.end():]).end()
    return (start, end, first)

def days_after_start(row):
    start_date = row["Treatment start date"]
    current_date = row["Date of scan"]
    return (datetime.strptime(current_date, '%m/%d/%y') - datetime.strptime(start_date, '%m/%d/%y')).days

def extractFeatures(df):
    df_2 = pd.DataFrame()
    df["date_dist"] = df[["Treatment start date", "Date of scan"]].apply(days_after_start, axis=1)
    df["date_dist"] = (df["date_dist"] - df["date_dist"].mean())/df["date_dist"].std()

    df["timepoint"] = df["Scan timepoint (baseline = prior to treatment start, ontx = during treatment or prior to progression if stopped treatment , progression = time of RECIST defined progression)"]

    for index, row in df.iterrows():
        if index == 0:
            df_2 = pd.DataFrame.from_dict(findGroups(row))
        else:
            df_2 = df_2.append(pd.DataFrame.from_dict(findGroups(row)), ignore_index=True)

    df_2['is_baseline'] = (df_2['timepoint'] == 'baseline')

    df_2 = df_2.append(pd.DataFrame.from_dict({"before_text": [""], "after_text": [""], "organs":[""], "Patient ID":[None]}), ignore_index=True)

    organ_le = LabelEncoder()
    df_2["organs"] = organ_le.fit_transform(df_2["organs"])

    df_2 = df_2[df_2["Patient ID"] != None]
    return df_2

def extractText(df_train, id_list):
    # group the reports by patient and baseline
    column_patient = 'Patient ID'
    column_baseline = "Scan timepoint (baseline = prior to treatment start, ontx = during treatment or prior to progression if stopped treatment , progression = time of RECIST defined progression)"
    df_train = df_train[df_train["Patient ID"].isin(id_list)]

    df_train['is_baseline'] = (df_train[column_baseline] == 'baseline')
    groupped_text = df_train.groupby([column_patient, 'is_baseline'])['clean_report_text'].apply(lambda x: x.sum())

    for i, v in groupped_text.iteritems():
        patient, baseline = i
        if (patient, not baseline) not in groupped_text:
            groupped_text[(patient, not baseline)] = 'none'

    # now create the different dataframes
    groupped_text = groupped_text.to_frame().reset_index()
    baseline_text = groupped_text[groupped_text['is_baseline'] == True]
    progress_text= groupped_text[groupped_text['is_baseline'] == False]

    groupped_bert = df_train.groupby([column_patient, 'is_baseline'])['bert_text'].apply(lambda x: x.sum())

    for i, v in groupped_bert.iteritems():
        patient, baseline = i
        if (patient, not baseline) not in groupped_bert:
            groupped_bert[(patient, not baseline)] = 'none'

    # now create the different dataframes
    groupped_bert = groupped_bert.to_frame().reset_index()
    baseline_bert = groupped_bert[groupped_bert['is_baseline'] == True]
    progress_bert= groupped_bert[groupped_bert['is_baseline'] == False]
    '''
    groupped_df_text = df_train.groupby([column_patient, 'is_baseline'])['clean_report_text'].apply(lambda x: x.sum())

    predictions = df_train.groupby(['Patient ID'])["Objective Response per RECIST v1.1"].first()

    # fill missing reports with nothing
    for i, v in groupped_df_text.iteritems():
        patient, baseline = i
        if (patient, not baseline) not in groupped_df_text:
            print(patient, baseline)
            groupped_df_text[(patient, not baseline)] = 'none'



    print(len(groupped_df_text))

    #print(groupped_df_text)

    # now create the different dataframes
    baseline_text = groupped_df_text[groupped_df_text['is_baseline'] == True]
    print(len(baseline_text))
    progress_text = groupped_df_text[groupped_df_text['is_baseline'] == False]
    print (len(progress_text))

    groupped_df_bert = df_train.groupby([column_patient, 'is_baseline'])['bert_text'].apply(lambda x: x.sum())
    for i, v in groupped_df_bert.iteritems():
        patient, baseline = i
        if (patient, not baseline) not in groupped_df_bert:
            print(patient, baseline)
            groupped_df_bert[(patient, not baseline)] = 'none'
    print(len(groupped_df_bert))
    baseline_bert = groupped_df_bert[groupped_df_bert['is_baseline'] == True]
    print(len(baseline_bert))
    progress_bert = groupped_df_bert[groupped_df_bert['is_baseline'] == False]
    print (len(progress_bert))

    return (baseline_text, progress_text, predictions)
    '''
    print(len(progress_text))
    print(len(baseline_text))
    print(len(progress_bert))
    print(len(baseline_bert))
    return baseline_text, progress_text, baseline_bert, progress_bert
