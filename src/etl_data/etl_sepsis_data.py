import os
import pickle
import pandas as pd
from datetime import datetime
from datetime import timedelta

PATH_TRAIN = r"C:\Users\Kai\PycharmProjects\MedInf\Train.csv"
PATH_VALIDATION = r"C:\Users\Kai\PycharmProjects\MedInf\Test.csv"
PATH_TEST = r"C:\Users\Kai\PycharmProjects\MedInf\Validation.csv"
PATH_OUTPUT = r"C:\Users\Kai\PycharmProjects\MedInf\SepsisPrediction-1-master\data\sepsis\processed_data_25_6"

def create_dataset(path, observation_window=25, prediction_window=6):
    """
    :param path: path to the directory contains raw files.
    :param observation window: time interval we will use to identify relavant events
    :param prediction window: a fixed time interval that is to be used to make the prediction
    :return: List(pivot vital records), List(labels), time sequence data as a List of List of List.
    """
    PATH_OUTPUT = r"C:\Users\Kai\PycharmProjects\MedInf\SepsisPrediction-1-master\data\sepsis\processed_data_"+str(observation_window)+"_"+str(prediction_window)
    seqs = []
    labels = []
    # load data from csv;
    df = pd.read_csv(path, sep=";")

    # construct features
    grouped_df = df.groupby('hadm_id')
    for name, group in grouped_df: 
        # calculate the index_hour
        # for the patients who have the sepsis, index hour is #prediction_window hours prior to the onset time
        # for the patients who don't have the sepsis, index hour is the last event time
        bla = group.iloc[-1, -1]
        bla2 = group.iloc[-1, 1]
        if group.iloc[-1,-1] == 1:
            index_hour = datetime.strptime(group.iloc[-1,1], '%Y-%m-%d %H:%M:%S') - timedelta(hours=prediction_window)
        else:
            index_hour = datetime.strptime(group.iloc[-1,1], '%Y-%m-%d %H:%M:%S')

        # get the date in observation window
        group['admittime'] = pd.to_datetime(group['admittime'])
        filterd_group = group[(group['admittime'] >= (index_hour - timedelta(hours=observation_window))) & (group['admittime'] <= index_hour)]
        # construct the records seqs and label seqs
        data = filterd_group.iloc[:, 2:-1]
        record_seqs = []
        for i in range(0, data.shape[0], 1):
            record_seqs.append(data.iloc[i].tolist())

        if len(record_seqs) != 0:
            seqs.append(record_seqs)
            labels.append(group.iloc[-1, -1])

    return seqs, labels


def main():
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    # Train set
    print("Construct train set")
    train_seqs, train_labels = create_dataset(PATH_TRAIN)

    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "sepsis.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "sepsis.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Validation set
    print("Construct validation set")
    train_seqs, train_labels = create_dataset(PATH_VALIDATION)

    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "sepsis.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "sepsis.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Test set
    print("Construct test set")
    train_seqs, train_labels = create_dataset(PATH_TEST)

    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "sepsis.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "sepsis.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

    print("Complete!")


if __name__ == '__main__':
    main()
