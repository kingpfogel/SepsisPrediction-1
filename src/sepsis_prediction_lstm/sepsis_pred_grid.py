import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim

# import sys
# sys.path.append("C:/Users/Kai/PycharmProjects/MedInf/SepsisPrediction-1-master/src/sepsis_prediction_lstm")

from utils_gs import train, evaluate, best_evaluate
from mydatasets import calculate_num_features, VisitSequenceWithLabelDataset, time_collate_fn
from mymodels import MyLSTM


def run(out_dir, epochs, LR, emb_size, hidden_size, num_layers):
    torch.manual_seed(0)

    PATH_TRAIN_SEQS = "SepsisPrediction-1-master/data/sepsis/processed_data_12_6_test/sepsis.seqs.train"
    PATH_TRAIN_LABELS = "SepsisPrediction-1-master/data/sepsis/processed_data_12_6_test/sepsis.labels.train"
    PATH_VALID_SEQS = "SepsisPrediction-1-master/data/sepsis/processed_data_12_6_test/sepsis.seqs.validation"
    PATH_VALID_LABELS = "SepsisPrediction-1-master/data/sepsis/processed_data_12_6_test/sepsis.labels.validation"
    PATH_TEST_SEQS = "SepsisPrediction-1-master/data/sepsis/processed_data_12_6_test/sepsis.seqs.test"
    PATH_TEST_LABELS = "SepsisPrediction-1-master/data/sepsis/processed_data_12_6_test/sepsis.labels.test"
    PATH_OUTPUT = out_dir
    os.makedirs(PATH_OUTPUT, exist_ok=True)

    NUM_EPOCHS = epochs
    BATCH_SIZE = 64
    NUM_WORKERS = 0

    # Data loading
    # print('===> Loading entire datasets')
    train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
    train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
    valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
    valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
    test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
    test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

    num_features = calculate_num_features(train_seqs)

    train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels)
    valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels)
    test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels)

    count_positive = 0
    count_negative = 0
    for data, label in train_dataset:
        if label == 1:
            count_positive += 1
        else:
            count_negative += 1

    weights = [count_positive if label == 1 else count_negative for data, label in train_dataset]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, collate_fn=time_collate_fn, sampler=sampler, num_workers=NUM_WORKERS)

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=time_collate_fn,
                              num_workers=NUM_WORKERS)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=time_collate_fn,
                             num_workers=NUM_WORKERS)

    model = MyLSTM(num_features, emb_size, hidden_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    device = torch.device("cpu")
    model.to(device)
    criterion.to(device)

    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

        is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
        if is_best:
            best_val_acc = valid_accuracy
            torch.save(model, os.path.join(PATH_OUTPUT, "MyLSTM.pth"))

    best_model = torch.load(os.path.join(PATH_OUTPUT, "MyLSTM.pth"))
    # print("\nEvaluation metrics on train set: \t")
    # acc, prec, rec, f1, auc, mcc = best_evaluate(best_model, device, train_loader)

    # print("\nEvaluation metrics on validation set: \t")
    # acc, prec, rec, f1, auc, mcc = best_evaluate(best_model, device, valid_loader)

    # print("\nEvaluation metrics on test set: \t")
    acc, prec, rec, f1, auc, mcc = best_evaluate(best_model, device, test_loader)

    print("LR: " + str(LR) + "\t" + "Epochs: " + str(NUM_EPOCHS) + "\t" +
          "EMB: " + str(emb_size) + "\t" + "HID: " + str(hidden_size) + "\t" +
          "LAY: " + str(num_layers) + "\t" + "AUC: " + str(auc) + "\t" + str(os.path.join(PATH_OUTPUT, "MyLSTM.pth")))
