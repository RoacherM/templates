from sklearn.metrics import classification_report


def evaluate(trues, preds):
    report = classification_report(trues, preds, output_dict=True, zero_division=1)
    return report