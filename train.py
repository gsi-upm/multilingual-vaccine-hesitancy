import os
import sys
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.metrics import classification_report
import pendulum

def write_results(path, contents):
    with open(path, "a") as f:
        contents = "\n=====\n".join(contents)
        f.write(contents)

train_dataset = sys.argv[1]

test_datasets = (
    "data/eng/test.csv",
    "data/isi_tweet_en.csv",
    "data/isi_tweet_it.csv",
    "data/eu-jav.csv",
    "data/TwitterVax-it-test.csv",
)


from happytransformer import HappyTextClassification, TCTrainArgs

happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                   model_name="distilbert-base-uncased",
                                   num_labels=2)  # Don't forget to set num_labels! 
args = TCTrainArgs(num_train_epochs=3, fp16=True, batch_size=32)
happy_tc.train(train_dataset, args=args)

results = list()
results.append(f"Training data: {train_dataset}")

for test_dataset in test_datasets:
    preds = happy_tc.test(test_dataset)

    preds = pd.Series([pred.label for pred in preds])
    preds = preds.apply(lambda p: p.split("_")[-1]).astype(int)
    
    labels = pd.read_csv(test_dataset)["label"]
    cr = classification_report(labels, preds)
    result = f"{test_dataset}\n{cr}"
    results.append(result)

alpha = os.path.splitext(os.path.basename(train_dataset))[0]
write_results(f'results/{alpha}-{pendulum.now("Europe/Madrid").format("YYYY-MM-DD_HH-mm")}', results)