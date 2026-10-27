# Sentiment analysis with BERT

Classifying airline tweets as complaints or non-complaints, and comparing a fine-tuned BERT model with a TF-IDF + Naive Bayes baseline.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumankrsh/Sentiment-Analysis.ipynb/blob/main/BERT_for_Sentiment_Analysis.ipynb)

## Result

On the validation set (340 tweets held out from training):

| Model | Accuracy | AUC |
|---|---|---|
| TF-IDF + Naive Bayes (baseline) | 75.6% | 0.85 |
| Fine-tuned BERT (`bert-base-uncased`) | 81.5% | 0.90 |

Fine-tuned BERT is about 6 percentage points more accurate than the baseline, trained on only 3,400 labelled tweets.

## Approach

**Data.** 1,700 complaint and 1,700 non-complaint tweets that mention US airlines, split 90/10 into training and validation sets. A separate file of 4,555 unlabelled tweets is used for a final prediction pass.

**Baseline.** Lower-casing, punctuation and stop-word removal, TF-IDF on 1- to 3-word n-grams, then a multinomial Naive Bayes classifier.

**BERT.** `bert-base-uncased` from Hugging Face `transformers`, with a small classifier (one hidden layer of 50 units) on top of the `[CLS]` output. Tweets have @mentions removed and are capped at 64 tokens. The whole model is fine-tuned for 2 epochs with AdamW (learning rate 5e-5, batch size 32) on a Colab GPU.

After validation, BERT is retrained on all 3,400 tweets and used to flag non-complaint tweets in the unlabelled set.

## How to run

The notebook keeps its outputs, so you can read the full run on GitHub without running anything.

To run it yourself, open it in Colab with the badge above, switch to a GPU runtime (**Runtime → Change runtime type → GPU**) and run the cells in order. The notebook downloads the data itself.

One caveat: the notebook pins `transformers==2.8.0` (2020), which no longer installs on current Colab runtimes. To run it today, change that install cell to a recent `transformers` release and update any imports that have since moved.

To run locally instead, put the three CSV files from this repo in a `data/` folder next to the notebook and skip the download cell. A CUDA GPU is strongly recommended.

## Files

| File | What it is |
|---|---|
| `BERT_for_Sentiment_Analysis.ipynb` | The notebook |
| `Sentiment_Analysis.ipynb` | The same notebook re-saved from Colab (whitespace differences only) |
| `complaint1700.csv`, `noncomplaint1700.csv` | Labelled training tweets |
| `test_data.csv` | Unlabelled tweets for the final prediction pass |

## Credit

The notebook follows the Skim AI tutorial [Fine-tuning BERT for Sentiment Analysis](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/) and uses the same dataset. Some of the notebook's explanatory text comes from that tutorial. That text quotes 72.65% and 82.65% accuracy; the table above uses the outputs printed in this notebook's own run (75.59% and 81.47%).
