# Profiling Poets from Spanish Sonnets

This is the code repository for a project done individually in the University of Washington's CSE 599G (Deep Learning). A Python 3.11 environment was used.

See `writeup.pdf` for more details.

The data folder is empty since its contents were identical to DISCO with the following structure:

```
data
    disco_files
        tei
        txt
    author_metadata.tsv
    poem_metadata.tsv
```

## Checklist

### Data processing

- [X] Load in DISCO data (text, authorial metadata)
- [X] Extract n-gram features with relevant libraries
- [X] Extract rhyme schema with relevant libraries
- [X] Split into train, val, test

### SVM Classification

- [X] Build (binary) classifier for gender
- [X] Build multiclass classifier for country
- [X] Build multiclass classifier for time period

### Bi-LSTM

- [X] Decide on pre-trained word embeddings for Spanish
- [X] Build classifier for gender
- [X] Build classifier for country
- [X] Build classifier for time period

### Before BERT

- [X] Consider alternatives to stratified splits for SVM/Bi-LSTM (e.g., weighting or downsampling)

### BETO (BERT architecture), trained on Google Colab

- [X] Finetune BETO on gender classification
- [X] Finetune BETO on country classification
- [X] Finetune BETO on time period classification

### Refactor code

- [ ] Remove logical repetition between classifications tasks (data processing, plots, etc)
- [ ] Rewrite/clean up local notebooks with the code used for the final version and analysis steps shown in the paper (and upload here)
- [ ] Remove unused fonemas, libEscansion packages (were used for experimenting with unlabeled poetry at some point but were very inefficient)

### Future work: Ensemble models

- [ ] Combine SVM with finetuned BETO
- [ ] Combine Bi-LSTM with finetuned BETO