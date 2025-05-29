# CSE 599G (Deep Learning) Course Project

By Alysa Meng

## Data processing

- [X] Load in DISCO data (text, authorial metadata)
- [X] Extract n-gram features with relevant libraries
- [X] Extract rhyme schema with relevant libraries
- [X] Split into train, val, test

## SVM Classification

- [X] Build (binary) classifier for gender
- [X] Build multiclass classifier for country
- [X] Build multiclass classifier for time period

## Bi-LSTM

- [X] Decide on pre-trained word embeddings for Spanish
- [X] Build classifier for gender
- [X] Build classifier for country
- [X] Build classifier for time period

## Before BERT

- [X] Consider alternatives to stratified splits for SVM/Bi-LSTM (e.g., weighting or downsampling)

## BETO (BERT architecture)

- [ ] Finetune BETO on gender classification
- [ ] Finetune BETO on country classification
- [ ] Finetune BETO on time period classification

## Ensemble models (if time, probably won't have time)

- [ ] Combine SVM with finetuned BETO
- [ ] Combine Bi-LSTM with finetuned BETO