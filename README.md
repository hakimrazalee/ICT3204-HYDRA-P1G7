# ICT3204 P1G7 HYDRA Tool

## The repository contains the models used for the coursework 2 submission.
Metrics Comparison:
SVM Prediction Results:
- Precision: 0.278
- Recall: 1
- F1 Score: 0.435

IFO Prediction Results:
- Precision: 0.253
- Recall: 0.907
- F1 Score: 0.396

NER Prediction Results:
- Precision: 0.982
- Recall: 0.948
- F1 Score: 0.964

**The second model used for comparison is located in the PyCaret folder.**

How to run: 
git clone https://github.com/hakimrazalee/ICT3204-HYDRA-P1G7.git

install the following dependencies:
```
pip install spacy
pip install allennlp
pip install allennlp-models
pip install Dash
pip install dash-cytoscape
pip install dash-html-components
pip install dash-bootstrap-components
```

Function descriptions:
```python3
train(configpath, serialpath) # To train the model
logtosentence(inputpath, outputpath) # Data Preprocessing to ingest in model
predict(resultpath, modelpath, outputpath) # Predict if there are anomalies in the logs
printResults(resultpath, wordspath, tagspath, conll_result) # Print the results of the anomalies
convert(conll_result, finalpath) # Converts the conll_result into a readable format
kgraph(conll_result, inputpath) # Creates a dashboard with a knowledge graph
```
