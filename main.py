import csv
import dash_cytoscape as cyto
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import os
import dash
import pandas as pd
import numpy as np
import json
import re
import spacy
import plotly.graph_objects as go
import networkx as nx
from matplotlib import pyplot as plt

spacy.load('en_core_web_sm')


def train(configpath, serialpath):
    string = "allennlp train " + configpath + " -s " + serialpath
    os.system(string)


def logtosentence(inputpath, outputpath):
    sentence = ""
    source = pd.read_csv(inputpath, on_bad_lines='skip', engine='python')

    for i, line in enumerate(source.values):
        s = line
        # check if source is not fully empty
        if s.any():
            # strip() formats the whitespaces at the start and end of the string
            # replace() finds \n in the middle of the string to be replaced
            for row in s:
                try:
                    if not np.isnan(row):
                        continue
                except:
                    if row:
                        row = row.strip()
                        row = row.replace('\n', " ")
                        row = row.replace('\r', "")
                        row = row.replace('\t', " ")
                        row = row.replace('\"', "'")
                        row = row.replace(',', " ")
                        # add the sanitised row into formatted sentence string
                        sentence = sentence + row + " "
            sentence = sentence + " // "

    with open(outputpath, 'w', encoding='utf-8') as f:
        data_set = {"sentence": sentence}
        json.dump(data_set, f, ensure_ascii=False)


def predict(outputpath, modelpath, inputpath):
    print("Predicting Log file: " + inputpath)
    string = "allennlp predict --output-file " + outputpath + " " + modelpath + " " + inputpath
    os.system(string)


def printResults(resultpath, wordspath, tagspath, conll_result):
    finalList = []
    f = open(resultpath, "r")
    text = f.read()
    words = re.compile("words(.*)$").search(text).group(1)
    newWords = re.sub('[:",\][}]', '', words)
    listWords = newWords.replace(' ', '\n')
    w = open(wordspath, "w")
    w.write(listWords[1:])
    w.close()

    tags = re.compile("tags(.*)$").search(text).group(1)
    newTags = re.sub('[:",[\]]', '', tags.split('words')[0])
    listTags = newTags.replace(' ', '\n')
    t = open(tagspath, "w")
    t.write(listTags[1:])
    t.close()

    newf = "-X- I-O "
    filepath = tagspath

    with open(filepath) as fp:
        lines = fp.read().splitlines()
    with open(filepath, "w") as fp:
        for line in lines:
            print(newf + line, file=fp)

    # Opening up the created text Files
    tagList = pd.read_csv(tagspath, sep=" ", header=None)
    wordList = pd.read_csv(wordspath, sep=" ", header=None, skip_blank_lines=False)

    # Extracting the words that are at first column, column 0
    wordList = wordList[0]

    # Insertion of words into the tagList at the first column, column 0
    tagList.insert(0, " ", wordList)

    # Insertion of the '-DOCSTART- -X- O' for .CONLL format
    # Adding it at the end
    tagList.loc[-1] = ['-DOCSTART-', '-X-', 'O', '']
    # Moving it to the top
    tagList.index = tagList.index + 1
    tagList = tagList.sort_index()

    # To break each row in a newline
    for list in tagList.values:
        # '// -X- I-O O' denotes the end of each row
        # Replace with an '!' to prevent model confusion
        if list[0] == "//":
            finalList.append("!")
        else:
            finalList.append(list)
            
    # Saving the new file as .conll format
    (pd.DataFrame(finalList)).to_csv(
    conll_result, header=None, index=None, sep=' ', mode='w')

    with open(outputpath, 'r') as file:
        filedata = file.read()
    new = filedata.replace('\n-X- I-O O', '\n')
    with open(outputpath, 'w') as file:
        file.write(new)


def convert(conll_result, final_output):
    sentence = ""
    substring = ""
    sentences_list = []
    anomalies_dict = {"Word": [], "Label": []}
    source = []

    with open(conll_result) as f:
        reader = csv.reader(f, delimiter=";")
        # skip the headers
        next(reader, None)
        for r in reader:
            source.append(r)
        # remove the empty bracket [] at the end for better formatting
        if not source[-1]:
            del source[-1]
        source.append("")

    for row in source:
        try:
            # split the string by whitespaces
            s = row[0].split()
            if s and len(s) > 3:
                # check if current word is labelled an anomaly
                if s[3] != 'O':
                    anomalies_dict["Word"].append(s[0].replace(',', ''))
                    anomalies_dict["Label"].append(s[3][2:])
                sentence = sentence + s[0] + " "
            elif len(s) == 1:
                # if len(s) == 1 means its the end of the line ('!')
                # check if no label means no anomalies were detected
                if not len(anomalies_dict["Label"]):
                    sentence = "No anomalies detected - " + sentence
                else:
                    substring = "!! Anomalies Detected >> "
                    for i, anomaly in enumerate(anomalies_dict["Label"]):
                        substring = substring + anomalies_dict["Word"][i] + " ("
                        substring = substring + anomalies_dict["Label"][i] + "), "
                    sentence = substring[:-2] + " - " + sentence
                sentences_list.append(sentence + "\n")
                # cleaning all used variables and dictionary to be reused
                sentence = ""
                substring = ""
                anomalies_dict["Label"].clear()
                anomalies_dict["Word"].clear()
        except:
            continue

    with open(final_output, 'w', encoding='utf-8') as f:
        f.write("\n".join(str(item) for item in sentences_list))


def kgraph(conll,log):
    f1 = open(conll, "r")
    order = ["Anomalies Detected"]
    anomalies = ["Anomalies Detected"]
    tag = ["-"]
    str_list = []
    for line in f1:
        if '!' not in line:
            if line.split()[2] != 'O':
                if line.split()[3] == "U-Anomaly":
                    if str_list:
                        if str_list[-1] not in anomalies:
                            anomalies.append(str_list[-1])
                        else:
                            tag.pop()
                            order.pop()
                    str_list.append(line.split()[0])
                    order.append(str(str_list.index(line.split()[0])))
                    tag.append(line.split()[3][2:])
    anomalies.append(str_list[-1])
    df = pd.read_csv(log)
    nodes = [
        {"data": {"id": id_no, "label": label, "NER-tag": ner_tag}}
        for id_no, label, ner_tag in zip(order, anomalies, tag)
    ]

    edges = [
        {"data": {"source": "Anomalies Detected", "target": target, "NER-tag": ner_tag}}
        for target, ner_tag in zip(order[1:], tag[1:])
    ]

    app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])

    app.layout = html.Div(id='parent', children=
    [

        html.H1(id='H1', children='H.Y.D.R.A', style={'textAlign': 'center',
                                                                              'marginTop': 40, 'marginBottom': 0}),
        html.H5(id='H3', children='High-Yielding-Data-Recon-Ai', style={'textAlign': 'center',
                                                      'marginTop': 5, 'marginBottom': 40}),

        cyto.Cytoscape(
            id="cytoscape-layout-3",
            layout={"name": "circle"},
            style={"width": "100%", "height": "500px"},
            elements=nodes + edges,
            stylesheet=[
                # Class selectors
                {
                    "selector": "node",
                    "style": {"label": "data(label)", "color":"white", "background-color": "yellow"},
                },
                {
                    "selector": '[ NER-tag = "Anomaly" ]',
                    "style": {"color":"white","background-color": "blue", "line-color": "red"},
                },
            ],
        ),
        dcc.Markdown('''
            ##### Log File Ingested:
            ''')
        ,
        dcc.Clipboard(id="table_copy", style={"fontSize": 20}),
        dash_table.DataTable(
            df.to_dict("records"),
            [{"name": i, "id": i} for i in df.columns],
            filter_action='native',
            fixed_rows={'headers': True},
            style_cell={
                'minWidth': 95, 'maxWidth': 95, 'width': 95
            },
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white'
            },
            style_data={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white'
            },
            id="table_cb",
            page_size=20

        )

    ]
                          )

    @app.callback(
        Output("table_copy", "content"),
        Input("table_copy", "n_clicks"),
        State("table_cb", "data"),
    )
    def custom_copy(_, data):
        dff = pd.DataFrame(data)
        # See options for .to_csv() or .to_excel() or .to_string() in the  pandas documentation
        return dff.to_csv(index=False)  # includes headers

    app.run_server(debug=True)


configpath = "ner75epoch.json"
wordspath = "./preprocessing/words.txt"
tagspath = "./preprocessing/tags.txt"
conll_result = "./predictions/predictions.txt"
resultpath = "./preprocessing/results.txt"
modelpath = "models/model.tar.gz"
inputpath = "./logs/input.csv"
serialpath = "./models"
outputpath = "./preprocessing/preparedlogs/input.txt"
finalpath = "final_output.txt"

train(configpath, serialpath)
logtosentence(inputpath, outputpath)
predict(resultpath, modelpath, outputpath)
printResults(resultpath, wordspath, tagspath, conll_result)
convert(conll_result, finalpath)
kgraph(conll_result, inputpath)

