import re

import nltk
nltk.download('punkt')

from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
from nltk.lm import MLE, WittenBellInterpolated
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter


# Training data file


train_data_files = [
    './training-data/banana-report/1.txt', 
    './training-data/banana-report/2.txt', 
    './training-data/banana-report/3.txt',
    './training-data/banana-report/4.txt',
    './training-data/banana-report/5.txt',
    './training-data/banana-report/6.txt',
    './training-data/banana-report/7.txt',
    './training-data/banana-report/8.txt',
    './training-data/banana-report/9.txt',
    './training-data/banana-report/10.txt',
]  # list of training data files
train_data_files = [
    './training-data/rose-poem/1.txt', 
    './training-data/rose-poem/2.txt', 
    './training-data/rose-poem/3.txt',
    './training-data/rose-poem/4.txt',
    './training-data/rose-poem/5.txt',
    './training-data/rose-poem/6.txt',
    './training-data/rose-poem/7.txt',
    './training-data/rose-poem/8.txt',
    './training-data/rose-poem/9.txt',
    './training-data/rose-poem/10.txt',
]  # list of training data files

train_texts = []
for train_data_file in train_data_files:
    with open(train_data_file) as f:
        train_text = f.read().lower()
        train_text = re.sub(r"\[.*\]|\{.*\}", "", train_text)
        train_text = re.sub(r'[^\w\s]', "", train_text)
        train_texts.append(train_text)
all_scores = []
for train_text in train_texts:
# set ngram number
    n = 4

    # pad the text and tokenize
    training_data = list(pad_sequence(word_tokenize(train_text), n, 
                                    pad_left=True, 
                                    left_pad_symbol="<s>"))

    # generate ngrams
    ngrams = list(everygrams(training_data, max_len=n))
    print("Number of ngrams:", len(ngrams))

    # build ngram language models
    model = WittenBellInterpolated(n)
    model.fit([ngrams], vocabulary_text=training_data)
    print(model.vocab)

    # testing data file
    test_data_file = "./testdata.txt"

    # Read testing data
    with open(test_data_file) as f:
        test_text = f.read().lower()
    test_text = re.sub(r'[^\w\s]', "", test_text)

    # Tokenize and pad the text
    testing_data = list(pad_sequence(word_tokenize(test_text), n, 
                                    pad_left=True,
                                    left_pad_symbol="<s>"))
    print("Length of test data:", len(testing_data))

    # assign scores
    # assign scores
    scores = []
    for i, item in enumerate(testing_data[n-1:]):
        s = model.score(item, testing_data[i:i+n-1])
        scores.append(s)

    all_scores.append(scores)



# all_scores = [ [1,2,3,4], [1,2,3,4] ]

def max_array(arrays):
    x = len(arrays[0])
    max_array = [max(values) for values in zip(*arrays)]
    return max_array

max_scores = max_array(all_scores)

scores_np = np.array(max_scores)


# set width and height
width = 8
height = np.ceil(len(testing_data)/width).astype("int32")
print("Width, Height:", width, ",", height)

# copy scores to rectangular blank array
a = np.zeros(width*height)
a[:len(scores_np)] = scores_np
diff = len(a) - len(scores_np)

# apply gaussian smoothing for aesthetics
a = gaussian_filter(a, sigma=1.0)

# reshape to fit rectangle
a = a.reshape(-1, width)

# format labels
labels = [" ".join(testing_data[i:i+width]) for i in range(n-1, len(testing_data), width)]
labels_individual = [x.split() for x in labels]
labels_individual[-1] += [""]*diff
labels = [f"{x:60.60}" for x in labels]

# create heatmap
fig = go.Figure(data=go.Heatmap(
                z=a, x0=0, dx=1,
                y=labels, zmin=0, zmax=1,
                customdata=labels_individual,
                hovertemplate='%{customdata} <br><b>Score:%{z:.3f}<extra></extra>',
                colorscale="burg"))
fig.update_layout({"height":height*28, "width":1000, "font":{"family":"Courier New"}})
fig['layout']['yaxis']['autorange'] = "reversed"
fig.show()