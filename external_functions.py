import csv
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import spacy
from random import shuffle
import re

nlp = spacy.load("en_core_web_lg")
pattern1 = re.compile(r'[\.!]{2,}') # Används för att ersätta bindesstreck samt två eller fler punkter/utropstecken med ett mellanslag, tex: "This was a good movie..I liked it a lot"
pattern2 = re.compile(r'(?<![A-Z])[\.!]') # Matchar punkttecken (eller utropstecken) i "This was a good movie.I liked it a lot" men INTE i "I like the U.S"
pattern3 = re.compile(r'[^A-Za-z ]') # Matchar de tecken som ska tas bort (alla tecken utom A-Z, a-z och mellanslag)
pattern4 = re.compile(r' {2,}') # Används för att ersätta två mellanslag (eller fler) med ett mellanslag



def custom_round(x):
    if x > 5:
        return 5
    elif x < 1:
        return 1
    else:
        return round(x)
    


def dict_to_csv(dictionary, file_path):
    # Write the dictionary to a CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
        writer.writeheader()
        for row in zip(*dictionary.values()):
            writer.writerow(dict(zip(dictionary.keys(), row)))



def make_prediction(model, vectorizer, indicies, review, round_result = True):
    cleaned_input = np.array([lemmatize_text(clean_text(review))])
    vectorized_input = vectorizer.transform(cleaned_input)[:,indicies]
    result = model.predict(vectorized_input.reshape(1, -1).toarray())
    if round_result:
        return custom_round(result[0,0])
    else:
        return result[0,0]



def ndarray_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for row in file:
            data.append(row.rstrip().split(","))
    return np.array(data, dtype='O') # dtype='O' är nödvänig med hänsyn till minnesåtgång



def plot_results(dictionary, metrics, skip_num = 0):
    """Plottar figurer sida vid sida för de metrics som anges i listan metrics"""
    plt.figure(figsize=(18, 6))

    for idx, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), idx+1)
        plt.plot(list(range(1, len(dictionary[metric]) + 1)), dictionary[metric][skip_num:])
        plt.plot(list(range(1, len(dictionary[metric]) + 1)), dictionary['val_'+metric][skip_num:])
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.title(f'Training and Validation {metric.capitalize()}', fontweight='bold')
        plt.legend([metric, 'val_'+metric])
        plt.xlim(skip_num + 1, len(dictionary[metric]))
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



def print_confusion_matrix(y_true, y_pred, num_classes):
    matrix = confusion_matrix(y_pred, y_true)
    fig, ax = plt.subplots()
    labels = [str(x+1) for x in range(num_classes)]
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_xlabel("True Class")
    ax.set_ylabel("Predicted Class")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    plt.show()



def clean_text(text):
    text = pattern1.sub(" ", text)
    text = pattern2.sub(" ", text)
    text = pattern3.sub("", text)
    text = pattern4.sub(" ", text)
    return text.strip().lower() # .strip() är nödvändigt eftersom pattern2 kan ge ett inledande mellanslag



def lemmatize_text(text):
    """Lemmatiserar, tar bort stoppord och ord med längd 1 och 2"""
    doc = nlp(text)
    lemmatized_list = [token.lemma_ for token in doc if not token.is_stop]
    lemmatized_text = " ".join(word for word in lemmatized_list if len(word) > 2)
    return lemmatized_text



def lemmatize_texts_parallel(texts): # Används ej i slutlig implementation
    """Lemmatiserar, tar bort stoppord och ord med längd 1 och 2"""
    if type(texts[0]) != str:
        texts = [str(text) for text in texts] # Nödvändig eftersom "strängar" i ndarrays är av datatypen numpy.str_
    n_processes = multiprocessing.cpu_count()
    docs = list(nlp.pipe(texts, n_process=n_processes, disable=["parser", "ner"])) # Processerar texten parallelt med Spacy
    lemmatized_texts = np.array([" ".join([token.lemma_ for token in doc if not token.is_stop and len(token.lemma_) > 2]) for doc in docs])
    return lemmatized_texts



def shuffle_words(text):
    word_list = text.split(" ")
    shuffle(word_list)
    return " ".join(word_list)