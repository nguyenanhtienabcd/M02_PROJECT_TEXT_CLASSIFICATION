from sklearn.preprocessing import LabelEncoder as le
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


# read data
FILE_PATH = '2cls_spam_text_cls.csv'
df = pd.read_csv(FILE_PATH)

# convert data to list
messages = df['Message'].tolist()
labels = df['Category'].tolist()


# create preprocess data functions
# --------------------------------
def lowercase(text):
    return text.lower()


def punctuation_removal(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def tokenize(text):
    return nltk.word_tokenize(text)


def remove_stopwords(tokens):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]


def stemming(tokens):
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

# --------------------------------


def preprocess_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = nltk.word_tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens


# creat a library
message = [preprocess_text(text) for text in messages]
label = labels


def create_dictionary(messages):
    dictionary = []
    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)
    return dictionary

# creat a feature for each message


def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features


if __name__ == "__main__":
    dictionary = create_dictionary(message)
    X = np.array([create_features(tokens, dictionary) for tokens in message])
    y = np.array(label)

    # split data into val, test, train data
    VAL_SIZE = 0.2
    TEST_SIZE = 0.125
    SEED = 0
    SHUFFLE = True

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        random_state=SEED,
        shuffle=SHUFFLE
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=SHUFFLE
    )

    # training data
    model = GaussianNB()
    print('start training...')
    model.fit(X_train, y_train)
    print('training done!')

    # check model
    y_val_predicted = model.predict(X_val)
    y_test_predicted = model.predict(X_test)
    val_accuracy = accuracy_score(y_val, y_val_predicted)
    test_accuracy = accuracy_score(y_test, y_test_predicted)
    print(f'Validation Accuracy: {val_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')

    # create predict function
    def predict_spam(text):
        tokens = preprocess_text(text)
        features = create_features(tokens, dictionary)
        prediction = model.predict([features])
        # chuyển đổi nhãn mã hóa thành nhãn gốc
        return prediction[0]

    text = 'tommorow, can we go to the park'
    print('predict result is: ', predict_spam(text))
