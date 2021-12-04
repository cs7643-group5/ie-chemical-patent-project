from preprocessing import ee_preprocessor
import pickle


def store_data():

    sentences_train, tags_train, ee_sentences_train, ee_labels_train = ee_preprocessor.read_folder_ee('data/ee_train/')
    sentences_val, tags_val, ee_sentences_val, ee_labels_val = ee_preprocessor.read_folder_ee('data/ee_dev/')

    data_train = (ee_sentences_train, ee_labels_train)
    data_val = (ee_sentences_val, ee_labels_val)

    with open("data/ee_data.pickle", "wb") as f:
        pickle.dump((data_train, data_val), f)

    return None


store_data()


