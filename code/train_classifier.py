import logging
from sklearn.svm import SVC
logger = logging.getLogger(__name__)
try:
    import _pickle as pickle
    import gc
    print("using cPickle")
except:
    import pickle
    print("using pickle")


def _train_and_save_classifier(emb_array, label_array, class_names, classifier_filename_exp):
    logger.info('Training Classifier at')
    model = SVC(kernel='linear', probability=True, verbose=False)
    model.fit(emb_array, label_array)

    with open(classifier_filename_exp, 'wb') as outfile:
        gc.disable()
        pickle.dump((model, class_names), outfile)
        gc.enable()
    logging.info('Saved classifier model to file "%s"' % classifier_filename_exp)
