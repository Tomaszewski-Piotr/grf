import common as common
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import os
from common import log_verbose
import pathlib
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder

use_GPU = True   # use GPU when training
xgb_verbosity = 1 # verbosity level

# creates xgboost regression model for given x, y input
# the model is created using 80% of data and evaluated using remaining 20
# evaluation results are returned together with the model
def create_regression_model(x_all, y_all):

    # split into test and train
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all,
                                                        test_size=0.2)  # 80% training and 20% test
    #train the model
    if use_GPU:
        xgb_reg = MultiOutputRegressor(xgb.XGBRegressor(verbosity=xgb_verbosity, tree_method='gpu_hist', gpu_id=0))
    else:
        xgb_reg = MultiOutputRegressor(xgb.XGBRegressor(verbosity=xgb_verbosity))#, tree_method='gpu_hist', gpu_id=0))
    xgb_reg.fit(x_train, y_train)

    #test on the test set and calculate the metrics
    y_pred = xgb_reg.predict(x_test) # Predictions
    MSE = mse(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R_squared = r2_score(y_test, y_pred)

    log_verbose("\nRMSE: ", np.round(RMSE, 2))
    log_verbose("R-Squared: ", np.round(R_squared, 2))

    #prepare actual vs predicted table
    out_array = np.concatenate((y_test, y_pred), axis=1)
    out = pd.DataFrame(out_array)
    no_outputs = y_test.shape[1]

    #provide appropriate titles
    titles = [None] * no_outputs * 2
    for i in range(0, no_outputs):
        titles[i] = 'Actual'+str(i+1)
        titles[i+no_outputs] = 'Predicted'+str(i+1)
    out.columns = titles

    return RMSE, R_squared, out, xgb_reg

# creates classification model for the provided data set where
# classes are placed in "class" column
#return the model and evaluation results
def create_classification_model(classification_set):
    #separate features from classes
    all_x = classification_set.drop(columns=['class'])
    # create a dataframe with only the target column
    all_y = classification_set[['class']]

    #encode the classes (ordinal encoding)
    encoder = LabelEncoder()
    encoder.fit(all_y.values.ravel())
    # get ordinal encoding
    all_ordinal_y = encoder.transform(all_y.values.ravel())

    #split the dataset in train test 80/20
    x_train, x_test, y_train, y_test = train_test_split(all_x, all_ordinal_y,
                                                        test_size=0.2)  # 80% training and 20% test
    # train the model
    if use_GPU:
        xgb_classifier = xgb.XGBClassifier(verbosity=xgb_verbosity, use_label_encoder=False, tree_method='gpu_hist', gpu_id=0)
    else:
        xgb_classifier = xgb.XGBClassifier(verbosity=xgb_verbosity, use_label_encoder=False)  # , tree_method='gpu_hist', gpu_id=0))
    xgb_classifier.fit(x_train, y_train)

    # test on the test set
    y_pred = xgb_classifier.predict(x_test)  # Predictions
    accuracy = accuracy_score(y_test, y_pred)
    log_verbose(accuracy)
    # store test predictions and the confusion matrix
    y_test_names = encoder.inverse_transform(y_test)
    y_pred_names = encoder.inverse_transform(y_pred)
    cm = confusion_matrix(y_test_names, y_pred_names, labels=encoder.classes_)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    out = pd.DataFrame(columns=['Actual', 'Predicted'])
    out['Actual'] = y_test_names
    out['Actual_encoded'] = y_test
    out['Predicted'] = y_pred_names
    out['Predicted_encoded'] = y_pred
    return xgb_classifier, encoder, accuracy, cm_df, out

def process_classification_model(classification_set, name, writer):
    classification_set_df = pd.concat(classification_set, axis=0, ignore_index=True)
    xgb_class, encoder, accuracy, cm, out = create_classification_model(classification_set_df)
    dump(xgb_class, open(common.out_file(name+'_classifier' + common.model_suffix), "wb"))
    dump(encoder, open(common.out_file(name+'_encoder' + common.model_suffix), "wb"))
    out.to_excel(writer, sheet_name = name + '_classification')
    cm.to_excel(writer, sheet_name = name + '_confusion matrix')
    return accuracy

def create_models():
    i=0
    complete_classification_set = [] # to be used for classification training
    low_classification_set = []  # to be used for classification training
    high_classification_set = []  # to be used for classification training
    with pd.ExcelWriter(common.out_file('output.xlsx')) as writer:
        summary = pd.DataFrame(columns=['Name', 'RMSE', 'R_squared', 'Accuracy'])  # report
        for x_file_name in common.find_data_csv():
            basename = os.path.basename(x_file_name)[2:-len(common.csv_suffix)]
            base_dir = os.path.dirname(x_file_name)
            y_file_name = pathlib.Path(base_dir, 'y_' + basename + common.csv_suffix)
            log_verbose(' Retrieving data for: ' + basename)
            x_all = pd.read_csv(x_file_name, header=None)
            y_all = pd.read_csv(y_file_name, header=None)
            #create regression model for given class
            RMSE, R_squared, out, xgb_reg = create_regression_model(x_all, y_all)
            summary.loc[i] = [basename, RMSE, R_squared, None]
            out.to_excel(writer, sheet_name=basename)
            dump(xgb_reg, open(common.out_file(basename+common.model_suffix), "wb"))
            i = i+1
            x_all['class'] = basename
            complete_classification_set.append(x_all) #add to the set used to do classification training
            if 'low_porosity' in basename:
                low_classification_set.append(x_all)  # add to the set used to do classification training
            else:
                high_classification_set.append(x_all)  # add to the set used to do classification training
        # construct the classification model
        complete_accuracy = process_classification_model(complete_classification_set, 'complete', writer)
        low_accuracy = process_classification_model(low_classification_set, 'low_porosity', writer)
        high_accuracy = process_classification_model(high_classification_set, 'high_porosity', writer)
        #classification_set_df = pd.concat(complete_classification_set, axis=0, ignore_index=True)
        #xgb_class, encoder, accuracy, cm, out = create_classification_model(classification_set_df)
        #dump(xgb_class, open(common.out_file('classifier' + common.model_suffix), "wb"))
        #dump(encoder, open(common.out_file('encoder' + common.model_suffix), "wb"))
        #out.to_excel(writer, sheet_name='Classification')
        #cm.to_excel(writer, sheet_name='Classification confusion matrix')
        summary.loc[i + 1] = ['Classification', None, None, complete_accuracy]
        summary.loc[i + 2] = ['Low porosity classification', None, None, low_accuracy]
        summary.loc[i + 3] = ['High porosity classification', None, None, high_accuracy]
        summary.to_excel(writer, sheet_name='Summary')




if __name__ == '__main__':
    create_models()