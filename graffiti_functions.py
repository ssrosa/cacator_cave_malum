import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandasql import sqldf 
import copy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import confusion_matrix 
import itertools

import pydotplus
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

#########################################################
# DATA CLEANING #########################################
def keyword_matrix(pysqldf, orig_categories, keywords):
    
    '''
    A tool for exploring the incidence of certain key words within 
    a data set's categories/classes. It uses a Pandasql query to 
    get counts of a key word for each category in the data and
    constructs a one-row dataframe of the counts.
    
    Each key word's row gets appended to the previous rows in the 
    DataFrame.
    
    In this version, the data set to query is hard coded as 'df'.
    The category to search within is hard coded as
    'In English.'
    
    Parameters:
    
    pysqldf ()

    orig_categories (list.) The categories in the data set to traverse.
    Each should correspond to a category/class in the data set.
    
    keywords (list.) The words to search for and count. If a single 
    word is sought, it can be passed as ['word']. Can't be multi-level,
    i.e. a list of lists.
    
    
    Returns: 
    
    keywords_df (pandas.core.frame.DataFrame.) The full DataFrame with
    the counts of all key words in all categories.
    '''
    
    #A list of category names for the df
    categories = copy.deepcopy(orig_categories)
    categories.insert(0, 'Key_Word')

    #list of columns to add to the query string. (Doesn't have 'Key Word.')
    cats = copy.deepcopy(orig_categories)

    #Instantiate an empty dataframe to hold results
    keywords_df = pd.DataFrame(columns = categories)

    #Build the dataframe
    for keyword in keywords:
        #Build query string.
        q = '''SELECT '{}' Key_Word,'''.format(keyword)
        #Build query string by iterating over list of columns
        #Simultaneously add the keyword str as many types as there are categories
        for cat in cats:
            q += ''' SUM(
                    CASE WHEN
                    [In English] LIKE '%{}%'
                    AND Category == '{}'
                    THEN 1 ELSE 0 END
                    ) {},
            '''.format(keyword, cat, cat)
        q = q[:-14]
        q += '''
        FROM df_raw;'''
        #Apply the query string with the given keyword to the df.
        results = pysqldf(q)
        #Concat the new row onto the df
        keywords_df = pd.concat([keywords_df, results])

    keywords_df.reset_index(inplace = True)
    keywords_df.drop(['index'], axis = 1, inplace = True)
    return keywords_df

def write(df, words_sought, read_from, write_to, phrase, exact = False):
    '''
    A tool for changing the values of certain cells in a DataFrame under
    certain conditions. Looks for a given value, whetehr an exact or partial
    match, and then replaces the value of a given column in that row with a
    new given value. Can replace the very value that was sought or another in
    that row. The value sought can be a list or a single value.
    
    Parameters:
    
    df (pandas.core.frame.DataFrame.) The df to search for words and in which to
    write the new phrase.
    
    words_sought (list.) Key words to look for in the "read from" column. Can be a
    single value passed as ['value'].
    
    read_from (str.) The column in which to search for the words sought.
    
    write_to (str.) The column to write the new phrase into. Can be the same as 
    read_from.
    
    phrase (str, int, or float.) The phrase to be written into the cell.
    
    exact (bool.) Set to False by default. If True, looks for exact match (==)
    not just a partial match ("in.")
    
    Returns:
    
    Doesn't return a new object. Writes the new value into the DataFrame in place.
    Use with caution!
    '''
    
    #Iterate over each row in the given df
    for index in df.index:
        #If one of the given words is found in the given column
        if not exact:
            if any(word_sought in df.at[index, read_from] for word_sought in words_sought):
                #Write the new phrase to the other given column
                df.at[index, write_to] = phrase
        else: 
            #If the word sought matches the value of the cell
            if df.at[index, read_from] == words_sought:
                #Write the new phrase to the other given column
                df.at[index, write_to] = phrase

###########################################################
# CONFUSION MATRIX ########################################
def plot_conf_matrix(cm, classes, normalize=False, 
                          title='Confusion Matrix', cmap=plt.cm.Blues):
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Matrix, normalized")
#    else:
#        print('Matrix')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

###########################################################
# DECISION TREE ###########################################
def draw_graph(clf):
    '''
    Visualizes a decision tree using GraphViz.
    
    Parameters:
    
    clf (DecisionTreeClassifier.) Must be already fit with data.
    
    Returns:
    
    image (graph.create_png().) Call the assigned variable to draw the image. 
    '''
    dot_data = StringIO()

    #Feeds from decision tree classifier instantiated above
    export_graphviz(clf, out_file=dot_data, filled=True, 
                    rounded=True,special_characters=True)

    #Create graph
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
    image = graph.create_png()
    
    #Draw image
    return image

###########################################################
# HYPERPARAMETERS #########################################
def hyper(df, param, to_set, X_train, X_test, y_train, y_test):
    '''
    A tool to 
    '''
    #Instantiate a classifier with all default hyperparameters 
    #except whatever was passed  
    clf = DecisionTreeClassifier(
            criterion = param if to_set == 'criterion' else 'entropy',
            max_depth = param if to_set == 'depth' else None,
            min_samples_split = param if to_set == 'samples_split' else 2,
            min_samples_leaf = param if to_set == 'samples_leaf' else 1,
            max_features = param if to_set == 'features' else None,
            max_leaf_nodes = param if to_set == 'leaf_nodes' else None,
            min_impurity_decrease = param if to_set == 'impurity_decrease'  else 0
                                 )
    #Fit the data to the classifier
    clf.fit(X_train,y_train) 
    #Predict values for training data
    y_hat_train = clf.predict(X_train)
    #Predict values for testing data
    y_hat_test = clf.predict(X_test)

    #Get the scores
    prec_train = precision_score(y_train, y_hat_train, average = None).mean()
    prec_test = precision_score(y_test, y_hat_test, average = None).mean()
    recall_train = recall_score(y_train, y_hat_train, average = None).mean()
    recall_test =  recall_score(y_test, y_hat_test, average = None).mean()
    f1_train = f1_score(y_train, y_hat_train, average = None).mean()
    f1_test = f1_score(y_test, y_hat_test, average = None).mean()
    
    #Write the scores to the df
    df = df.append({'param_values': param,
               'prec_train': prec_train,
               'prec_test': prec_test,
               'recall_train': recall_train,
               'recall_test': recall_test,
               'f1_train': f1_train,
               'f1_test': f1_test              
              }, ignore_index = True)
    
    return df

def compare_hypers(params, to_set, X_train, X_test, y_train, y_test):
    '''
    A tool to
    '''
    score_columns = ['param_values', 'prec_train', 'prec_test', 
                     'recall_train', 'recall_test', 'f1_train', 'f1_test']
    
    #Instantiate adf to use for storing scores
    df = pd.DataFrame(columns = score_columns)
    
    #Run a model for each parameter and record
    #its scores in the df
    for param in params:
        df = hyper(df, param, to_set, X_train, X_test, y_train, y_test)
    #Return the df with the scores from all the models
    return df

def plot_hypers(df, title):
    '''
    A tool to
    
    '''
    # Number of iterations to plot along the x axis
    x = df['param_values']
   # tick_marks = np.arange(x.shape[0])
    #Draw a figure
    plt.figure(figsize=(10,6))
    
    plt.plot(x, df['prec_train'], 'b-', label = 'Training precision')
    plt.plot(x, df['prec_test'], 'b:', label = 'Test precision')
    plt.plot(x, df['recall_train'], 'g-', label = 'Training recall')
    plt.plot(x, df['recall_test'], 'g:', label = 'Test recall')
    plt.plot(x, df['f1_train'], 'r-', label = 'Training F1')
    plt.plot(x, df['f1_test'], 'r:', label = 'Test F1')
    #plt.xticks(tick_marks, x, rotation=45)
    plt.ylabel('Scores')
    plt.xlabel('Hyperparameter values')
    plt.title(title)
    plt.legend()
    plt.show()

###########################################################
# COMPARING SCORES ########################################
def write_scores(df, y_train, y_hat_train, y_test, y_hat_test):
    '''
    Populates a DataFrame with the sores for a model. Similar to
    classification report but shorter and tidier.

    Parameters:
    df (pd.DataFrame.) Already instantiated with its columnss.

    the rest (np.array.) Targets/classes from the train/test split 
    and from running a prediction with the model.

    Returns:
    df (pd.DataFrame.) The same df as before, now with scores!
    '''
    #Get scores from training and test sets
    prec_train = precision_score(y_train, y_hat_train, average = None).mean()
    prec_test = precision_score(y_test, y_hat_test, average = None).mean()
    recall_train = recall_score(y_train, y_hat_train, average = None).mean()
    recall_test =  recall_score(y_test, y_hat_test, average = None).mean()
    f1_train = f1_score(y_train, y_hat_train, average = None).mean()
    f1_test = f1_score(y_test, y_hat_test, average = None).mean()
    #Append the training scores
    df = df.append({'set': 'training',
               'precision': prec_train,
               'recall': recall_train,
               'f1': f1_train,           
              }, ignore_index = True)
    #Append the test scores
    df = df.append({'set': 'test',
               'precision': prec_test,
               'recall': recall_test,
               'f1': f1_test              
              }, ignore_index = True)
    #Set index
    df.set_index('set', inplace = True)
    
    return df

###########################################################
###########################################################
###########################################################
