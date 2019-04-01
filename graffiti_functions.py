import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandasql import sqldf 
import copy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import confusion_matrix 
import itertools

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

def keyword_matrix(orig_categories, keywords):
    
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

    #Fill values in a column conditionally
#Words sought can be a list or a single value
#Exact match or just 'appears in' can be specified
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


#Example function to visualize a confusion matrix without yellow brick
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