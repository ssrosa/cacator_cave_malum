# CACATOR CAVE MALUM: Four Classifiers for the Beautiful, Filthy Graffiti of Pompeii

Using a [data set of Pompeii graffiti by Alex Rose](https://core.tdar.org/dataset/445837/database-for-the-scratched-voices-begging-to-be-heard-the-graffiti-of-pompeii-and-today), can we build a classifier for the graffiti of Pompeii that ignores the textual content and it takes signals from data on location, number of characters, &c?

# Files

**graffiti_slides.pdf**: A slide deck to present findings and recommendations to archeologists.

**graffiti.csv**: The raw data set.

**graffiti_function.py**: A script with all my code for manipulating the data and generating visualizations.

**images**: Illustrations of the archeological site and graffiti artefacts. 

**index.ipynb**: A Jupyter Notebook with my data exploration and analysis.

# Methods 
My goal was to classify graffiti without knowing its message in translation. I wanted to use the features associated with the 'site context' (a term I made up) of the graffito: its particular location in a place, on a wall, &c. My hunch was that a graffito found in a site identified as, say, a brothel, would be the same sort of lewd graffiti as all the others found in brothels. I discarded the translation column from the dataset after engineering a couple features from it: character count and presence or absence of explanation points. These seemed like the sort of reasonable signs that researchers could look for in the writing without understanding Vulgar Latin.

I was dissatisfied with the classes used by the creator of the dataset so I reorgnized the data into new classes that were more meaningful and balanced.

I used simple classifiers -- logistic regression and decision trees -- with significatn hyperparameter tuning but could not break 50% accuracy.

Ensemble classifiers -- random forests and gradient boosting -- did slightly better but still couldn't break 50% accuracy.

# Conclusions

The startling finding was that the most important features were those describing the text of the graffiti, including the character count feature I had engineered. This suggested that my hunch about the importance of the site context of a graffito was wrong: classification really did depend on knowing what the graffito said. 

# Recommendations

In the nontechnical slide deck I recommend that archeologists not rely on machine learning to classify the graffiti they discover in the ongoing investigation of Pompeii: rather I recommend that they invest in training more researchers to translate Vulgar Latin.