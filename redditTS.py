# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:32:31 2022

@author: Joseph Levesque

Professor Carrie Beam

UMGC DATA 620 - 9040

Assignment 12.1

This script provides functions enabling the user to complete the tasks required
for Assignment 12.1, including:
    1) Pulling a time series dataset from the reddit API via getSubredditYear()
        and getSubredditRange();
    2) Saving the dataset as a .pkl file, either by individual year or collectively, 
        via saveYearList();
    3) Loading a dataset from a .pkl file via loadYearList();
    4) Converting reddit's time-from-epoch timestamps to readable date-time Strings 
        via getDateTimeFromUTC();
    5) Preprocessing and transforming a single year's data into a term-document 
        matrix via processTextList();
    6) Retrieving dataframes containing aggregate token counts for corpora in a 
        given year range and optionally saving them as .csv files via 
        getCountDataframesInRange(); and
    7) Saving pandas dataframes to .csv with the syntax used in the other functions 
        within this script via dfToCsv().

All functions are preceeded by full javadoc documentation.  In-line comments are 
also provided for convenience.

Additional scripts which were run in the console on output from the functions are 
provided in comments at the bottom of this file.

While it is noted within the code, I would caution future users--or graders--of
the functions for pulling data from the reddit API.  Their server is quite slow,
only permits 500 posts to be pulled per query, throttles repeated queries, and
furthermore is rather unreliable and occasionally goes partially offline for varying
periods of time (2013 r/Politics data, for example, was unavailable during the 
week this project was assigned).  The dataset collection process took a bit over 
14 hours on my fairly decent hardware to trawl the 3.6 million posts made to 
r/Politics.  I would not recommend running the functions to gather the dataset
without either being absolutely certain that everything was set up properly or 
setting the optional "limit" argument in the call to api.search_submissions() within
the getSubredditYear() function.
"""

from psaw import PushshiftAPI
import datetime as dt
import pickle
import re
import pandas as pd
import warnings
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

## Create a custom tokenizer class to pass to the CountVectorizer.
# This will permit the stemming and lemmatization of the corpus.
# Note that this was adapted from the CountVectorizer documentation found at:
# https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
class LemmaStemTokenizer:
    
     def __init__(self):
         
         # Initialize the lemmatizer and stemmer on construction of a 
         #  LemmaStemTokenizer Object
         self.lemmatizer = WordNetLemmatizer()
         self.stemmer = PorterStemmer()
     
     def __call__(self, doc):
         
         # Lemmatize and stem words in addition to tokenizing a document when called
         return [self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(doc)]

## Retrieves and optionally saves to .csv dataframes containing aggregate token
#   counts for corpora in range(firstYear, lastYear + 1).
#
# @arg firstYear the year of the first corpus.
# @arg lastYear the year of the last corpus.
# @arg saveToFile boolean value determining whether the dataframes should be
#                   saved to the current work directory as "tokenCountsXXXX.csv",
#                   where XXXX is the associated year.
#                 This is an optional argument.  The default is false.
#
# @return a list containing a dictionary for each year in range(firstYear, lastYear + 1).
#         Each dictionary contains two elements:
#            1) "year", the year associated with the counts in the dictionary
#            2) "counts", a pandas dataframe with two columns:
#                   1) "token", the 'word' from the vocabulary assoicated the counts in the dataframe
#                   2) "count", the number of times the token appeared in the corpus
#
# @postState if saveToFile is set to true, a .csv file named "tokenCountsXXXX.csv",
#               where XXXX is the associated year, is created in the current work
#               directory for each year's data within the range [firstYear, LastYear]
#               inclusive.         
def getCountDataframesInRange(firstYear:int, lastYear:int, saveToFile:bool = False):
    
    # Create a list to populate with the count data
    countList = []
    
    # Iterate through the specified years
    for year in range(firstYear, lastYear + 1):
        
        # Load the raw data for that year; print timestamps to update the user.
        # Expect about 2 minutes per iteration of the loop/year (SSD, i7 processor)
        print("[" + dt.datetime.now().strftime("%H:%M:%S") + "]: Loading raw data from " + str(year) + "...")
        docList = loadYearList(year)
        print("[" + dt.datetime.now().strftime("%H:%M:%S") + "]: Raw data from " + str(year) + " loaded.")
        
        # Preprocess the data, then generate the counts dataframe
        print("[" + dt.datetime.now().strftime("%H:%M:%S") + "]: Processing data from " + str(year) + "...")
        yearDataFrame = processTextList(docList)
        print("[" + dt.datetime.now().strftime("%H:%M:%S") + "]: Data from " + str(year) + " processed.")
        
        # If the dataFrame is empty (2013), continue to the next year
        if(yearDataFrame is None):
            print("[" + dt.datetime.now().strftime("%H:%M:%S") + "]: Data from " + str(year) + " is empty.  Skipping to the next year...\n")
            continue
        
        # If saveToFile argument is True, then save the dataframe as a .csv
        # Note that the filename will be "tokenCountsXXXX.csv", where XXXX 
        #   is the associated year.
        if(saveToFile):
            dfToCsv(yearDataFrame, year)
            print("[" + dt.datetime.now().strftime("%H:%M:%S") + "]: Counts from " + str(year) + " saved to tokenCounts" + str(year) + ".csv .\n")
        
        # Combine the year and the dataframe in a dictionary
        curYearDict = {
                "year": year,
                "counts": yearDataFrame
                }
        # Append the dictionary to the list to be returned
        countList.append(curYearDict)
    
    print("[" + dt.datetime.now().strftime("%H:%M:%S") + "]: Dataframe generation complete.")
    return countList
        
        
## Process a corpus in the form of a list of Strings.  This includes:
#       1) Removing punctuation, capitalization, and English stop words
#       2) Converting characters with accents to their base ASCII form
#       3) Lemmatizing and stemming the remaining tokens
#
# @arg stringList a List of Strings containing the documents to process
# @arg minNGram the smallest n-gram to consider.  Default 1.
# @arg maxNGram the largest n-gram to consider.  Default 3.
# @arg maxVocabulary the largest vocabulary to consider.  E.g., if maxVocabulary = 100,
#                       then only the 100 most frequent n-grams will be included
#                       in the result.  Default 100.
#                     If no cap is desired, then pass "maxVocabulary = None".
# @arg isTotalCounts If True, then only aggregate token counts for the full corpus
#                       will be returned.  If False, token counts for each document
#                       will be returned in a term-document matrix.  Default True.
#
# @return Either:
#           1) if isTotalCounts = True, a pandas dataframe with two columns:
#               1) "Token", the tokens in the vocabulary
#               2) "Count", the number of times each token occurs in the corpus
#           2) if isTotalCounts = False, a dictionary containing two elements:
#               1) "tdm", the term-document matrix.  Note that this is stored as
#                   a sparse matrix.  Each row is a document and each column is a
#                   token.  To convert to a readable array, simply use self.toarray()  
#               2) "vocab", a dictionary containing token-index pairs, where the
#                   indices correspond to the columns of the term-document matrix.
#           3) None, if the argument stringList is empty.
def processTextList(stringList: list,
                    minNGram: int = 1, 
                    maxNGram: int = 3, 
                    maxVocabulary:int = 100, 
                    isTotalCounts:bool = True):
    
    # Return if the StringList is empty
    if(len(stringList) == 0):
        print("The list is empty.")
        return
    
    # Instantiate a list to populate with the processed text
    processedList = []
    
    # Remove punctuation
    #processedList = stringList.map(lambda x: re.sub("[;,\.!?]", "", x))
    #processedList = list(map(lambda x: re.sub("[\[\]\"\'\`\-\&$%():;,\.!?|â€™œ˜]", "", x), stringList))
    # Encoding issues required an alternate way of defining the regex statement to target punctuation.
    # "[^ \w\s]" => not (alphanumeric or whitespace)
    processedList = list(map(lambda x: re.sub("[^ \w\s]", "", x), stringList))
    
    # Load the custom stop word list created for this project
    try:
        with open("lemStemStopWords.pkl", "rb") as file:
            stopWordsList = pickle.load(file)
    except:
        print("Error loading the stop words file.  Execution terminated.")
        return
    # Additional words may be added to this list.
    # Note, however, that any added words must be stemmed and lemmatized as in 
    #   the LemmaStemTokenizer class.  E.g.:
    """
    # List of words to add to the stop word list
    newWordsToAdd = ["foo", "bar", "baz"]
    
    # Initialize the lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # Lemmatize and tokenize all the words to add
    newWordsToAdd = [stemmer.stem(lemmatizer.lemmatize(t)) for t in newWordsToAdd]
    
    # Concatenate the lists
    stopWordsList.extend(newWordsToAdd)
    
    # If desired, save the new list for future use
    try:
        with open("lemStemStopWords.pkl", "wb") as file:
            pickle.dump(lemStemStopWords, file)
    except:
        print("Error saving the stop words file.")
        
    """
    
    # Instantiate a CountVectorizer Object
    vectorizer = CountVectorizer(
            # Input is a list of strings, not a file
            input = "content",
            # Convert characters with accents to their base forms (we don't care about diacritics, etc.)
            strip_accents = "unicode",
            # Remove stop words
            stop_words = stopWordsList,
            # Remove capitalization
            lowercase = True,
            # Stem and lemmatize using the custom tokenizer class defined in redditTS.py
            tokenizer = LemmaStemTokenizer(),
            # Consider up to trigrams (if default settings for processTextList() are used)
            # This is so things like "black lives matter" and names (e.g., FirstName_LastName) are grouped
            ngram_range = (1,3),
            # ngrams should be of words, not characters
            analyzer = "word",
            # Only include the top "maxVocabulary"--default 100--features in the vocabulary
            max_features = maxVocabulary
            # Encoding?
            
            )
    
    # Generate the term-document matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        termDocumentMatrix = vectorizer.fit_transform(processedList)
    
    # For future code reusability, the option to return the term-document matrix
    #   is provided.  The default, which is all that is necessary for the
    #   DATA 620 assignment, is just to return the total counts within the corpus.
    # If isTotalCounts == True (Default), return a pandas dataframe containing
    #   the total counts within the corpus for each token in the vocabulary
    if(isTotalCounts):
        
        # Sum over the rows of the termDocumentMatrix to get the total count per token
        totalCounts = termDocumentMatrix.sum(axis = 0)
        # Convert from numpy ndarray to a regular list
        totalCounts = totalCounts.tolist()[0]
        
        # Get a list of the vocabulary with indices corresponding to the term-document matrix
        try:
            # Newer versions of sklearn use this function
            vocabList = vectorizer.get_feature_names_out()
        except:
            try:
                # Older versions of sklearn use this function.  It is deprecated
                #   in newer versions, and removed outright in v1.2.
                vocabList = vectorizer.get_feature_names()
            except:
                print("Your version of sklearn requires an alternate function call."
                      + "  Replace 'vocabList = vectorizer.get_feature_names() with"
                      + " 'vocabList = vectorizer.get_feature_names_out()'.")
                return
        
        # Combine the token counts and vocabulary list into a dataframe
        dataDict = {
                "Token" : vocabList,
                "Count" : totalCounts
                }
        tbr = pd.DataFrame(dataDict)
        
        # Sort the dataframe by Counts descending
        tbr = tbr.sort_values(by = "Count", ascending = False)
        
        return tbr
    
    # Else (!isTotalCounts), return a dictionary containing the term-document
    #   matrix ("tdm") and the vocabulary ("vocab")
    else:
        tbr = {
                # Note that tdm is stored as a sparse matrix of type 'numpy.int64'.
                # To convert to a readable array, simply use tbr["tdm"].toarray()
                "tdm" : termDocumentMatrix,
                "vocab" : vectorizer.vocabulary_
                }
        return tbr


## Compiles data from the given subreddit during the interval [firstYear, lastYear]
#   inclusive, separated by year.
#
# @arg subReddit the subreddit to search for data.
# @arg firstYear the first year to search for posts created during.  Must be an int.
# @arg lastYear the last year to search for posts created during.  Must be an int.
# @arg saveToFile boolean value determining whether the compiled data should be
#                   saved to the current work directory as "titlesXXXX.pkl", where
#                   XXXX is the associated year.
#                   This is an optional argument.  The default is false.
#
# @return a list of dictionaries containing two elements:
#           1) "year", the year associated with the data in the dictionary
#           2) "data", a list containing the titles of posts created in the
#               subreddit "subReddit" during the year "year".
#         A dictionary is appended to the list for each year in range(firstYear, lastYear + 1).  
#
# @postState if saveToFile is set to true, a .pkl file named "titlesXXXX.pkl",
#               where XXXX is the associated year, is created in the current work
#               directory for each year's data within the range [firstYear, LastYear]
#               inclusive.
def getSubredditRange(subReddit:str, firstYear:int, lastYear:int, saveToFile:bool = False):
    
    # Create a list to populate with the reddit data.
    # List elements will be dictionaries which will have elements as follows:
    #   [0]: "year":int, the year of the data
    #   [1]: "data":list[str], the data pulled from reddit with the year "year"
    yearList = []
    
    # Iterate through the years [firstYear, lastYear].
    # Note that this is inclusive of lastYear.
    for year in range(firstYear, lastYear + 1):
        
        # Print a timestamp to let the user know the program isn't frozen.
        # Extracting a full year's posts takes quite a while.
        print("[" + dt.datetime.now().strftime("%H:%M:%S") + "]: Retrieving data from " + str(year) + "...")
        
        # Retrieve the current (in the iteration) year's data from reddit
        curYearList = getSubredditYear(subReddit, year)
        
        # Combine the year and the retrieved data in a dictionary
        curYearDict = {
                "year": year,
                "data": curYearList
                }
        yearList.append(curYearDict)
        
        # Print a timestamp to let the user know the program isn't frozen.
        # Extracting a full year's posts takes quite a while.
        print("[" + dt.datetime.now().strftime("%H:%M:%S") + "]: Finished retrieving data from " + str(year) + ".")
        
        # If the optional argument saveToFile is set to true, then save the data to
        #   the current work directly as "titlesXXXX.pkl", where XXXX is the year
        #   associated with the data
        if(saveToFile):
            saveYearList(curYearList, year)
            # Print a timestamp and keep the user updated
            print("[" + dt.datetime.now().strftime("%H:%M:%S") + "]: Data from " 
                    + str(year) + " saved as titles" + str(year) + ".pkl .")
        
    return yearList
        
## Compiles data from the given subreddit during the given year.
#
# @arg subReddit the subreddit to search for data.
# @arg year the year to search for posts created during.
#
# @return a list of the titles of posts created in the subreddit "subReddit"
#           during the year "year".
def getSubredditYear(subReddit: str, year: int):
    
    # Instantiate a PushshiftAPI object
    api = PushshiftAPI()
    
    # Get the time since the epoch of the start and end of the year
    start = int(dt.datetime(year,1,1).timestamp())
    end = int(dt.datetime(year,12,31).timestamp())
    
    # Get a list of psaw Objects representing posts made to the specified 
    #   subreddit during the specified year
    subList = list(api.search_submissions(
            after = start,
            before = end,
            subreddit = subReddit,
            #limit = 100, # For testing purposes; retrieving a full year takes about an hour
            filter = ["title"]
            )
    )
    
    # Extract submission titles from the list of psaw Objects
    subList = [psawObject[1] for psawObject in subList]
    
    return subList

## Saves a list of titles of posts from year XXXX as "titlesXXXX.pkl" in the 
#   current work directory.
# To save multiple years, the syntax to pass for the year argument is "XXXX-YYYY",
#   where the years are in range(XXXX, YYYY + 1).
#
# @arg listToSave the list of titles to be saved as a .pkl file.
# @arg year the year of the data to save.  This may be an int or String.
# @arg isProcessed True if the input is processed (i.e., tokenized/lemmatized
#                   with stop words/capitalization/punctuation removed); False
#                   otherwise.
#                   The default is False.
#
# @postState a file named "titlesXXXX.pkl", where XXXX is the year provided
#               as an argument, is created in the current work directory.
def saveYearList(listToSave, year, isProcessed:bool = False):
    
    # Generate the filename
    if(isProcessed):
        fileName = "titles" + str(year) + "Processed.pkl"
    else:
        fileName = "titles" + str(year) + ".pkl"
    
    # Create the file and save the list
    with open(fileName, 'wb') as file:
        pickle.dump(listToSave, file)

## Loads a year's data stored as a .pkl file in the current work directory.
# To load multiple years, the syntax to pass for the year argument is "XXXX-YYYY",
#   where the years are in range(XXXX, YYYY + 1).
#
# @arg year the year of the data to load.  This may be an int or String.
# @arg isProcessed True if the desired data is processed (i.e., tokenized/lemmatized
#                   with stop words/capitalization/punctuation removed); False
#                   for the raw data.
#                   The default is False.
#
# @return a list of the titles of posts for the given subreddit during the 
#           specified year.
def loadYearList(year, isProcessed:bool = False):
    
    try:
        # Generate the filename
        if(isProcessed):
            fileName = "titles" + str(year) + "Processed.pkl"
        else:
                fileName = "titles" + str(year) + ".pkl"
        
        # Open the file and load the list
        with open(fileName, 'rb') as file:
            listToLoad = pickle.load(file)
        
        # Return the list loaded from the file
        return listToLoad
    except:
        print("There is no .pkl file in the current work directory associated with year " + str(year) + ".")

## Saves a dataframe containing aggregated token counts to a .csv file
#
# @arg dfToSave the dataframe to save as a .csv
# @arg year the year of the data being saved.  May be an int or String.
#
# @postState a file named "tokenCountsXXXX.csv", where XXXX is the year provided
#            as an argument, is created in the current work directory.
def dfToCsv(dfToSave, year):
    
    # Generate the filename
    fileName = "tokenCounts" + str(year) + ".csv"
    
    # Write to file
    # 'index=False' => ignore row names (row indices)
    dfToSave.to_csv(fileName, index=False)

## Converts time-from-epoch UTC to a readable datetime String.
#
# @arg the timestamp to convert, e.g. 1388534639
#
# @return a String of the form "MM/DD/YYYY, HH:MM:SS"
def getDateTimeFromUTC(timestamp):
    tbr = dt.datetime.utcfromtimestamp(timestamp).strftime("%D, %H:%M:%S")
    return tbr

"""
# This was used via console to modify the dataframes to add proportions in addition to counts.
# "returned" was a local variable storing the output of getCountDataframesInRange().
# Proportions are the proportion of the sum of the counts of the top 100 tokens.
for yearDict in returned:
    df = yearDict["counts"]
    totalCount = df["Count"].sum()
    newCol = []
    for num in df["Count"]:
        prop = float(float(num) / float(totalCount))
        newCol.append(prop)
    df["Proportion"] = newCol

# The .csv files were then overwritten with:
for yearDict in returned:
    dfToCsv(yearDict["counts"], yearDict["year"])
"""

"""
# In order to prevent having to manually load 14 files into tableau and deal with
#   non-linked tables, a column with the year was added to each table.  This could
#   have been done with ids and another table, but that would've taken more space
#   for the same result.
# "returned" was a local variable storing the output of getCountDataframesInRange().
for yearDict in returned:
    df = yearDict["counts"]
    newCol = [yearDict["year"]]*len(df)
    df["Year"] = newCol
    dfToCsv(yearDict["counts"], yearDict["year"])
"""

"""
# This creates a single .csv file with the counts for all years, instead of one for each.
# "returned" was a local variable storing the output of getCountDataframesInRange().
fullDf = pd.DataFrame({
        "Token": [],
        "Count": [],
        "Proportion": [],
        "Year": []})
for yearDict in returned:
    df = yearDict["counts"]
    fullDf = fullDf.append(df)
dfToCsv(fullDf, "2008-2022")
"""

"""
# This was used via console to determine the number of posts within the entire corpus:

fullDataset = loadYearList("2008-2022")
count = 0

for yDict in fullDataset:
    count += len(yDict["data"])

# The result was 3,667,561 posts in the dataset.
"""

#testCorpus = ["This is a document", "This is another document", "foo bar baz", "is this a document?"]