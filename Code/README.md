The SplitData.py file takes in full chunks of text, cleans the text and splits it into 10 
word text blocks, then separates it into training, validation, and testing sets (each with
its own folder). It also creates files containing all of the features in the training 
dataset for uni-gram and bi-gram.

The cs221.py file trains, tests, and validates our KNN, logistic regression, and SVM
models. We changed model variable in line 14 to be 1 for unigram or 2 for bigram.
Otherwise, we just ran to program to collect our results. We also changed the valPath
(Line 133) to Test_Test and only ran the logistic unigram C = 10 with L2 regularization,
in order to test our final model.

The RNN.py file contains our RNN model.
