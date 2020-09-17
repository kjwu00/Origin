# Origin 

Classification of English-Translated Texts by Source Language

## Introduction

As more and more people from many nations learn to speak English,
an increasing amount of text is translated from foreign languages to English.
We are building a system that facilitates this cross-cultural communication
by identifying the source languages of transaled texts from around the world.
We have build a corpus of professionally translated short stories in six different
source languages and the goal is for our model to classify brief passages of those
stories by their original language.

## Task

Our task is to automatically detect the source language of a text that was translated
to English (or possibly originally written in English itself). More specifically, we aim
to create a system that can classify an 100 word fragment of an English-translated fictional
story into one of six source languages: Spanish, French, Korean, Portugeuese, Russian, or English
(original language).

## Result
We compared multiple classification models (multiclass logistic regression, SVM, RNN) with unigram and bigram features. 
We found that tthe multiclass logistic regression model with unigram features was optimal, achieving
an accuracy of 83%.

##
Please view paper.pdf for a more detailed write up of this project.
