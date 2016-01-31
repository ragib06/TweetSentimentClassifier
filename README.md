## UIC CS 583 Project 2 ##

This is a classification project to classify tweet text of Election sentiment for the candidates Obama and Romeny.

Libraries required:
* scikit-Learn: http://scikit-learn.org/stable/install.html
* nltk: http://www.nltk.org/install.html

Instructions to run:
- Go to the 'src' directory from command line
- Run the command "python TweetClassifier.py"

The general command options:

 python TweetClassifier.py [-d data_file_path] [-t test_file_path]

 If you provide nothing, it will load the default file and will run a 10-fold cross-validation and then show report.
 If you only provide data_file_pith, it'll load that file, run a 10-fold cross-validation and then show report.
 If you provide data_file_path and test_file_path, it'll train the classifier with data_file_path file and test against test_file_path



Contributors:

Ragib Ahsan,

Hasan Iqbal
