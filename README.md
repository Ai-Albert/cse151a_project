# cse151a_project

## Preprocessing
Since we have 2 datasets, we will aggregate the data by joining on the movie name and year. As part of our preprocessing procedure, we take the following steps:
1) convert `Duration` from strings to minutes (`int64`)
2) convert `winner` to `0`s for `False` and `1`s for `True`
3) get rid of all the rows with any null values
4) normalize the numerical data
5) encode the categorical data using one-hot encoding because there is no ordering amongst the categorical variables

## Milestone 3 Results and Next Steps

***Where does your model fit in the fitting graph.***

The difference in mse between train and test isn't large so there's no sign of overfitting.

***What are the next 2 models you are thinking of and why?***

gradient descent with sigmoid curve activation function because it is classification
gradient descent with relu + softmax activation function because it is another great activation for this classification task, and we wanted to try out different activation functions.

***Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?***

The model was not very accurate (accuracy around 0.75 for testing dataset and 0.73 for training dataset).

We could add an epsilon to shift the linear regression line for better accuracy. However, we think using other models that are more suitable for classification might be a better approach to this task.
