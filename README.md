# cse151a_project

## Preprocessing
Since we have 2 datasets, we will aggregate the data by joining on the movie name and year. As part of our preprocessing procedure, we take the following steps:
1) convert `Duration` from strings to minutes (`int64`)
2) convert `winner` to `0`s for `False` and `1`s for `True`
3) get rid of all the rows with any null values
4) normalize the numerical data
5) encode the categorical data using one-hot encoding because there is no ordering amongst the categorical variables
