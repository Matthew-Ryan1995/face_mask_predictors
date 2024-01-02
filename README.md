# face_mask_predictors
Analysing survey data to determine influential predictors of face mask usage in Australia.

# The rows are filtered in the plot of AU in time
In ../code/reproduce_paper_fig.py, the rows are grouped by "endtime" with the group size larger than or equal to 300.

# For different states, original count changes to binary classification, showing proportion of wearing mask or not
needed to classify no and yes for wearing the mask outside the home. now cleaning the rows to see whether the figures change obviously

threshold = 1006
original row: 53833, new: 51826

threshold = 10029
original row: 53933, new: 34994