# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
Provided data set 'bankmarketing_train.csv' contains clients data of a Portuguese banking institution relate to  direct marketing campaigns (phone calls).
We see to predict if if the client will subscribe a term deposit (column 'y)'.

The best performing model with the accuracy of *91,76024%* is LogisticRegression from the HyperDrive pipeline.

## Scikit-learn Pipeline

We are training multiple scikit-learn [LogisticRegression]([LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)) models with different values of the `-C`(Inverse of regularization strength) and `-max_iter`(Maximum number of iterations taken for the solvers to converge)  hyperparameters.

The training pipeline consists of the following steps:
1. Loading tabular data.
2. Cleaning data: removing missing values, one hot encoding for categorical features, transforming text features into numerical features
3. Using a HyperDrive to choose the values of hyper-parameters (C and max_iter) and perform scikit-learn LogisticRegression model training run with these values
4. Log the metric for each run and save the corresponding model.

### Data
'bankmarketing_train.csv' is a data set [publicly available](https://archive.ics.uci.edu/ml/datasets/Bank%2BMarketing) for research.

Input variables:
* *age* (numeric)
* *job* : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
* *marital* : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
* *education* (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
* *default*: has credit in default? (categorical: 'no','yes','unknown')
* *housing*: has housing loan? (categorical: 'no','yes','unknown')
* *loan*: has personal loan? (categorical: 'no','yes','unknown')
* *contact*: contact communication type (categorical: 'cellular','telephone')
* *month*: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
* *day_of_week*: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
* *duration*: last contact duration, in seconds (numeric). Note:  data set used for the project doesn't contain obsevations with 0 value. 
* *campaign*: number of contacts performed during this campaign and for this client (numeric, includes last contact)
* *pdays*: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
* *previous*: number of contacts performed before this campaign and for this client (numeric)
* *poutcome*: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
* *emp.var.rate*: employment variation rate - quarterly indicator (numeric)
* *cons.price.idx*: consumer price index - monthly indicator (numeric)
* *cons.conf.idx*: consumer confidence index - monthly indicator (numeric)
* *euribor3m*: euribor 3 month rate - daily indicator (numeric)
* *nr.employed*: number of employees - quarterly indicator (numeric)

Output variable (desired target):
* *y*: has the client subscribed a term deposit? (binary: 'yes','no')

### Classification algorithm

LogisticRegression is a probabilistic classification model. It predicts the likelihood of a binary outcome using logit function, in our case the probability that the client will subscribe a term deposit.
[Logit function]() or the log-odds is the logarithm of the odds: log(p / 1 - p), where p is a probability (of Y being one of the categories).

### Hyperparameter tuning

HyperDrive helps us to try different combinations of the hyper-parameter values to be able to choose the best combination which maximazes the chosen metric: 'Accuracy'.

#### Choice of the parameter sampler

[RandomParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py) is  my choice to randomly try out different combinations of hyperparameters. Generally it provides good results in less time compared to exhaustive GridParameterSampling. 

`max-iter` hypermarameter has a descete value. I chose `choice` distribution with the list of values which have a significant difference to see the impact of their values on the target metric. 
C hyperparameter has a continious value (float). I chose `loguniform` distribution for C.

#### Early stopping policy

LogisticRegression has only one iteration while training. So there is no need for stopping policy: [NoTerminationPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.noterminationpolicy?view=azure-ml-py) 

## AutoML

AutoML's run best performing model (within 30 minutes timeout) is [VotingEnsemble](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-runtime/azureml.train.automl.runtime.ensemble.votingensemble?view=azure-ml-py).
VotingEnsemble is ensemble model created from previous AutoML iterations, that implements soft voting (the output class is the prediction based on the average of probability given to that class). 

Best pipeline details:
```
Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(allow_chargram=None, enable_dnn=None,
                                 enable_feature_sweeping=None,
                                 feature_sweeping_config=None,
                                 feature_sweeping_timeout=None,
                                 featurization_config=None, force_text_dnn=None,
                                 is_cross_validation=None,
                                 is_onnx_compatible=None, logger=None,
                                 observer=None, task=None, working_dir=None)),
                ('prefittedsoftvotingclassifier...
                                                                                                    min_samples_split=0.15052631578947367,
                                                                                                    min_weight_fraction_leaf=0.0,
                                                                                                    n_estimators=10,
                                                                                                    n_jobs=1,
                                                                                                    oob_score=False,
                                                                                                    random_state=None,
                                                                                                    verbose=0,
                                                                                                    warm_start=False))],
                                                                     verbose=False))],
                                               flatten_transform=None,
                                               weights=[0.07142857142857142,
                                                        0.5,
                                                        0.07142857142857142,
                                                        0.07142857142857142,
                                                        0.07142857142857142,
                                                        0.07142857142857142,
                                                        0.07142857142857142,
                                                        0.07142857142857142]))],
         verbose=False)
```


## Pipeline comparison

HyperDrive tuning of the Logistic Regression gives *91,76024%* of accuracy with C between 119 and 174 and and max-iter between 125 and 200.
![HyperDrive best 10 runs](./hd-best-10-runs.JPG)

AutoML VotingEnsemble best run score is *91,71%*.
![AutoML best 10 runs](./automl-best-10-runs.JPG)

We can see that HyperDrive pipeline slightly outperforms AutoML run. This can be explained by the fact that AutoML tries out different algorithms with default hyperparameters values. AutoML model gives a good indication of the final model research direction and can augment human expertise.

HyperDrive pipe helps to automatically explore hyperparameter values which help to maximize the target performance metric.
The drawback of this approach is that we limit our exploration to one type of model and the chosen hyperparameters.
I suppose this approach makes sense when we'd like to make adjustments to the model that we already know performs well on the given data. 

AutoML pipeline on the other hand helps us to explore different models which provide good performance based on the target performance metric. This approach helps to screen more possibilities and discover models based on different algorithms. 

## Future work

Running AutoML with featurazation='auto' detected the imbalance in data, which can lead to biased predictions. To deal with it we need either to use a metric that is more appropriate for this matter (AutoML suggests using 'AUC_weighted'), either balance the data.


# Resources
* [Tune hyperparameters for your model with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#define-search-space)
* [Estimator Azure ML package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.estimator?view=azure-ml-py)
* [Model Azure ML class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py)
* [AutoMLRun class](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.run.automlrun?view=azure-ml-py)
* [GitHub - Azure ML notebooks](https://github.com/Azure/MachineLearningNotebooks)
* [Article - Voting Classifier using Sklearn](https://www.geeksforgeeks.org/ml-voting-classifier-using-sklearn/)
* [Article - WHAT and WHY of Log Odds by Piyush Agarwal](https://towardsdatascience.com/https-towardsdatascience-com-what-and-why-of-log-odds-64ba988bf704)
