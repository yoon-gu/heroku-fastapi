# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Yoon-gu Hwang created the random forests model with default parameter. Its dump file `rf.joblib` is located in `model`.

## Intended Use

This model aims to predict a person's salary when the following information is given

1. `age`
1. `workclass`
1. `fnlgt`
1. `education`
1. `education_num`
1. `marital_status`
1. `occupation`
1. `relationship`
1. `race`
1. `sex`
1. `capital_gain`
1. `capital_loss`
1. `hours_per_week`
1. `native_country`

## Training Data

The UCI Machine Learning Repository provides Census Income Dataset. You can see the details of the dataset on https://archive.ics.uci.edu/ml/datasets/census+income.

## Evaluation Data

20% of data is used for evaluation with random sampling.

## Metrics
The model's evaulation results:

- precision : 0.720734506503443
- recall : 0.6124837451235371
- fbeta : 0.6622144112478031

## Ethical Considerations

Race, gender and education contained in this data are private information.

## Caveats and Recommendations

The salary column has imbalance with resepct to the number of samples. To resolve this, you could apply resample to the balance.