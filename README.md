# Music Genre Classification Using Machine Learning

A comparative study of classical and deep learning approaches to music genre classification on the GTZAN dataset. Five models - kNN, SVM, XGBoost, ANN, and CNN - are evaluated across four feature representations: 30-second and 3-second tabular audio features, a custom self-extracted feature set, and spectrogram images.

## Key findings:

No single classifier dominated: XGBoost performed best on 30-second data, SVM on 3-second data, and ANN on the self-extracted set, with kNN close behind throughout.
Feature representation mattered more than model choice - every model improved substantially on the 3-second segmented data (more training samples) and underperformed on the reduced self-extracted feature set.
The CNN, trained on raw spectrograms, significantly underperformed all tabular-feature classifiers due to overfitting from limited training data, despite dropout regularization.
Hyperparameter tuning, training curves, and confusion matrices were used to evaluate and compare model behavior across genres.

Includes hyperparameter search results, training/validation curves, and confusion matrix analysis for each model and feature set.
