Bidirectional RNN in PyTorch for Text Classification - Base problem category as per Ready Tensor specifications.

* pytorch
* sklearn
* python
* pandas
* numpy
* scikit-optimize
* flask
* nginx
* uvicorn
* docker
* text classification
* lstm
* gru

This is a Text Classifier that uses a Bidirectional Recurrent Neural Network (RNN) implemented through PyTorch.

The Bidirectional RNN consists of 5 layers that include embedding, GRU or LSTM, relu, and linear layers and supports both LSTM and GRU implementations. 

The data preprocessing step includes tokenizing the input text and pad/truncating it to fixed sequence length. In regards to processing the labels, a label encoder is used to turn the string representation of a class into a numerical representation.

Hyperparameter Tuning (HPT) is conducted by finding the optimal rnn strategy, embedding size, latent dimension, and batch size that model uses to run.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as clickbait, drug, and movie reviews as well as spam texts and tweets from Twitter.

This Text Classifier is written using Python as its programming language. PyTorch is used to implement the main algorithm while SciKitLearn is used to evaluate the model and preprocess the data. Numpy, pandas, and NLTK are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.



