# Sentiment-Analysis
RNN based sentiment analyser on amazon_phone_reviews
 
#Pre-process function filters all unwanted data in input files
#Only ‘summary’ and ‘ratings’ are considered
#Define max_features=100000, maxlen=250(i.e., cut texts after this number of words (among to max_features most common words), 
batch_size=32
#Use Tokenizer to convert text into sequences so the network can deal with it as input.
#words from each text is converted into sequence of vectors to input the neural network

Building the model:
#Sequential model is used. Layers are added  via .add() method.
#Each resulting sequence of vectors is fed as input to the embedding layer. Then LSTM layer is added. The dense layer has 
one neuron and uses a rectifier activation function.
#The model uses logarithmic loss (binary_classentropy) and is optimized using the efficient ADAM optimization procedure.
#The output of this model is the loss and accuracy of a text belonging to positive or negative category. 
#After the model is trained, we evaluate its loss and accuracy on the test dataset.

Output :
Accuracy   : 89.19999%
Loss	     : 38.6335%
Result Analysis :  We infer that a better accuracy can be obtained  if a network is trained with more datasets,  
perhaps using a larger embedding and adding more hidden layers.
