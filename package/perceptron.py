from random import seed
from random import randrange
from csv import reader
# code based in majority from https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
class Perceptron():

    # Calculate accuracy percentage
    def accuracy_metric(self,actual, predicted):
		correct = 0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
			    correct += 1
	    return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self,dataset, algorithm, n_folds, *args):
	    folds = cross_validation_split(dataset, n_folds)
	    scores = list()
	    for fold in folds:
		    train_set = list(folds)
		    train_set.remove(fold)
		    train_set = sum(train_set, [])
		    test_set = list()
		    for row in fold:
			    row_copy = list(row)
			    test_set.append(row_copy)
			    row_copy[-1] = None
		    predicted = algorithm(train_set, test_set, *args)
		    actual = [row[-1] for row in fold]
	        accuracy = accuracy_metric(actual, predicted)
		    scores.append(accuracy)
	    return scores

    # Make a prediction with weights
    def predict(self,row, weights):
	    activation = weights[0]
	    for i in range(len(row)-1):
		    activation += weights[i + 1] * row[i]
	    return 1.0 if activation >= 0.0 else 0.0

    # Estimate Perceptron weights using stochastic gradient descent
    def train_weights(self,train, l_rate, n_epoch):
	    weights = [0.0 for i in range(len(train[0]))]
	    for epoch in range(n_epoch):
		    for row in train:
			    prediction = predict(row, weights)
			    error = row[-1] - prediction
			    weights[0] = weights[0] + l_rate * error
			    for i in range(len(row)-1):
				    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	    return weights

    # Perceptron Algorithm With Stochastic Gradient Descent
    def perceptron(self, train, test, l_rate, n_epoch):
	    predictions = list()
	    weights = train_weights(train, l_rate, n_epoch)
	    for row in test:
		    prediction = predict(row, weights)
		    predictions.append(prediction)
	    return(predictions)

if __name__ == '__main__':
    # Test the Perceptron algorithm on the sonar dataset
    seed(1)
    # load and prepare data
    filename = 'sonar.all-data.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
	    str_column_to_float(dataset, i)
    # convert string class to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # evaluate algorithm
    n_folds = 3
    l_rate = 0.01
    n_epoch = 500
    scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
