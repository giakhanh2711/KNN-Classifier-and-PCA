import utils

train_file = "gisette/gisette_trainSet.txt"
train_label_file = "gisette/gisette_trainLabels.txt"

X, Y = utils.load_data(train_file, train_label_file)