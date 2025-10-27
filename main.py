import utils
import knn_classifier

import pandas as pd
import argparse
import numpy as np
import os

train_file = "gisette/gisette_trainSet.txt"
train_label_file = "gisette/gisette_trainLabels.txt"

test_file = "gisette/gisette_testSet.txt"
test_label_file = "gisette/gisette_testLabels.txt"

ks = list(range(1, 11))
n_fold = 5
n_components = [2, 5, 10]


# Load data
print("\n Load training data:")
X, Y = utils.load_data(train_file, train_label_file)

print("\n Load test data")
X_test, Y_test = utils.load_data(test_file, test_label_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, default="brute", help="Algorithm used to compute the nearest neighbors")
    parser.add_argument("--metric", type=str, default="minkowski", help="Distance measure method")
    parser.add_argument("--p", type=int, default=2, help="p=2 + minkowski: euclidean")
    parser.add_argument("--weights", type=str, default="uniform", help="weights to calculate probability of a label")

    args = parser.parse_args()
    print(f"\nKNN for binary classification with following configurations:")
    for key in vars(args):
        print(f"   {key}: {vars(args)[key]}")
    print("\n")

    knn_classifier = knn_classifier.KNNClassifier(ks, n_components, n_splits=n_fold,
                                                  algorithm=args.algorithm, metric=args.metric, p=args.p, weights=args.weights)
    
    # # ============== 5-fold cross validation on the training data to select the best k ================
    print("1. 5-fold cross validation to select the best k")
    error_rate = knn_classifier.kf_choose_k(X, Y)

    print("2. Report average validation error:")
    df = pd.DataFrame({"k": ks,
                       "error rate on val set (%)": np.round(error_rate, 4) * 100})
    df.to_csv("average_validation_error.csv", index=False)
    print(f"save file successfully!")

    best_k = ks[np.argmin(error_rate)]
    print(f"\n **Best k after using {n_fold} cross validation is {best_k}** \n")

    # test with best k
    count_error_test, predicted_labels = knn_classifier.train_and_predict(best_k, X, Y, X_test, Y_test)
    print(f"3. Evaluate on test set with best k = {best_k}")
    print(f"  {count_error_test}/{Y_test.shape[0]}")
    print(f"  Error rate: {count_error_test / Y_test.shape[0]} = {count_error_test*100 / Y_test.shape[0]}%")

    # Save predicted labels on test set to file
    df = pd.DataFrame({"true label": Y_test,
                       "predicted_label": predicted_labels})
    df.to_csv("predicted_labels.csv", index=False)
    print(f"save file successfully!\n")
    

    # # ===================== Averaged error over all training data points for each k ===================
    print("4. Report average error over all training data:")
    error_rate = knn_classifier.average_error_on_training(X, Y)
    df = pd.DataFrame({"k": ks,
                       "error rate (%)": np.round(error_rate, 4) * 100})
    df.to_csv("training_error.csv", index=False)
    print(f"save file successfully!\n")

    # =================================================================================================
    # ============================================== PCA ==============================================
    # =================================================================================================
    print("\n\n ** PCA and KNN **")
    print("1. PCA then 5-fold cross validation to select the best k")
    error_rates = knn_classifier.pca_and_kf_choose_k(X, Y)
    
    print("2. Report average validation error:")
    df = pd.DataFrame(np.round(error_rates, 4) * 100)
    df.insert(0, "n_components", n_components)
    os.makedirs("pca", exist_ok=True)
    df.to_csv("pca/average_error_val_pca.csv", header=["n_components"] + [f"k = {k}" for k in ks], index=False)
    print("save file successfully!")

    # Evaluate on test set with best k
    best_component_pca_idx, best_k_pca_idx = np.where(error_rates == np.min(error_rates))
    best_component_pca, best_k_pca = n_components[best_component_pca_idx.item()], ks[best_k_pca_idx.item()]
    count_error_test, predicted_labels = knn_classifier.train_and_predict(best_k_pca, X, Y, X_test, Y_test,
                                                                          n_component=best_component_pca)
    
    print(f"\n3. Evaluate on test set with best k = {best_k_pca} and best component pca = {best_component_pca}")
    print(f"  {count_error_test}/{Y_test.shape[0]}")
    print(f"  Error rate: {count_error_test / Y_test.shape[0]} = {count_error_test * 100 / Y_test.shape[0]}%")
    
    df = pd.DataFrame({"true label": Y_test,
                       "predicted_label": predicted_labels})
    df.to_csv("pca/predicted_labels_pca.csv", index=False)
    print("save file successfully!")


    # Training set only
    print("4. Report average error over all training data:")
    error_rate_pca = []
    for n_component in n_components:
        error_rate = knn_classifier.average_error_on_training(X, Y, n_component)
        error_rate_pca.append(error_rate)
    
    df = pd.DataFrame(np.round(error_rate_pca, 4) * 100)
    df.insert(0, "n_components", n_components)
    df.to_csv("pca/average_error_training_pca.csv", header=["n_components"] + [f"k = {k}" for k in ks], index=False)
    print("save file successfully!\n")

        