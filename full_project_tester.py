import knn
import naivebayes
import lin_class
import mlp
import cnn
import np_base as base


#Full tester for project - tests all model types and computes + ranks accuracies

#Main function here, run all tests
def run_tests():
    results={}
    run_neighbortests(results)
    run_test(results, 'Naive Bayes', naivebayes.run_default)
    run_test(results, 'Linear Classifier', lin_class.run_default)
    run_test(results, 'Multilayer Perceptron', mlp.run_default)
    run_test(results, 'Convolutional Neural Net', cnn.run_default)
    print("Rankings: ")
    rank=1
    for name, accuracy in sorted(results.items(), key=lambda x : x[1], reverse=True): #Sort results by score, highest first
        print(f'{rank}. {name}: {accuracy:.2f}% Accurate')
        rank+=1


def run_test(results, label, func):
    print(f'Running {label}...')
    results[label]=func()
    print(f'Done! {label} Accuracy: {results[label]}%')

def run_neighbortests(results): #kept KNN separate to keep code cleaner
    trainimg, trainlab, valimg, vallab = base.get_sets(0.2) #use same sets to more accurately compare since it's doing the same thing
    run_test(results, '1-Nearest Neighbor', lambda: knn.run(k=1, trainimg=trainimg, trainlab=trainlab, valimg=valimg, vallab=vallab))
    run_test(results, '3-Nearest Neighbor', lambda: knn.run(k=3, trainimg=trainimg, trainlab=trainlab, valimg=valimg, vallab=vallab))
    run_test(results, '5-Nearest Neighbor', lambda: knn.run(k=5, trainimg=trainimg, trainlab=trainlab, valimg=valimg, vallab=vallab))

if __name__== "__main__":
    run_tests()