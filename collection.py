import knn
import naivebayes
import lin_class
import mlp
import cnn

def run_tests():
    results={}
    run_test(results, 'K-Nearest Neigbor', knn.run_default)
    run_test(results, 'Naive Bayes', naivebayes.run_default)
    run_test(results, 'Linear Classifier', lin_class.run_default)
    run_test(results, 'Multilayer Perceptron', mlp.run_default)
    run_test(results, 'Convolutional Neural Net', cnn.run_default)
    print("Rankings: ")
    rank=1
    for name, accuracy in sorted(results.items(), key=lambda x : x[1], reverse=True):
        print(f'{rank}. {name}: {accuracy:.2f}% Accurate')
        rank+=1


def run_test(results, label, func):
    print(f'Running {label}...')
    results[label]=func()
    print(f'Done! {label} Accuracy: {results[label]}')

run_tests()