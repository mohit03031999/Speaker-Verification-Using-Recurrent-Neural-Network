'''
@author: us
'''

## split data to k folds training and testing sets
def split_data(data, k):
    trains = []
    tests = []

    fold_length = 10 // k

    for fold in range(k):
        train = {
            'samples': [],
            'labels': []
        }
        test = {
            'samples': [],
            'labels': []
        }

        for key in data:
            label = [0] * 4
            label[key] = 1

            for i in range(len(data[key])):
                spectro = data[key][i]

                ## put data into train or test based on its index and the fold number
                if i % 10 >= fold * fold_length and i % 10 < (fold+1) * fold_length:
                    test['samples'].append(spectro)
                    test['labels'].append([label] * spectro.shape[0])
                else:
                    train['samples'].append(spectro)
                    train['labels'].append([label] * spectro.shape[0])

        trains.append(train)
        tests.append(test)

    return trains, tests
