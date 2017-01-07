from KNN_Functions import *


data_input = input("""Please choose a method of data input:
1. Attempt to read test dataset from local directory (recommended)
2. Input document and label data files directly
""")
if data_input == '1':
    data_file = 'news_articles.mtx'
    label_file = 'news_articles.labels'
elif data_input == '2':
    data_file = input(
        'Please enter the name of the file containing your matrix market data: '
    )

    label_file = input(
        'Please enter the name of the file containing the labels for your previous data: '
    )
else:
    print('Error: Not valid choice. Please start again.')
    raise SystemExit

# setup phase
corpus_dict = read_mmtx(data_file, skiplines=2) # read in document data
corpus_labels = read_mmtx_labels(label_file) # read in label data
num_docs = (len(corpus_dict.keys())-1) # find number of documents, this is referenced in error handling
vec_norms = {} # initialise vector norm dictionary
for doc in corpus_dict.keys(): # precompute the vector norms to save on repeated computations later
    vec_norms[doc] = vector_norm(corpus_dict[doc])

similarity_dict = {} # create the global distance dictionary which preserves results
# this saves on computations, especially with the cross validation step.

classifying = True # controls outer loop to keep program returning to initial menu

while classifying:
    choice = input(
        """Would you like to:
        1: Predict the k-nearest neighbours of a specific document?
        2: Test the overall accuracy of the classifier using cross validation?
        3: Exit the program.
        Please enter the appropriate number for your choice.
        """
    )

    if choice == '1':
        repeat = True # controls inner loop so users can repeat same type of classification
        while repeat:
            print('\n') # formatting
            # take document
            doc_to_classify = input('Please enter a document to classify on: ')
            if int(doc_to_classify) < 0 or int(doc_to_classify) > num_docs: # error handling
                print('Error: Document number invalid. Is it above zero and less than', num_docs, '?')
                print('Restarting KNN process...')
                continue
            # specify k
            k = int(input('Please choose a number of neighbours to classify on: '))
            if k < 0 or k > num_docs:
                print('Error: K value invalid. Is it above zero and less than', num_docs, '?')
                print('Restarting KNN process...')
                continue
            # choose weighted or not
            weighted = input('Would you like to use a weighted KNN? Please enter y or n to select: ').lower()
            if weighted == 'y':
                weighted = True
            elif weighted == 'n':
                weighted = False
            else:
                print('Error: Please use a value or either "y" or "n"')
                print('Restarting KNN process...')
                continue
            # if all parameters set, run classification
            result = classify_document_knn(corpus_dict, corpus_labels, similarity_dict, vec_norms, doc_to_classify, k, weighted)
            print('Predicted label is', result)
            print('Actual label was', corpus_labels[doc_to_classify])
            repeat = input('Would you like to classify another document? Please answer y or n. ').lower()
            if repeat == 'y':
                repeat = True
            elif repeat == 'n':
                repeat = False
            else:
                print('Error: Please use a value or either "y" or "n"')
                break

    if choice == '2':
        print(
            """For validating accuracy, we use Leave-One-Out Cross Validation.
    You can choose to evaluate on a specific value of k, i.e. 5
    Or, you can dynamically determine the best value of by testing a range of values of k.
        """)

        repeat = True
        while repeat:
            type_of_cv = input("""Please select an option:
        1: Perform Leave-One-Out for a specific value of k.
        2: Compare performance of k over a range of values.
        """
            )

            if type_of_cv == '1': # lets users pick one value of k. simply sets lower and upper to same value.
                value_of_k = int(input('Please choose a value of k to evaluate: '))
                if value_of_k < 0 or value_of_k > num_docs: # error handling to ensure document exists
                    print('Error: K value invalid. Is it above zero and less than', num_docs, '?')
                    print('Restarting Cross Validation process...\n')
                    continue
                # call cross validation function
                leave_one_out_cross_validate(corpus_dict, corpus_labels, similarity_dict,
                                             vec_norms, lower=value_of_k, upper=value_of_k)
                repeat = input('\nWould you like to perform another validation? (y/n) ').lower()
                if repeat == 'y':
                    repeat = True
                elif repeat == 'n':
                    repeat = False
                else:
                    print('Error: Please use a value or either "y" or "n"')
                    break

            if type_of_cv == '2': # same as option one, just solicits two arguments for lower and upper
                lower_bound_k = int(input('Please choose a lower bound of k to evaluate: '))
                upper_bound_k = int(input('Please choose an upper bound of k to evaluate: '))
                # check documents are in range and that lower range is less than upper range
                if lower_bound_k < 0 or lower_bound_k > num_docs or\
                   upper_bound_k < 0 or upper_bound_k > num_docs or upper_bound_k < lower_bound_k:
                    print('Error: K values invalid. Are they above zero and less than', num_docs, '?')
                    print('Please also ensure that the lower bound is less than, or equals, the upper bound')
                    print('Restarting Cross Validation process...\n')
                    continue
                leave_one_out_cross_validate(corpus_dict, corpus_labels, similarity_dict,
                                             vec_norms, lower=lower_bound_k, upper=upper_bound_k)
                repeat = input('\nWould you like to perform another validation? (y/n)').lower()
                if repeat == 'y':
                    repeat = True
                elif repeat == 'n':
                    repeat = False
                else:
                    print('Error: Please use a value or either "y" or "n"')
                    break

    if choice == '3':
        raise SystemExit

print('\nThank you. Have a nice day!')