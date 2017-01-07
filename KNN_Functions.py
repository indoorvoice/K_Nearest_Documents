
def read_mmtx_labels(filename):
    """Reads a file of two comma-separated, 'document: label' pairs into a dictionary."""
    label_dict = {} # dict to store the labels of documents

    for line in open(filename, 'r'):
        # remove newline, split on comma and turn into list. assign the elements to the relevant variable
        line_elements = (line.replace('\n', '')).split(',')
        document = line_elements[0]
        label = line_elements[1]

        label_dict[document] = label # create entry in dict for this pairing

    return label_dict


def read_mmtx(filename, skiplines=2):
    """Reads sparse, market matrix structures into a dictionary of nested dictionaries.

    Intended to parse the structure: Document, Term, Frequency
    Can skip unneeded lines. Defaults to 2 lines skipped."""
    document_dict = {} # will hold each document's contents as a dict of {word: frequency} pairs

    for index, line in enumerate(open(filename, 'r')): # use enumerate to skip lines without needed data
        if index < skiplines:
            continue

        line_elements = (line.replace('\n', '')).split(' ') # strip newline character and split on whitespace
        document = line_elements[0] # parse out the individual elements
        word = line_elements[1]
        freq = int(line_elements[2])

        # construct dictionary-of-dictionaries, each inner dictionary is a document with key, value of word; frequency
        if document in document_dict:
            document_dict[document][word] = freq
        else:
            document_dict[document] = {} # if first entry, create dictionary to hold that document's words
            document_dict[document][word] = freq # assign that new dictionary the current line's word; frequency pair

    return document_dict


def dot_product(corpus_dict, doc1, doc2):
    """Computes dot product of two dictionaries nested within the same dictionary

    Finds mutual keys of dictionaries and sums the product of these pairs."""

    dot_product = 0 # initialise dot product
    # convert each document's keys to set to find the common words between them
    key_intersection = set(corpus_dict[doc1].keys()) & set(corpus_dict[doc2].keys())

    # for each common word, get the product of each document's frequency. Sum values as you go.
    for key in key_intersection:
        dot_product += corpus_dict[doc1][key] * corpus_dict[doc2][key]
    return dot_product


def vector_norm(doc):
    """Computes vector norm for the values of a dictionary"""

    # square all dictionary's values, then sum them. Finally, take the square root of the summed values.
    vector_norm_value = (sum([val**2 for val in doc.values()]))**0.5

    return vector_norm_value

def cosine_similarity(corpus_dict, vec_norms, doc1, doc2):
    """Returns the Cosine Similarity of two dictionaries of values. Must be nested in same dictionary."""

    # Call each function, return the ratio between them
    return dot_product(corpus_dict, doc1, doc2) / (vec_norms[doc1] * vec_norms[doc2])


def vote(similarity_dict, list_of_nearest_neighbours, label_dict, query_doc, weighted):
    """Takes a list of neighbour documents, determines a vote for each and sums them by label.

    Defaults to unweighted."""
    tally = {} # dictionary to hold the votes of each label

    for neighbour in list_of_nearest_neighbours:
        label = label_dict[neighbour] # find the label for each numeric neighbour

        if weighted: # if weighted, set the vote_value to be the inverse distance
            vote_value = (1 / get_distance_between(similarity_dict, neighbour, query_doc))
        else: # if not weighted, set it to be 1
            vote_value = 1

        if label not in tally: # if it's the first of that label, set the dict value
            tally[label] = vote_value
        else: # otherwise, add the current vote to the already initialised value
            tally[label] += vote_value
    else:
        winner = max(tally, key=tally.get) # return the key which has the highest associated value
        return winner

def k_nearest_neighbours(doc_dict, similarity_dict, vector_norms, query_doc, k):
    """Returns a list of the K-nearest neighbours, defined by cosine similarity.

    Stores all computations in a persistent global dictionary, specified in signature.
    If distance already calculated, references distance from that dictionary instead of recalculating.
    """

    local_similarity_dict = {} # local dictionary stores all results, is used for second section
    for doc in doc_dict.keys(): # go through all the documents in the training set
        if doc == query_doc:
            continue # skip itself if it encounters itself
        else:
            # check if the distance has already been computed in dict[a][b], dict[b][a] format
            # if so, read from the persistent similarity dict
            if doc in similarity_dict and query_doc in similarity_dict[doc]:
                local_similarity_dict[doc] = similarity_dict[doc][query_doc]
            elif query_doc in similarity_dict and doc in similarity_dict[query_doc]:
                local_similarity_dict[doc] = similarity_dict[query_doc][doc]
            else:
                # if new, compute the similarity, add to local dict
                this_similarity = cosine_similarity(doc_dict, vector_norms, query_doc, doc)
                local_similarity_dict[doc] = this_similarity
                # and add to persistent dict as well
                if doc not in similarity_dict:
                    similarity_dict[doc] = {}
                similarity_dict[doc][query_doc] = this_similarity

    # sort the local dict by values, with highest first as we are using similarities
    k_ranked_similarites = sorted(
        local_similarity_dict.items(), key=lambda x: x[1], reverse=True)

    k_nearest_neighbours = [None]*k # initialise array of correct size

    for i in range(k): # parse the document numbers out into the new array, this tidies things up
        k_nearest_neighbours[i] = k_ranked_similarites[i][0]

    return k_nearest_neighbours


def classify_document_knn(corpus_dict, label_dict, similarity_dict, vector_norms, query_doc, k, weighted):
    """Performs KNN for a given document in a corpus, votes on the class and returns it."""

    knn_document = k_nearest_neighbours(corpus_dict, similarity_dict, vector_norms, query_doc, k)
    predicted_label = vote(similarity_dict, knn_document, label_dict, query_doc, weighted)

    return predicted_label


def get_distance_between(similarity_dict, doc1, doc2):
    """Returns the distance between two documents. Distance referenced from existing dictionary."""

    if doc1 in similarity_dict and doc2 in similarity_dict[doc1]: # check in dict[a][b]
        return similarity_dict[doc1][doc2]
    elif doc2 in similarity_dict and doc1 in similarity_dict[doc2]: # check in dict[b][a]
        return similarity_dict[doc2][doc1]
    else:
        return -1


def check_correct_prediction(query_doc, label_dict, predicted_label):
    """Checks if a label is equal to that of a specified item in a dictionary. Returns 1 if true."""
    if label_dict[query_doc] == predicted_label:
        return 1
    else:
        return 0



def leave_one_out_cross_validate(corpus_dict, label_dict, similarity_dict, vector_norms, weighted=False,
                                 lower=1, upper=10):
    """Performs leave-one-out Cross Validation over a corpus of documents, printing output throughout.

    Takes advantage of precomputed results in all runs after the first."""

    print("""Beginning Cross Validation.
Please allow approximately 40 seconds for the first result.
All subsequent results should take approximately 3 seconds each.
    """)
    best = 0 # stores the highest performing value of k
    for k in range(lower, upper+1):
        correct, total = 0, 0
        for query_doc in corpus_dict.keys(): # for each doc in corpus, predict class, check if correct
            predicted_class = classify_document_knn(corpus_dict, label_dict, similarity_dict,
                                                    vector_norms, query_doc, k, weighted)
            correct += check_correct_prediction(query_doc, label_dict,predicted_class)

            total += 1 # keep track of prediction count

        accuracy = correct/total # ratio of correct to total gives accuracy
        if accuracy > best:
            best = k
        print(k, 'NN Accuracy:', round(accuracy* 100, 7), '%')

    else: # once the for loop is exhausted, print the best performing value of k
        print('Highest Accuracy of K:', best)