## If we want to generate files :
### Steps to execute the code : 
1. Uncommemnt lines 174-194
2. Comment lines 196-208
3. On terminal use commandline:
    python3 neuralLanguageModelling.py

## If we want to give user input :
### Steps to execute the code : 
1. On terminal use commandline:
    python3 neuralLanguageModelling.py

## Files generated

#### Neural LM : 
1. 2018101112-LM1-train-perplexity.txt - It contains sentence wise perplexity and average perplexity score of all sentences in training set of corpus.
2. 2018101112-LM1-validation-perplexity.txt - It contains sentence wise perplexity and average perplexity score of all sentences validation set of corpus.
3. 2018101112-LM1-test-perplexity.txt - It contains sentence wise perplexity and average perplexity score of all sentences test set of corpus.

#### Statistical n-gram LM (Kneser Ney) :
1. 2018101112-LM2-train-perplexity.txt - It contains sentence wise perplexity and average perplexity score of all sentences in training set of corpus.
2. 2018101112-LM2-validation-perplexity.txt - It contains sentence wise perplexity and average perplexity score of all sentences validation set of corpus.
3. 2018101112-LM2-test-perplexity.txt - It contains sentence wise perplexity and average perplexity score of all sentences test set of corpus.

#### Statistical n-gram LM (Witten Bell) :
1. 2018101112-LM3-train-perplexity.txt - It contains sentence wise perplexity and average perplexity score of all sentences in training set of corpus.
2. 2018101112-LM3-validation-perplexity.txt - It contains sentence wise perplexity and average perplexity score of all sentences validation set of corpus.
3. 2018101112-LM3-test-perplexity.txt - It contains sentence wise perplexity and average perplexity score of all sentences test set of corpus.


## Perplexity

Perplexity is calculated in the following:
1. The corpus is converted to sentences. 
2. Sentences are divided into test set, validation and training set after using shuffle of random library.
2. Then the language model is created on training and validatoion set.
3. Then each sentence in all the 3 sets are evaluated and the above text files have been created.
4. At last the average perplexity score is put in the file.
5. Perplexity is calculated using the following formula : 
    1. float(1)/float(math.exp(float(probability)/float(n)))
    2. Here probability = probablity of each sentence in the test set.
        1. Probability of each sentence is calculated by the formula exp(math.log(p1) + math.log(p2) + math.log(p3) + .... + math.log(pN)) 
    3. Here n =  number of tokens in sentance after preprocessing after preprocessing .