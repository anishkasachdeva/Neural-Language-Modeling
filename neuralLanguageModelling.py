import numpy as np
from math import exp, log
import collections
from random import shuffle
import re
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical


toSee = 0
seen = 0
basicPreprocessedText = ''
input_file = "./brown.txt"
train_file = "./train.txt"
test_file = "./test.txt"
validation_file = "./validation.txt"
trainingSetTokens = list()
tokensFrequency = collections.Counter()
word2id = {}
id2word = list()
loss = 'categorical_crossentropy'
optimizer = 'adam'
activation = 'softmax'
metrics = ['accuracy']


def preProcessWord(words):
	modifiedWordsList = list()
	modifiedWordsList.append("<")
	modifiedWordsList.append("<")
	modifiedWordsList.append("<")

	for word in words:
		while len(word) >= 2 and word[0].isalpha() == False:
			word = word[1:]
		while len(word) >= 2 and word[len(word)-1].isalpha() == False:
			word = word[:-1]
		if ( '^' in word or "'" in word ): 
			if len(word) > 1:
				changedWord = ''
				for j in word :
					if j != '^' and j != "'":
						changedWord += j 
				word = changedWord
		if len(word) == 1:
			if word != 'a' and word != 'i':
				word = '-'
		if word.isalpha() == True : 
			modifiedWordsList.append(word)
	modifiedWordsList.append(">")

	return modifiedWordsList


def createSplit(fourgramSequencesList, validationBeginner):
	xTrain = fourgramSequencesList[:validationBeginner,:-1]
	yTrain = fourgramSequencesList[:validationBeginner:,-1]
	xValidation = fourgramSequencesList[validationBeginner:,:-1]
	yValidation = fourgramSequencesList[validationBeginner:,-1]

	return xTrain, yTrain, xValidation, yValidation


def writeInFile(file,sen,a,b,c):
	file.write(sen)
	file.write(a)
	file.write(b)
	file.write(c)


def makeUnknown(tokensFrequency):
	tokensFrequency["<unkn>"] = 0
	minimumFrequency =  100
	for word in tokensFrequency:
		if tokensFrequency[word] < minimumFrequency:
			tokensFrequency["<unkn>"] += tokensFrequency[word]
			tokensFrequency[word] = 0
	return tokensFrequency


def calculatePerplexity(probabilityLogSummation, tokenSequences):
	probabilityLogSummation = probabilityLogSummation/(len(tokenSequences)-3)
	calc = 1/exp(probabilityLogSummation)
	return calc


def calculateSentenceProbability(i):
	words = i.split()
	modifiedWordsList = preProcessWord(words)
	tokenSequences = list()
	if len(modifiedWordsList) < 5:
		return -1
	for j in modifiedWordsList:
		if j in word2id:
			tokenSequences.append(word2id[j])
		else :
			tokenSequences.append(word2id["<unkn>"])
		
	probabilityLogSummation = 0.0
	for k in range(3,len(tokenSequences)):
		pred = model.predict(np.array([tokenSequences[k-3 : k]]), verbose = 0)
		probabilityLogSummation += log(pred[0][tokenSequences[k]])
	return calculatePerplexity(probabilityLogSummation, tokenSequences)


# Preprocess Data
with open(input_file,"r") as file :
	for line in file :
		basicPreprocessedText += line[:-1]
		basicPreprocessedText += " "

basicPreprocessedText = basicPreprocessedText.lower()
basicPreprocessedText = basicPreprocessedText.replace("-"," ")
allSentences = basicPreprocessedText.split('.')
shuffle(allSentences)
lengthOfSentences = len(allSentences) 


for l in range(int((0.7 + 0.1) * lengthOfSentences)):
	sen = allSentences[l]
	words = sen.split()
	modifiedWordsList = preProcessWord(words)

	if len(modifiedWordsList) > 4 :
		seen = seen + 1
		for j in modifiedWordsList:
			trainingSetTokens.append(j)
			tokensFrequency[j] = tokensFrequency[j] + 1
	if l == (int((0.7) * lengthOfSentences) - 1) :
		toSee = seen

tokensFrequency = makeUnknown(tokensFrequency)

trainingSequencesId = list()
for i in range(len(trainingSetTokens)):
	tok = trainingSetTokens[i]
	if tokensFrequency[tok] < 100:
		trainingSetTokens[i] = "<unkn>"
		tok = trainingSetTokens[i]
	if tok not in word2id:
		word2id[tok] = len(id2word)
		trainingSequencesId.append(len(id2word))
		id2word.append(tok)
	elif tok in word2id:
		trainingSequencesId.append(word2id[tok])
		
fourgramSequencesList = list()
seen = 0
validationBeginner = 0

for i in range(3,len(trainingSequencesId)):
	if trainingSequencesId[i] == word2id["<"]:
		continue
	if trainingSequencesId[i] == word2id[">"]:
		seen = seen + 1
	fourgramSequencesList.append([trainingSequencesId[i-3],trainingSequencesId[i-2],trainingSequencesId[i-1],trainingSequencesId[i]])
	if seen == toSee:
		validationBeginner = len(fourgramSequencesList)


fourgramSequencesList = np.array(fourgramSequencesList)
xTrain, yTrain, xValidation, yValidation = createSplit(fourgramSequencesList, validationBeginner)

yTrain = to_categorical(yTrain, num_classes = len(id2word)) #One hot encoding
yValidation = to_categorical(yValidation, num_classes = len(id2word))

#Building the neural model
model = models.Sequential()
model.add(layers.Embedding(len(id2word), 100, input_length = 3))
model.add(layers.LSTM(150))
model.add(layers.Dense(len(id2word), activation = activation))
print(model.summary())
model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
model.fit(xTrain, yTrain, epochs = 500, verbose = 2, validation_data = (xValidation, yValidation))


# Writing the perplexities to the files
# validationSentencesList = allSentences[int((0.7)*lengthOfSentences):int((0.7+0.1)*lengthOfSentences)]
# with open(validation_file,'a') as file:
# 	for sen in validationSentencesList:
# 		p = calculateSentenceProbability(sen)
# 		if p != -1:
# 			writeInFile(file,sen,"\t",str(p),"\n")

# testSentencesList = allSentences[int((0.7+0.1)*lengthOfSentences):]
# with open(test_file,'a') as file:
# 	for sen in testSentencesList:
# 		p = calculateSentenceProbability(sen)
# 		if p != -1:
# 			writeInFile(file,sen,"\t",str(p),"\n")

# trainSentencesList = allSentences[:int((0.7)*lengthOfSentences)]
# with open(train_file,'a') as file:
# 	for sen in trainSentencesList:
# 		p = calculateSentenceProbability(sen)
# 		if p != -1:
# 			writeInFile(file,sen,"\t",str(p),"\n")

while(1):
	print("Enter your sentence and then press the Enter Key. ")
	inputSentence = input()
	print("--------------------")
	p = calculateSentenceProbability(inputSentence)
	if p == -1:
		print("Enter a valid sentence since preprocessing of your input sentence lead to 0 tokens. ")
		print("--------------------")
	else:
		print("Preplexity of the input sentence = ",end = '')
		print(p)
		print("--------------------")
	print()