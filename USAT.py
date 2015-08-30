# -*- coding: utf-8 -*-

from pprint import pprint
import nltk
import yaml
import sys
import os
import re
import math
from lxml import html
import requests
import plotly.plotly as py

from plotly.graph_objs import *

class Splitter(object):

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences


class POSTagger(object):

    def __init__(self):
        pass
        
    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos

class DictionaryTagger(object):

    def __init__(self, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        """
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    #self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0

def sentence_score(sentence_tokens, previous_token, acum_score):    
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                token_score *= -1.0
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

def sentiment_score(review):
	for sentence in review:
		print sentence 
		print sentence_score(sentence, None, 0.0)
	return sum([sentence_score(sentence, None, 0.0) for sentence in review])
	
def mysentence_score(sentence):
	return (sentence_score(sentence, None, 0.0))	
#pass URL of comments... as of now 'review-text-full' hard coded  
def GetReviewComments(hostURL,reviewText):
	page = requests.get(hostURL)
	tree = html.fromstring(page.text)
	reviews = tree.xpath('//span[@class="review-text-full"]/text()')
	total = len(reviews)
	print "Total Number of row for processing" + str(total)
	return reviews


if __name__ == "__main__":
	splitter = Splitter()
	postagger = POSTagger()
	dicttagger = DictionaryTagger([ 'D:/Code/Python/USAT/positive.yml', 'D:/Code/Python/USAT/negative.yml', 'D:/Code/Python/USAT/inc.yml', 'D:/Code/Python/USAT/dec.yml', 'D:/Code/Python/USAT/inv.yml'])

	modules = ["battery~charger~charging", "ram", "screen~display", "wifi", "camera", "ui", "speaker~music~audio", "os~ios~android"]
	
	if(len(sys.argv) > 1):
		url = str(sys.argv[1])
		print url
	else:
		url = "http://www.flipkart.com/redmi-2-prime/p/itme9t7hgvuepqm7?pid=MOBE9T7GTHERTDAC&ref=L%3A5997859224050682229&srno=p_5&query=redmi2&otracker=from-search"

		
	reviews = GetReviewComments(url,"review-text-full")
	
	ind = 0
	mod_arr = []
	while (ind < len(modules)):
		mod_arr.append(modules[ind])
		ind = ind + 1
		
	#index 0 for +ve,index 1 for -ve,and index 2 for neutral
	semtimentArray = [[0 for y in xrange(3)] for x in xrange(10)]
	for i in xrange(len(reviews)):
		if i>=0:
			text = reviews[i].lower()
#			print text
			splitted_sentences = splitter.split(text)

			pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

			dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)

		idx = 0
		for idx in xrange(len(mod_arr)):
			for sentence in dict_tagged_sentences:
				
				splitStrs = mod_arr[idx].split("~")

				for splitStr in splitStrs:
					if  str(sentence).find(splitStr) >=0:
						score = mysentence_score(sentence)
						if score>0:
							semtimentArray[idx][0] = semtimentArray[idx][0] + 1 #score
						elif score<0:
							semtimentArray[idx][1] = semtimentArray[idx][1] + 1 #math.copysign(score,+0.0)
						else:
							semtimentArray[idx][2] = semtimentArray[idx][2] + 1 #score
						break
	
	num_of_module = 0
	i = 0
	for i in xrange(len(mod_arr)):
		num_of_module = num_of_module + 1
		print str(mod_arr[i])
		iTotal = 0
		j = 0
		for j in range(len(semtimentArray[i])):
			iTotal = iTotal + semtimentArray[i][j]
		
		j = 0
		for j in range(len(semtimentArray[i])):
			if (iTotal > 0):
				semtimentArray[i][j] = round((semtimentArray[i][j] / float(iTotal)) * 100)
				
			else:
				semtimentArray[i][j] = 0
			print semtimentArray[i][j]

			
	# Plot the results in graph - Start
	
	top_labels = ['Good', 'Bad', 'Neutral']

	colors = ['rgba(0, 128, 0, 0.9)', 'rgba(255, 0, 0, 0.9)', 'rgba(0, 0, 255, 0.9)']

	x_data = [[]]
	y_data = []

	traces = []

	j = 0
	while (j < num_of_module):
		temp = str(mod_arr[j])
		temp2 = temp.split("~")
		y_data.append(temp2[0])
		j = j + 1

	j = 0
	while (j < num_of_module):
		x_data[j].append(semtimentArray[j][0])
		x_data[j].append(semtimentArray[j][1])
		x_data[j].append(semtimentArray[j][2])
		j = j + 1
		if(j < num_of_module):
			x_data.append([])

		
	for i in range(0, len(x_data[0])):
		for xd, yd in zip(x_data, y_data):
			traces.append(Bar(
			x=xd[i],
			y=yd,
			orientation='h',
			marker=Marker(
					color=colors[i],
					line=Line(
							color='rgb(0, 0, 0)',
							width=1,
					)
				)
			))

	layout = Layout(
		xaxis=XAxis(
			showgrid=False,
			showline=False,
			showticklabels=False,
			zeroline=False,
			domain=[0.15, 1]
		),
		yaxis=YAxis(
			showgrid=False,
			showline=False,
			showticklabels=False,
			zeroline=False,
		),
		barmode='stack',
		title='User Satisfaction Review',
		height=600,
		width=600,
		paper_bgcolor='rgb(248, 248, 255)',
		plot_bgcolor='rgb(248, 248, 255)',
		margin=Margin(
			l=120,
			r=10,
			t=140,
			b=80
		),
		showlegend=False,
	)

	annotations = []

	for yd, xd in zip(y_data, x_data):
		# labeling the y-axis
		annotations.append(Annotation(xref='paper', yref='y', x=0.14, y=yd,
									  xanchor='right', text=str(yd),
									  font=Font(family='Arial', size=14,
												color='rgb(67, 67, 67)'),
									  showarrow=False, align='right'))
		# labeling the first percentage of each bar (x_axis)
		annotations.append(Annotation(xref='x', yref='y', x=(xd[0] / 2), y=yd,
									  text=str(xd[0]) + '%',
									  font=Font(family='Arial', size=14,
												color='rgb(248, 248, 255)'),
									  showarrow=False))
		# labeling the first Likert scale (on the top)
		if yd == y_data[-1]:
			annotations.append(Annotation(xref='x', yref='paper', x=(xd[0] / 2),
										  y=1.1, text="",
										  font=Font(family='Arial', size=14,
										  color='rgb(67, 67, 67)'),
										  showarrow=False))
		space = xd[0]
		for i in range(1, len(xd)):
				# labeling the rest of percentages for each bar (x_axis)
				annotations.append(Annotation(xref='x', yref='y', x=space
											  + (xd[i] / 2), y=yd,
											  text=str(xd[i]) + '%',
											  font=Font(family='Arial', size=14,
														color='rgb(248, 248, 255)'),
											  showarrow=False))
				# labeling the Likert scale
				if yd == y_data[-1]:
					annotations.append(Annotation(xref='x', yref='paper', x=space +
												  (xd[i] / 2), y=1.1,
												  text="",
												  font=Font(family='Arial', size=14,
															color='rgb(67, 67, 67)'),
												  showarrow=False))
				space += xd[i]

	layout['annotations'] = annotations

	fig = Figure(data=traces, layout=layout)

	plot_url = py.plot(fig, filename='bar-colorscale')
	
	# Plot the results in graph - End	