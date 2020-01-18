import spacy
from spacy.matcher import Matcher

# det noun adp - a lot of
QUANT = [[{'POS':'DET'},{'POS':'NOUN'},{'POS':'ADP'}]]
# adv adv+ - right here, over there
ADV = [[{'POS':'ADV'},{'POS':'ADV','OP':'+'}]]
# # adp det? noun+ - in December, in the kitchen
# ADP = [[{'POS':'ADP'},{'POS':'DET','OP':'?'},{'POS':'NOUN','OP':'+'}]]
# (aux part? adj)|(aux part? verb) - is not bad, is annoying
AUX = [[{'POS':'AUX'},{'POS':'PART','OP':'?'},{'POS':'ADJ'}],
       [{'POS':'AUX'},{'POS':'PART','OP':'?'},{'POS':'VERB'}]]
# det adj? noun+ a delicious meal
NP = [[{'POS': 'DET'},{'POS':'ADJ','OP':'?'},{'POS': 'NOUN','OP':'+'}]]
# part verb|aux - to do, to get
PART = [[{'POS': 'PART'},{'POS':'VERB'}],[{'POS': 'PART'},{'POS':'AUX'}]]
# (verb adj? noun)|(verb part verb) - watch tv, eat apple, like to eat
VERB =[[{'POS': 'VERB'},{'POS':'ADJ','OP':'?'},{'POS':'NOUN'}],
       [{'POS': 'PART'},{'POS':'AUX'}]]
# QUANT VERB AUX PART ADV NP
# Order by priority

nlp = spacy.load("en_core_web_sm")
# return list of range of phrases
def get_phrases(sentence):
  l = []
  spans = [] # [[start, end],[start, end]]
  temp = set()
  matcher = Matcher(nlp.vocab)
  doc = nlp(sentence)
  # QUANT VERB AUX PART ADV NP
  # Order by priority
  for _ in QUANT+VERB+AUX+PART+ADV+NP: matcher.add(sentence,None,_)
  matches = matcher(doc)
  for string, start, end in matches:
    spans.append([start,end])
  for i in spans:
    for j in spans[spans.index(i)+1:]:
      if i[0] in range(j[0],j[1]) or i[1] in range(j[0]+1,j[1]):
        if i[1]-i[0]>=j[1]-j[0]: temp.add((j[0],j[1]))
        else: temp.add((i[0],i[1]))
  for _ in temp: spans.remove([_[0],_[1]])
  # for (i,j) in ranges: l.append(str(doc[i:j]))
  index = 0
  sentence = sentence.lower().split(' ')
  for span in spans:
    while index not in range(span[0],span[1]):
      l.append(sentence[index])
      index += 1
    l.append(' '.join(sentence[span[0]:span[1]]))
    index += span[1] - span[0]
  for i in range(index,len(sentence)): l.append(sentence[i])
  return l