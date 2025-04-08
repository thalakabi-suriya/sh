1.
def edit_min_dis(s1, s2):
  n = len(s1)
  m = len(s2)

  dp = [[0 for _ in range(m+1)] for _ in range(n+1)]

  for j in range(m+1):
    dp[0][j] = j

  for i in range(n+1):
    dp[i][0] = i

  for i in range(1,n+1):
    for j in range(1,m+1):
      if s1[i-1] == s2[j-1]:
        dp[i][j] = dp[i-1][j-1]
      else:
        dp[i][j] = 1 + min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])

  return dp[n][m]

print(edit_min_dis("saturday","sunday"))
output:3

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

2)stemming
def stemming(s):
  if s.endswith('ing'):
    return s[0:-3]
  if s.endswith('ly') or s.endswith('ed'):
    return s[0:-2]
  return s

def lemming(s):
  lemmas = {
      'programming' : 'program' ,
      'loving' : 'Love' ,
      'lovely' : 'Love' ,
      'kind' : 'kind'
  }
  return lemmas.get(s.lower(),s)

for word in ['programming','loving','lovely','kind']:
  print(lemming(word))
  print(stemming(word))
output
program
programm
Love
lov
Love
love
kind
kind

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
3)regex
import re

text = """
Emails: sunil123@gmail.com, shyam.office@college.edu, contact_me@domain.org
Phones: 9876543210, 9123456789
"""
names = ["Sunil", "Shyam", "Ankit", "Surjeet", "Sumit", "Subhi", "Surbhi", "Siddharth", "Sujan"]
abc = "a ab abc abcc abccc abb abcd"

email_pattern = r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b'
phone_pattern = r'\b\d{10}\b'
su_pattern = r'\bSu\w{3}\b'
ab_pattern = r'ab(?:c)*'       # ab + 0 or more c's
a_pattern = r'a(?:bc)*'        # a + 0 or more 'bc'
an_pattern = r'ab(?:c)?'       # ab + 0 or 1 c


print(re.findall(email_pattern,text))
print(re.findall(phone_pattern,text))
print(re.findall(su_pattern,str(names)))
print(re.findall(ab_pattern,abc))
print(re.findall(a_pattern,abc))
print(re.findall(an_pattern,abc))

found = re.search(phone_pattern,text)
print(found.group() if found else "not found")

matched = re.match(r'\nEmails' , text)
print(matched.group() if matched else "not found")

replaced = re.sub(phone_pattern , "xxxxxxxxxx" , text)
print(replaced)

email_complied = re.compile(email_pattern)
print(email_complied.findall(text))

output
['sunil123@gmail.com', 'shyam.office@college.edu', 'contact_me@domain.org']
['9876543210', '9123456789']
['Sunil', 'Sumit', 'Subhi', 'Sujan']
['ab', 'abc', 'abcc', 'abccc', 'ab', 'abc']
['a', 'a', 'abc', 'abc', 'abc', 'a', 'abc']
['ab', 'abc', 'abc', 'abc', 'ab', 'abc']
9876543210

Emails

Emails: sunil123@gmail.com, shyam.office@college.edu, contact_me@domain.org
Phones: xxxxxxxxxx, xxxxxxxxxx

['sunil123@gmail.com', 'shyam.office@college.edu', 'contact_me@domain.org']

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

4)trigram
from collections import defaultdict
corpus = [
    "<s> I am Henry </s>",
    "<s> I like college </s>",
    "<s> Do Henry like college </s>",
    "<s> Henry I am </s>",
    "<s> Do I like Henry </s>",
    "<s> Do I like college </s>",
    "<s> I do like Henry </s>"
]

trigram_counts = defaultdict(int)

for sentence in corpus:
  words = sentence.replace('<s>' , "").replace('</s>','').strip().split()
  for i in range(len(words)-2):
    trigram = (words[i],words[i+1],words[i+2])
    trigram_counts[trigram] += 1

next_word_count = defaultdict(int)
for sentence in corpus:
  words = sentence.replace('<s>' , "").replace('</s>','').strip().split()
  for i in range(len(words)-3):
    if words[i] == "Do" and words[i+1] == "I" and words[i+2] == "like":
      next = words[i+3]
      next_word_count[next] += 1

print(max(next_word_count, key=next_word_count.get))

output
Henry

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
5)prepositions
sentences = [
    "I need a flight from Atlanta.",
    "Everything to permit us.",
    "I would like to address the public on this issue.",
    "We need your shipping address."
]


for sentence in sentences:
  words = sentence.strip().split()
  for word in words:
    w = word.lower()
    if w in ['i' ,'us','we','your']:
      print(f'{w} (PRP) ' , end=" ")
    elif w in ['need' , 'permit' , 'would', 'like','shipping']:
      print(f"{w} (VBP) " , end= " ")
    elif w in ['a','the', 'this']:
      print(f'{w}  (DT) ', end= " ")
    elif w in ["to"]:
      print(f'{w} (TO) ', end= " ")
    elif w in ['flight' , 'atlanta' , 'everything' ,'address','public','issue']:
      print(f'{w} (NN) ' , end = " ")
    elif w in ['from' , 'in' , 'on']:
      print(f'{w} (IN) ' , end = " ")
  print()
output
i (PRP)  need (VBP)  a  (DT)  flight (NN)  from (IN)  
everything (NN)  to (TO)  permit (VBP)  
i (PRP)  would (VBP)  like (VBP)  to (TO)  address (NN)  the  (DT)  public (NN)  on (IN)  this  (DT)  
we (PRP)  need (VBP)  your (PRP)  shipping (VBP)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
6)text summarixation
text = """
Artificial Intelligence (AI) has significantly transformed how we interact with technology.
From voice assistants like Siri and Alexa to recommendation systems on Netflix and YouTube, AI is all around us.
It enables machines to learn from data and improve over time without being explicitly programmed.
In healthcare, AI is helping doctors diagnose diseases more accurately and quickly.
In transportation, self-driving cars use AI to navigate and avoid obstacles.
Businesses use AI to automate repetitive tasks and gain insights from large datasets.
In education, AI-powered tutors assist students with personalized learning.
However, AI also poses ethical challenges such as job displacement, data privacy, and algorithmic bias.
As AI continues to evolve, it's important for society to strike a balance between innovation and regulation.
Responsible development and use of AI will shape the future of technology and humanity.
"""

import re
from collections import Counter

# Clean and split text into sentences
sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

# Tokenize words
words = re.findall(r'\b\w+\b', text.lower())
stopwords = set(["the", "is", "has", "a", "and", "to", "of", "in", "with", "for", "on", "as", "it", "we", "how", "from", "be", "are", "by", "this", "that", "such", "will"])

# Filter stopwords
filtered_words = [word for word in words if word not in stopwords]

# Word frequency
word_freq = Counter(filtered_words)

# Score sentences
sentence_scores = {}
for sent in sentences:
    score = 0
    for word in re.findall(r'\b\w+\b', sent.lower()):
        if word in word_freq:
            score += word_freq[word]
    sentence_scores[sent] = score

# Top 3 sentences
top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
extractive_summary = '. '.join(top_sentences) + '.'

print("📌 Extractive Summary:\n", extractive_summary)
output
📌 Extractive Summary:
 From voice assistants like Siri and Alexa to recommendation systems on Netflix and YouTube, AI is all around us. However, AI also poses ethical challenges such as job displacement, data privacy, and algorithmic bias. Businesses use AI to automate repetitive tasks and gain insights from large datasets.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

7)parsw-bear
# Grammar rules (as per your question)
grammar = {
    "S": [["NP", "VP"]],
    "NP": [["Det", "Nom"]],
    "VP": [["V", "NP"]],
    "Nom": [["Adj", "Nom"], ["N"]],
    "Det": [["the"]],
    "Adj": [["angry"], ["frightened"], ["little"]],
    "N": [["bear"], ["squirrel"]],
    "V": [["chased"]]
}

# Input sentence
tokens = ["the", "angry", "bear", "chased", "the", "frightened", "little", "squirrel"]
index = 0  # pointer to current token


def parse(symbol):
    global index

    if symbol not in grammar:
        # Terminal: match the token
        if index < len(tokens) and tokens[index] == symbol:
            node = (symbol, tokens[index])
            index += 1
            return node
        else:
            return None

    for rule in grammar[symbol]:
        saved_index = index
        children = []
        for part in rule:
            child = parse(part)
            if child is None:
                index = saved_index  # backtrack
                break
            children.append(child)
        else:
            return (symbol, children)

    return None


# Pretty print the parse tree
def print_tree(node, indent=0):
    if isinstance(node, tuple):
        label, children = node
        print("  " * indent + str(label))
        if isinstance(children, list):
            for child in children:
                print_tree(child, indent + 1)
        else:
            print("  " * (indent + 1) + str(children))


# Parse the sentence starting from S
tree = parse("S")
if tree:
    print("Parse Tree:")
    print_tree(tree)
else:
    print("Parsing failed.")
output
Parse Tree:
S
  NP
    Det
      the
        the
    Nom
      Adj
        angry
          angry
      Nom
        N
          bear
            bear
  VP
    V
      chased
        chased
    NP
      Det
        the
          the
      Nom
        Adj
          frightened
            frightened
        Nom
          Adj
            little
              little
          Nom
            N
              squirrel
                squirrel

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

8)parse -diagram
# Grammar definition
grammar = {
    "S": [["NP", "VP"]],
    "NP": [["Det", "Nominal"], ["N"]],
    "Nominal": [["N"]],
    "VP": [["V", "NP"]],
    "Det": [["A"]],
    "N": [["Restaurant"], ["Dosa"]],
    "V": [["Serves"]]
}

# Input sentence
tokens = ["A", "Restaurant", "Serves", "Dosa"]
index = 0  # global token pointer


def parse(symbol):
    global index

    if symbol not in grammar:
        # Terminal: match token
        if index < len(tokens) and tokens[index] == symbol:
            node = (symbol, tokens[index])
            index += 1
            return node
        else:
            return None

    for rule in grammar[symbol]:
        saved_index = index
        children = []
        for part in rule:
            child = parse(part)
            if child is None:
                index = saved_index  # backtrack
                break
            children.append(child)
        else:
            return (symbol, children)

    return None


# Pretty-print the parse tree
def print_tree(node, indent=0):
    if isinstance(node, tuple):
        label, children = node
        print("  " * indent + str(label))
        if isinstance(children, list):
            for child in children:
                print_tree(child, indent + 1)
        else:
            print("  " * (indent + 1) + str(children))


# Start parsing from 'S'
tree = parse("S")
if tree:
    print("Parse Tree:")
    print_tree(tree)
else:
    print("Parsing failed.")
output
Parse Tree:
S
  NP
    Det
      A
        A
    Nominal
      N
        Restaurant
          Restaurant
  VP
    V
      Serves
        Serves
    NP
      N
        Dosa
          Dosa


import nltk
from nltk.tree import Tree

# Build the parse tree manually based on the grammar and sentence structure
parse_tree = Tree('S', [
    Tree('NP', [
        Tree('Det', ['the']),
        Tree('Nom', [
            Tree('Adj', ['angry']),
            Tree('Nom', [
                Tree('N', ['bear'])
            ])
        ])
    ]),
    Tree('VP', [
        Tree('V', ['chased']),
        Tree('NP', [
            Tree('Det', ['the']),
            Tree('Nom', [
                Tree('Adj', ['frightened']),
                Tree('Nom', [
                    Tree('Adj', ['little']),
                    Tree('Nom', [
                        Tree('N', ['squirrel'])
                    ])
                ])
            ])
        ])
    ])
])

# Pretty-print the parse tree
parse_tree.pretty_print()


paragraph = """
Artificial Intelligence (AI) has rapidly evolved over the past decade, becoming a significant force across industries.
From healthcare to finance, AI systems are revolutionizing how tasks are performed.
In medicine, AI assists in early detection of diseases and personalized treatments.
In the automotive sector, self-driving technology powered by AI has made tremendous strides.
AI chatbots now handle customer service queries 24/7 with impressive accuracy.
However, with great power comes great responsibility.
There are concerns about privacy, ethics, and job displacement caused by AI systems.
Regulation is essential to ensure AI is used fairly and transparently.
Education systems are adapting to teach AI concepts at an early age.
Looking ahead, AI is expected to become more human-like, capable of understanding emotions and context.
But it’s crucial that its development remains aligned with human values and benefit to society.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import heapq

# # Download required resources
# nltk.download('punkt')
# nltk.download('stopwords')

# Tokenize into sentences
sentences = sent_tokenize(paragraph)
words = word_tokenize(paragraph.lower())

# Filter stopwords and punctuation
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

# Frequency table
word_freq = defaultdict(int)
for word in filtered_words:
    word_freq[word] += 1

# Sentence scores
sent_scores = defaultdict(int)
for sent in sentences:
    for word in word_tokenize(sent.lower()):
        if word in word_freq:
            sent_scores[sent] += word_freq[word]

# Extract top 3 scored sentences
summary_sentences = heapq.nlargest(3, sent_scores, key=sent_scores.get)
summary = " ".join(summary_sentences)

print("🔹 Extraction-based Summary:\n", summary)

lemma : 

from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

from nltk.stem import WordNetLemmatizer

from textblob import Word

from tabulate import tabulate

porter_stemmer = PorterStemmer()

lancaster_stemmer = LancasterStemmer()

snowball_stemmer = SnowballStemmer("english")

lemmatizer = WordNetLemmatizer()

words = [

 "Run", "Running", "Runs", "Ran", "Runner",

 "Runningly", "Rerun", "Overrunning", "Unrunnable",
  "Outran", "Misrun"

]

results = []

for word in words:

 lower_word = word.lower()

 results.append([

 word,

 porter_stemmer.stem(lower_word),

 lancaster_stemmer.stem(lower_word),

 snowball_stemmer.stem(lower_word),

 lemmatizer.lemmatize(lower_word, pos='v'), # Lemmatization as verb

 lemmatizer.lemmatize(lower_word, pos='n'), # Lemmatization as noun

 Word(lower_word).lemmatize("v"), # TextBlob lemmatization (verb)

 Word(lower_word).lemmatize("n") # TextBlob lemmatization (noun)

 ])

headers = ["Original", "Porter", "Lancaster", "Snowball", "WN Lemma (V)", "WN Lemma (N)", 

"TextBlob (V)", "TextBlob (N)"]

print(tabulate(results, headers=headers, tablefmt="grid"))


import nltk

from nltk.stem import SnowballStemmer, LancasterStemmer, RegexpStemmer, 

WordNetLemmatizer

nltk.download('wordnet')

nltk.download('omw-1.4')

nltk.download('punkt')

words = ["jumping", "happier", "easily", "babies", "running", "mice", "teeth", "caring", 

"stronger"]

snowball = SnowballStemmer("english")

lancaster = LancasterStemmer()

regexp = RegexpStemmer(r"ing$|ed$|es$", min=4)

lemmatizer = WordNetLemmatizer()

snowball_stems = [snowball.stem(word) for word in words]

lancaster_stems = [lancaster.stem(word) for word in words]regexp_stems = [regexp.stem(word) for word in words] 

lemmatized_nouns = [lemmatizer.lemmatize(word, pos='n') for word in words]

lemmatized_verbs = [lemmatizer.lemmatize(word, pos='v') for word in words]

lemmatized_adjectives = [lemmatizer.lemmatize(word, pos='a') for word in words]

print(f"{'Word':<12}{'Snowball':<12}{'Lancaster':<12}{'Regexp':<12}{'Lemma 

(Noun)':<15}{'Lemma (Verb)':<15}{'Lemma (Adj.)':<15}")

print("-" * 95)

for i in range(len(words)):

print(f"{words[i]:<12}{snowball_stems[i]:<12}{lancaster_stems[i]:<12}{regexp_stem

s[i]:<12}{lemmatized_nouns[i]:<15}{lemmatized_verbs[i]:<15}{lemmatized_adjectives

[i]:<15}")

parse tree :

import nltk

from nltk import CFG

import matplotlib.pyplot as plt

from nltk.tree import Tree

grammar = CFG.fromstring("""

 S -> NP VP

 NP -> Det N

 VP -> V NP

 Det -> 'the'

 N -> 'dog' | 'cat'

 V -> 'chased' | 'saw'

""")

parser = nltk.ChartParser(grammar)

sentence = ['the', 'cat', 'saw', 'the', 'dog']

for tree in parser.parse(sentence):

 print(tree.pretty_print())



# Define the grammar rules properly
grammar = {
    'S': [['NP', 'VP']],
    'NP': [['Det', 'Nom']],
    'VP': [['V', 'NP']],
    'Nom': [['Adj', 'Nom'], ['N']],
    'Det': [['the']],
    'Adj': [['little'], ['angry'], ['frightened']],
    'N': [['squirrel'], ['bear']],
    'V': [['chased']]
}

# Tokenized input sentence
sentence = "the angry bear chased the frightened little squirrel".split()
index = 0  # Global index to track the current word


# Recursive parse function
def parse(symbol):
    global index
    # Terminal
    if symbol not in grammar:
        if index < len(sentence) and sentence[index] == symbol:
            node = (symbol,)
            index += 1
            return node
        else:
            return None
    # Non-terminal
    for rule in grammar[symbol]:
        saved_index = index
        children = []
        for part in rule:
            result = parse(part)
            if result is None:
                index = saved_index
                break
            children.append(result)
        else:
            return (symbol, children)
    return None


# Helper to print parse tree
def print_tree(node, indent=0):
    if isinstance(node, tuple):
        if len(node) == 1:
            print('  ' * indent + node[0])
        else:
            print('  ' * indent + node[0])
            for child in node[1]:
                print_tree(child, indent + 1)


# Start parsing from 'S'
tree = parse('S')

if tree and index == len(sentence):
    print("✅ Parse successful! Here's the parse tree:\n")
    print_tree(tree)
else:
    print("❌ Failed to parse the sentence.")



------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
