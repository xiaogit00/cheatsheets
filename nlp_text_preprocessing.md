# Text Preprocessing
**Table of Content**
1. [Normalization](#normalization)
2. [POS Tagging](#pos-tagging)
3. []()
4. []()
5. [Appendix](#appendix)

## Normalization

### Stopword removal
`from nltk.corpus import stopwords` - stopwords  
`list(stopwords.words('english'))` -> list of stopwords  
```python
for i, tl in enumerate(token_lists):
    for j, token in enumerate(tl):
        if token in nltk_stopwords:
            token = ''
        tl[j] = token
    token_lists[i] = [ token for token in tl if token.strip() != '' ]
    
for token_list in token_lists:
    print(token_list)
```


### Regex cleaning
`s.lower()` - lower case  
`re.sub(r'\s+', ' ', s)` - remove duplicate whitespace  
`re.sub(r'([.]){2,}', ' ',s)` - remove ellipses  
`s = re.sub(r'([\w.-]+)([,;])([\w.-]+)', r'\g<1>\g<2> \g<3>', s)` - add missing whitespace after , and ;  
`s = re.sub(r'[\(\[].*?[\)\]]', ' ', s)` - remove words in brackets  

```python
# Typically you will have a function: normalize_sentence(s) that does word level normalization

def normalize_sentence(s):
    s = s.lower()
    s = re.sub()
    #...

for s in sentences:
    s_norm = normalize_sentence(s)

```

### Normalization after tokenization
`from nltk.tokenize import TweetTokenizer`  
`tweet_tokenizer = TweetTokenizer()`  
`token_lists = [tweet_tokenizer.tokenize(s) for s in sentences ]` -> a list of tokens
`for tl in token_lists: [token.lower()] for token in tl` -> make each token lower

## POS Tagging
Basically classifying words based on their formal linguistic rules and returning a list of tuples.  
`from nltk.tokenize import TweetTokenizer`  
`from nltk import pos_tag`  
`from nltk.help import upenn_tagset`  
`import spacy`  
`nlp = spacy.load('en_core_web_sm')`  
`from tqdm import tqdm`  

### upenn_tagset
A tagset is a way of classifying something as their part of speech. The upenn tagset defines the various classes a word can be. 

Example: 
- JJ: adjective
- NN: Noun
- RB: adverb
- VB: Verb
- IN - preposition
- . : sentence terminator: . ! ? 

For full list of tag set is defined below. 

### nltk's pos_tag
`tweet_tokenizer = TweetTokenizer()` - pos_tag requires a list of words
`pos_tag(tweet_tokenizer.tokenize('I love tea'))` -> [('I', 'PRP'), ('love', 'VBP'), ('tea', 'NN')]

### spacy's tagging by default
`import spacy`  
`nlp = spacy.load('en_core_web_sm')`  
`doc = nlp('i love tea')` -> returns a list of objects, each with 2 methods: .text and .tag_  
`doc[0].text` -> 'i'  
`doc[0].tag_` -> 'PRP'  

### Tagging workflow
```python
df = pd.read_csv('somedata.csv')
reviews = df['review'].tolist() # extract only the column you're interested in
adjective_frequencies = {} #keep count of adjectives

for review in tqdm(reviews): #tqdm just makes progress bars
    # Tokenize the review
    token_list = tweet_tokenizer.tokenize(review)
    # POS tag all words/tokens
    pos_tag_list = pos_tag(token_list)
    # Count the number of all adjectives
    for token, tag in pos_tag_list:
        if tag[0].lower() != 'j':
            # Ignore token if it is not an adjective (recall that JJ, JJR, JJS indicate adjectives)
            continue
        # Convert token to lowercase, otherwise "Good" and "good" are considered differently
        token = token.lower()
        if token not in adjective_frequencies:
            adjective_frequencies[token] = 1.0
        else:
            adjective_frequencies[token] = adjective_frequencies[token] + 1.0

show_wordcloud(adjective_frequencies)
```

### Stemming
`from nltk.stem.porter import PorterStemmer`
`from nltk.stem.snowball import SnowballStemmer`
`from nltk.stem.lancaster import LancasterStemmer`
`from nltk.stem import WordNetLemmatizer`

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer('english')

`stemmed_word = porter_stemmer.stem('accepted')` - stems the word to accept

### Lemmatization
`from nltk.stem import WordNetLemmatizer`
`wordnet_lemmatizer = WordNetLemmatizer()`
`lemmatized_word = wordnet_lemmatizer.lemmatize(word, pos='n')` -> pos $$\isin$$ {n, v, a, r}
`doc = nlp(sentence)`
```python
for token in doc:
    print (token.text, '=[{}]=>'.format(token.pos_), token.lemma_) 
```

---
### Appendix

#### Appendix 1: Upenn_tagset
$: dollar
    $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
'': closing quotation mark
    ' ''
(: opening parenthesis
    ( [ {
): closing parenthesis
    ) ] }
,: comma
    ,
--: dash
    --
.: sentence terminator
    . ! ?
:: colon or ellipsis
    : ; ...
CC: conjunction, coordinating
    & 'n and both but either et for less minus neither nor or plus so
    therefore times v. versus vs. whether yet
CD: numeral, cardinal
    mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
    seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
    fifteen 271,124 dozen quintillion DM2,000 ...
DT: determiner
    all an another any both del each either every half la many much nary
    neither no some such that the them these this those
EX: existential there
    there
FW: foreign word
    gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous
    lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte
    terram fiche oui corporis ...
IN: preposition or conjunction, subordinating
    astride among uppon whether out inside pro despite on by throughout
    below within for towards near behind atop around if like until below
    next into if beside ...
JJ: adjective or numeral, ordinal
    third ill-mannered pre-war regrettable oiled calamitous first separable
    ectoplasmic battery-powered participatory fourth still-to-be-named
    multilingual multi-disciplinary ...
JJR: adjective, comparative
    bleaker braver breezier briefer brighter brisker broader bumper busier
    calmer cheaper choosier cleaner clearer closer colder commoner costlier
    cozier creamier crunchier cuter ...
JJS: adjective, superlative
    calmest cheapest choicest classiest cleanest clearest closest commonest
    corniest costliest crassest creepiest crudest cutest darkest deadliest
    dearest deepest densest dinkiest ...
LS: list item marker
    A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
    SP-44007 Second Third Three Two * a b c d first five four one six three
    two
MD: modal auxiliary
    can cannot could couldn't dare may might must need ought shall should
    shouldn't will would
NN: noun, common, singular or mass
    common-carrier cabbage knuckle-duster Casino afghan shed thermostat
    investment slide humour falloff slick wind hyena override subhumanity
    machinist ...
NNP: noun, proper, singular
    Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
    Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
    Shannon A.K.C. Meltex Liverpool ...
NNPS: noun, proper, plural
    Americans Americas Amharas Amityvilles Amusements Anarcho-Syndicalists
    Andalusians Andes Andruses Angels Animals Anthony Antilles Antiques
    Apache Apaches Apocrypha ...
NNS: noun, common, plural
    undergraduates scotches bric-a-brac products bodyguards facets coasts
    divestitures storehouses designs clubs fragrances averages
    subjectivists apprehensions muses factory-jobs ...
PDT: pre-determiner
    all both half many quite such sure this
POS: genitive marker
    ' 's
PRP: pronoun, personal
    hers herself him himself hisself it itself me myself one oneself ours
    ourselves ownself self she thee theirs them themselves they thou thy us
PRP$: pronoun, possessive
    her his mine my our ours their thy your
RB: adverb
    occasionally unabatingly maddeningly adventurously professedly
    stirringly prominently technologically magisterially predominately
    swiftly fiscally pitilessly ...
RBR: adverb, comparative
    further gloomier grander graver greater grimmer harder harsher
    healthier heavier higher however larger later leaner lengthier less-
    perfectly lesser lonelier longer louder lower more ...
RBS: adverb, superlative
    best biggest bluntest earliest farthest first furthest hardest
    heartiest highest largest least less most nearest second tightest worst
RP: particle
    aboard about across along apart around aside at away back before behind
    by crop down ever fast for forth from go high i.e. in into just later
    low more off on open out over per pie raising start teeth that through
    under unto up up-pp upon whole with you
SYM: symbol
    % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R * ** ***
TO: "to" as preposition or infinitive marker
    to
UH: interjection
    Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
    huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
    man baby diddle hush sonuvabitch ...
VB: verb, base form
    ask assemble assess assign assume atone attention avoid bake balkanize
    bank begin behold believe bend benefit bevel beware bless boil bomb
    boost brace break bring broil brush build ...
VBD: verb, past tense
    dipped pleaded swiped regummed soaked tidied convened halted registered
    cushioned exacted snubbed strode aimed adopted belied figgered
    speculated wore appreciated contemplated ...
VBG: verb, present participle or gerund
    telegraphing stirring focusing angering judging stalling lactating
    hankerin' alleging veering capping approaching traveling besieging
    encrypting interrupting erasing wincing ...
VBN: verb, past participle
    multihulled dilapidated aerosolized chaired languished panelized used
    experimented flourished imitated reunifed factored condensed sheared
    unsettled primed dubbed desired ...
VBP: verb, present tense, not 3rd person singular
    predominate wrap resort sue twist spill cure lengthen brush terminate
    appear tend stray glisten obtain comprise detest tease attract
    emphasize mold postpone sever return wag ...
VBZ: verb, present tense, 3rd person singular
    bases reconstructs marks mixes displeases seals carps weaves snatches
    slumps stretches authorizes smolders pictures emerges stockpiles
    seduces fizzes uses bolsters slaps speaks pleads ...
WDT: WH-determiner
    that what whatever which whichever
WP: WH-pronoun
    that what whatever whatsoever which who whom whosoever
WP$: WH-pronoun, possessive
    whose
WRB: Wh-adverb
    how however whence whenever where whereby whereever wherein whereof why
``: opening quotation mark
    ` ``

### Appendix 2: show_wordcloud src
```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def show_wordcloud(source, max_words=50):
    try:
        wordcloud = WordCloud(max_words=1000)
        if type(source).__name__ == 'str' or type(source).__name__ == 'unicode':
            wordcloud.generate_from_text(source)
        else:
            wordcloud.generate_from_frequencies(source)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    except:
        raise ValueError("Invalid data type for source parameter: str or [(str,float)]")
           
```
