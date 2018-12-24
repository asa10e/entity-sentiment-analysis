# NLTK
import nltk
from nltk.corpus import wordnet, sentiwordnet
# Spacy
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_lg') # For word and sentence level tokenization and POS tagging
# StanfordCoreNLP
from pycorenlp import StanfordCoreNLP

# Other
import re
from curr_dic import curr_dic, curr_lis

"""Making NEGATORS word list"""
NEGATORS = ["ain't",'cannot',"can't","didn't","doesn't","don't","hadn't",'hardly',
            "hasn't","haven't","havn't","isn't",'lack','lacking','lacks','neither',
            'never','no','nobody','none','nor','not','nothing','nowhere',"mightn't",
            "mustn't","needn't","oughtn't","shouldn't","wasn't",'without',"wouldn't",
            'are not','can not', 'did not','does not','do not','had not', 'has not',
            'have not','is not','might not','must not','need not','ought not','should not',
            'was not','would not', "n't"]
NEGATORS.sort(key = lambda x: 1/len(x))

###########
bad_terms = ['BUZZ-','UPDATE','COMMENT','TABLE','GLOBAL MARKETS','BRIEF-','WRAPUP','FOREX','FX',
            'Technical Analysis','PRECIOUS','sources','INTERNATIONAL/REGIONAL','official source',
            'For best results when printing this announcement, please click on link below:',
            'Click the following link to watch video:','Source: Thomson Reuters\n\nDescription:',
            'The following files are available for download:',
            'North Korean','North Korea','NK','N.Korean','N. Korean','N. Korea','N.Korea',
            'North Korean Won',"North Korea's Won", 'South America','South American',
            'Latin America','Latin American','North America','North American',
            'Chilean Peso','Argentinian Peso','Colombian Peso','Cuban peso']
bad_terms.sort(key = lambda phrase: 1/len(phrase))

single_replacements = {curr: curr for curr in curr_lis}
single_replacements = {**single_replacements, **{curr.lower():curr for curr in curr_lis}}

# Construct a dictionary of currency combinations (usd cny)
combo_lis = []
for c1 in curr_lis:
    for c2 in [curr for curr in curr_lis if curr!=c1]:
        combo_lis.append((c1+c2))
combo_replacements = {pair: pair[:3]+pair[3:6] for pair in combo_lis}
combo_dashes = {pair[:3]+'-'+pair[3:6]:val for pair,val in combo_replacements.items()}
combo_slashes = {pair[:3]+'/'+pair[3:6]:val for pair,val in combo_replacements.items()}
combo_spaces = {pair[:3]+' '+pair[3:6]:val for pair,val in combo_replacements.items()}

combo_replacements = {**combo_replacements, **combo_dashes, **combo_slashes, **combo_spaces}
combo_replacements = {**combo_replacements, **{key.lower(): val for key, val in combo_replacements.items()}}
# 'USD CNY','USD/CNY','USDCNY','usd cny','usd/cny','usdcny' etc. -> 'USDCNY'
def text_cleaner(text):

    for term in bad_terms:
        text = text.replace(term, ' ')

    # Simplify negators
    text = ' '+text+' '
    for negator in NEGATORS:
        text = text.replace(' '+negator+' ', ' not ')

    text = re.sub(r'^(.*\(Reuters)','',text) # Just get rid of everything before (Reuters
    text = re.sub(r'^(.*\(Bloomberg)','',text) # Just get rid of everything before (Bloomberg
    text = re.sub(r'\s+', ' ', text) # Remove repeated spaces/newlines
    text = re.sub(r'(<\^).*', '', text) # Remove anything after <^

    text = re.sub(r'\((.*?)\)',' ', text) # Remove ('text')
    text = re.sub(r'\[(.*?)\]',' ', text) # Remove ['text']
    text = re.sub(r'<(.*?)>',' ', text) # Remove <'text'>
    text = re.sub(r'\S*@\S*\s?', ' ', text) # Remove emails
    text = re.sub(r'http\S+', ' ', text) # Remove URLs

    text = re.sub(r'\s+', ' ', text) # Remove repeated spaces/newlines again
    text = text.replace(' *','.')
    text = text.replace('..','.')

    # text = text.lstrip() # Trim leading whitespace, which allows the following few lines to work
    # text = re.sub(r'^By\s[a-zA-Z]+\s[a-zA-Z]+\s', '', text) # Removes ^By Asa Tenney
    # text = re.sub(r'^[a-zA-Z]+\s+[a-zA-Z]+,\s[a-zA-Z]{3}\s\d{1,2}', ' ', text) # Removes ^VATICAN CITY, Jan 26
    # text = re.sub(r'^[a-zA-Z]+,\s[a-zA-Z]{3,8}\s\d{1,2}', ' ', text) # Removes ^BEIJING, Jan 26
    text = re.sub(r'\s+', ' ', text) # Remove repeated spaces/newlines again

    text = re.sub(r'^US:', ' USD ', text)
    text = re.sub(r'^US\s', ' USD ', text)

    text = ' ' + text + ' ' # Add spaces surrounding the text, this makes finding whether or not a phrase occurs simpler

    # LOWERING THE TEXT HAS MANY IMPLICATIONS, JUST FYI
    #'Japan Military' WILL NOT TRIGGER JPY AS IT IS ITS OWN ENTITY, BUT 'japan military' WILL
    text = text.lower()

    for c in ''':'.,";-''':
        text = text.replace(c, ' '+c+' ')

    # 'USD CNY','USD/CNY','USDCNY','usd cny','usd/cny','usdcny' etc. -> 'USDCNY'
    for combo in combo_replacements.keys():
        text = text.replace(' '+combo+' ', ' '+combo_replacements[combo]+' ')

    for sing in single_replacements.keys():
        text = text.replace(' '+sing+' ', ' '+single_replacements[sing]+' ')

    text = re.sub(r'''[^a-zA-Z0-9,.;:/\-'"]+''', ' ', text) # Get rid of special characters, except some punctuation

    text = text.replace(" won ' t ", " won't ")

    return text
###########

########################################
# Many of these are already usually correctly identified as positive/negative, but not as often as we'd like
bull_terms = ['breaking','broke out','break out','rate hikes','rate hike',
 'breaks out','breaks higher','bull','bulls','bull market','firm','firms','firming','firmed','gain','gains',
 'gaining','gained','extend gains','extends gain','extends gains','grow','grows','growing','grew','high','higher',
 'edge higher','set to edge higher','highs','jump','jumps','jumped','jumping','lower deficit','low deficit',
 'lowering deficit','lowering the deficit','gdp growth','deficit lower','moving in the right direction','rally',
 'rallies','rallied','rallying','rebound','rebounds','rebounding','rise','rising','rose','rises','may rise','might rise',
 'risk minimize','risk minimized','risk minimise','risk minimised','strong','strength','strengthening','surge','surging','surges',
 'surged','underpin','underpins','underpinning','underpinned','uptick','upbeat','going up','boosts','boosted','boosting','boost']

bear_terms = ['bear','bears','bear market','declining','declined','dealt a blow','tariff','tariffs',
 'slump','declines','slumps','slumped','plunged','plunging','plunges','deflate','deflating','deflated',
 'deflates','going down','down','went down','fall','drop','drops','fell','falls','dropped','dropping',
 'falling','lose','lost','losing','lows','lower','edge lower','set to edge lower','poor','resist',
 'resistance','stumble','stumbles','stumbled','stumbling','tumble','tumbles','tumbled','tumbling',
 'weak','weaken','weakens','weakening','swoon','swoons','swooned','swooning','erases gains','erases most gains','erased','erases',
 'import duties','trade war','retreating','retreat','retreats','dip','dips','dipped','dipping','grows least','grew least','grows less',
 'growing less','grew less','growing least','drag','drags','dragging','dragged']

# Add surrounding spaces for better replacing
bull_terms = [' ' + t + ' ' for t in bull_terms]
bear_terms = [' ' + t + ' ' for t in bear_terms]
# Create dictionaries of phrase:bullish/bearish
bull_dic = dict(zip(bull_terms, [' bullish ' for _ in range(len(bull_terms))]))
bear_dic = dict(zip(bear_terms, [' bearish ' for _ in range(len(bear_terms))]))
bull_bear_dic = {**bull_dic, **bear_dic} # Combine the two dictionaries

def bull_bear_replacer(text):
    '''
    Replaces known bullish/bearish phrases in text with bullish/bearish, which are
    known to MeaningCloud.
    '''
    text = ' '+text+' '

    # Sort phrases be descending length
    phrase_list = list(bull_bear_dic.keys())
    phrase_list.sort(key = lambda phrase: 1/len(phrase))

    # Replace phrases in text (in order of descending length)
    for phrase in phrase_list:
        text = text.replace(phrase, bull_bear_dic[phrase])

    return text.strip()
##########################################

def start_end_idx(substring, text):
    '''
    Not the cleanest methodology but it does work.
    Given a substring, finds the beginning and inclusive ending index of the substring in the substring
    Finds all such occurences.
    Output is a list of lists such as [[0, 6], [30, 36]]
    If the substring does not occur in the text, an empty list is returned
    '''
    if 'Ω' in substring: return "Doesn't work if the substring is 'Ω'... sorry!"

    start_end_list = []
    while substring in text:
        start = text.index(substring)
        end = start + len(substring) - 1
        start_end_list.append([start, end])

        text = text.replace(substring, 'Ω'*len(substring), 1)
    return start_end_list

# TODO fix punctuation issue, Won issue
def curr_replace_text(text):
    """
    Uses start_end_idx to replace currency terms in text.
    End result is text like "USD is a cool country!"
    """
    text = ' '+text+' '
    text = text.replace(' US ',' USD ')
    text = re.sub(r'^(US )','USD ', text)
    text = re.sub(r'( US)$', ' USD', text)
    text = re.sub(r'(U\.S\.)$', 'USD', text)
    text = re.sub(r'(U\.K\.)$', 'GBP', text)
    text = re.sub(r'(E\.U\.)$', 'EUR', text)

    # Create a list of phrases in curr_dic in descending order, so we replace larger substrings first
    phrase_list = list(curr_dic.keys())
    phrase_list.sort(key = lambda phrase: 1/len(phrase))


    for phrase in phrase_list:
        start_end_tup = start_end_idx(phrase, text.lower())
        if start_end_tup == []: continue
        for index_pair in start_end_tup[::-1]:
            start, end = index_pair
            text = text[:start+1] + curr_dic[phrase] + text[end:]

    text = bull_bear_replacer(text)

    return text.strip()

def word_to_sentiment(word, pos=None):
    """
    Given a word and an optional part of speech tag,
    returns the average sentiment score for that word using senti_synsets.
    """
    if word in STOP_WORDS or word in NEGATORS:
        return 0

    if pos == None:
        synsets = list(sentiwordnet.senti_synsets(word))

    elif pos != None:
        def get_wordnet_pos(tag):
            """
            Translating Spacy's POS tags to WordNet's POS tags
            """
            if tag.startswith('ADJ'): return wordnet.ADJ
            elif tag.startswith('VERB'): return wordnet.VERB
            elif tag.startswith('NOUN'): return wordnet.NOUN
            elif tag.startswith('ADV'): return wordnet.ADV
            else: return wordnet.NOUN

        wordnet_pos = get_wordnet_pos(pos)
        synsets = list(sentiwordnet.senti_synsets(word, wordnet_pos))

    score = 0
    for el in synsets:
        score += el.pos_score()
        score -= el.neg_score()

    if score == 0 and pos != None:
        return word_to_sentiment(word, pos=None)

    if len(synsets) == 0: return 0

    score /= len(synsets)
    return score

def list_to_score(pos_tags):
    """
    Finds the average sentiment of a list of tuples of (word, pos) using word_to_sentiment()
    """
    score = 0
    curr_count = 0
    words_with_sentiment = 0
    for tup in pos_tags:
        word, pos = tup[0], tup[1]
        if word in curr_lis:
            curr_count += 1
        else:
            if word_to_sentiment(word, pos) != 0:
                words_with_sentiment += 1
                score += word_to_sentiment(word, pos)
    if score == 0:
        return 0
    else:
        return score/words_with_sentiment

# def stanford_sent(text):
#     # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
#     stanford_nlp = StanfordCoreNLP('http://localhost:9000')
#     response = stanford_nlp.annotate(text, properties={'annotators': 'sentiment',
#                                        'outputFormat': 'json',
#                                             'timeout': 1000,
#                        })
#
#     scores = [(int(part['sentimentValue']) - 2)/2 for part in response['sentences']]
#     # Translate scores from stanford system to [-1, 1] system
#
#     # print(scores)
#     return sum(scores)/len(scores)

def posneg_count(sent_tags):
    """
    Number of either positive or negative words in a list of pos_and_sent_tags
    """
    nonzero_sentiments = list(filter(lambda tup: tup[2] != 0, sent_tags))
    return len(nonzero_sentiments)

def unique_currs(pos_and_sent_tags):
    text = ' '.join([tup[0] for tup in pos_and_sent_tags])
    return set(filter(lambda c: c in text, curr_lis))

def unique_curr_count(pos_and_sent_tags):
    return len(unique_currs(pos_and_sent_tags))


# TODO handle currency pairs
class Sentence(object):
    """
    """
    def __init__(self, text):
        self.raw = text
        self.text = curr_replace_text(self.raw)
        self.tokens = [token for token in nlp(self.text)] # word_tokenize(self.text)
        self.pos_tags = list(zip(map(str, self.tokens), map(lambda token: token.pos_, self.tokens))) # nltk.pos_tag(self.tokens)
        self.sent_tags = [(tup[0], word_to_sentiment(word=tup[0], pos=tup[1])) for tup in self.pos_tags]
        self.pos_and_sent_tags = [(self.pos_tags[i][0], self.pos_tags[i][1], self.sent_tags[i][1]) for i in range(len(self.pos_tags))]
        self.total_sent = list_to_score(self.pos_tags)
        self.posneg_count = posneg_count(self.pos_and_sent_tags)

    def parse(self):
        """
        Splits a sentence into a list of chunks comprising the sentence (uses the pos_and_sent_tags attribute)
        """
        # Find number of currencies in the sentence
        if unique_curr_count(self.pos_and_sent_tags) == 1: # If this number is 1, return the entire sentence
            return [self.pos_and_sent_tags]
        elif unique_curr_count(self.pos_and_sent_tags) == 0: # If this number is 0, return an empty list
            return []
        if unique_curr_count(self.pos_and_sent_tags) > 1: # If this number is > 1, parse on commas
            phrase_list = []
            tags_so_far = []
            for tup in self.pos_and_sent_tags:
                token = tup[0]
                if (token != ',') and (token != ';'):
                    tags_so_far.append(tup)
                elif (token == ',') or (token == ';'):
                    phrase_list.append(tags_so_far)
                    tags_so_far = []
            phrase_list.append(tags_so_far)
        if len(phrase_list) == 0:
            return phrase_list

        # Discard phrases with a unique_curr_count of 0
        phrase_list = list(filter(lambda x: unique_curr_count(x) > 0, phrase_list))

        def phrase_splitter(phrase):
            if unique_curr_count(phrase) > 1 and posneg_count(phrase) > 1:
                part1 = phrase[:1]
                i = 0
                while unique_curr_count(part1) == 0:
                    part1 = phrase[:i]
                    i += 1
                while posneg_count(part1) == 0:
                    part1 = phrase[:i]
                    i += 1

                part2 = phrase[i-1:]
                if unique_curr_count(part2) == 0:
                    return phrase
                else:
                    return [part1, part2]
            else:
                return [phrase]

        split_phrase_list = []
        for phrase in phrase_list:
            split_phrase_list.extend(phrase_splitter(phrase))

        return split_phrase_list

    def sent_dic_wordnet(self):

        chunks = self.parse()
        sent_dict_list = []

        for chunk in chunks:
            sent_dict = dict()
            sent = list_to_score(chunk)

            # Check if a negator is in the chunk. if so, reverse score
            text = ' '.join([tag[0] for tag in chunk])

            if ' not ' in ' '+text+' ':
                sent *= -0.9

            # for word in NEGATORS:
            #     if ' '+word+' ' in ' '+text+' ':
            #         sent *= -0.9 # Literal is to diminish how much negation flips sentiment
            #         break

            for curr in unique_currs(chunk):
                sent_dict[curr] = sent
            sent_dict_list.append(sent_dict)

        # Combine each dict in sent_dict_list, summing common entries
        combined_dict = dict()
        for curr in unique_currs(self.pos_and_sent_tags):
            combined_dict[curr] = sum([dic.get(curr, 0) for dic in sent_dict_list])

        return combined_dict

    # def sent_dic_stanford(self):
    #
    #     chunks = self.parse()
    #     sent_dict_list = []
    #
    #     for chunk in chunks:
    #
    #         text = ' '.join([tag[0] for tag in chunk])
    #         text = text.replace(" ' ", "'")
    #         text = text.replace(" '", "'")
    #
    #         sent = stanford_sent(text)
    #         sent_dict = dict()
    #         for curr in unique_currs(chunk):
    #             sent_dict[curr] = sent
    #         sent_dict_list.append(sent_dict)
    #
    #     # Combine each dict in sent_dict_list, summing common entries
    #     combined_dict = dict()
    #     for curr in unique_currs(self.pos_and_sent_tags):
    #         combined_dict[curr] = sum([dic.get(curr, 0) for dic in sent_dict_list])
    #
    #     return combined_dict

def find_sent_dic(text):
    if type(text)!=str:
        return dict()
    S = Sentence(text)
    try:
        return S.sent_dic_wordnet()
    except:
        return dict()

def doc_sentiment(text):

    text = text_cleaner(text)

    sentences = [str(s).replace('.','') for s in nlp(text).sents]
    sent_dict_list = [find_sent_dic(s) for s in sentences]

    # Combine dictionaries, summing common entries:
    combined_dict = dict()
    for curr in curr_lis:
        combined_dict[curr] = sum([dic.get(curr, 0) for dic in sent_dict_list])
    combined_dict = {k:v for k,v in combined_dict.items() if v != 0}

    sign = lambda x: 'Negative' if x < -.01 else 'Positive' if x > .01 else 'Neutral'
    combined_dict = {k:sign(v) for k,v in combined_dict.items()}

    return combined_dict
