# -*- coding: utf-8 -*-


__author__="Fabian Kuschka-Kleibrink"
__date__="2015-07-20"

from pprint import pprint
import nltk
import yaml
import codecs


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

    def pos_tag(self, pos):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        # Format fuer DictionaryTagger anpassen
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos


# Tagged die POStagged Saetze mit positiven oder negativen Worten
class DictionaryTagger(object):
    def __init__(self, dictionary_paths):
        # Dictionaries laden
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        # Dictionary mit pos und neg Worten erstellen
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(str(key)))

    def tag(self, postagged_sentences):
        # Hilfs-Funktion zum taggen von Saetzen
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence):
        # Modifiziert von Alba
        # Funktion zum taggen der Saetze mit dem Dictionary
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
        while i < N:
            j = min(i + self.max_key_size, N)  # avoid overflow
            tagged = False
            while j > i:
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                literal = expression_form
                if literal in self.dictionary:
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token:  # if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j -= 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence


def value_of(sentiment):
    # Hilfsfunktion zur berechnung des Sentiment-Scores
    if sentiment == 'positive':
        return 1
    if sentiment == 'negative':
        return -1
    return 0


def sentence_score(sentence_tokens, previous_token, acum_score):
    # Funktion zur Berechnung des Sentiment-Scores
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)


def sentiment_score(review):
    # Hilfsfunktion zur berechnung des Sentiment-Scores
    return sum([sentence_score(sentence, None, 0.0) for sentence in review])


def extract_entity_names2(t):
    # Funktion zum Finden von NEs in einem Baum
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names2(child))
    return entity_names


if __name__ == "__main__":
    # Wird ausgefuehrt beim Start des Programms
    chapter = 10
    entity_names = []
    with codecs.open("chapters/chapter_" + str(chapter) + ".txt", 'r', encoding='utf-8') as f:
        text = f.read()


    # Initialisierung NEU
    postagger = POSTagger()
    dicttagger = DictionaryTagger(['dicts/positive-words.yml', 'dicts/negative-words.yml'])

    # Verarbeitung des Textes mit NLTK
    # Sentence Tokenizer
    sentences = nltk.sent_tokenize(text)
    print("sentences")
    pprint(sentences)
    # Word Tokenizer
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    print("tokenized_sentences")
    pprint(tokenized_sentences)
    # POS-Tagger
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    print("tagged_sentences")
    pprint(tagged_sentences)
    # Chunker
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    print("chunked_sentences")

    sentences_har = []
    sentences_ron = []
    sentences_her = []
    counter = 0
    entity_names = []

    # Filtern von NEs in den Baeumen
    for tree in chunked_sentences:
        print("sentence " + str(counter))
        print(tree)
        print(tagged_sentences[counter])
        entity_names = []
        entity_names.extend(extract_entity_names2(tree))
        print(entity_names)
        for entity in entity_names:
            print(entity.lower())
            if entity.lower() == 'harry':
                sentences_har.append(tagged_sentences[counter])
            if entity.lower() == 'harry potter':
                sentences_har.append(tagged_sentences[counter])
            if entity.lower() == 'potter':
                sentences_har.append(tagged_sentences[counter])
            if entity.lower() == 'ron':
                sentences_ron.append(tagged_sentences[counter])
            if entity.lower() == 'ronald':
                sentences_ron.append(tagged_sentences[counter])
            if entity.lower() == 'ronald weasley':
                sentences_ron.append(tagged_sentences[counter])
            if entity.lower() == 'weasley':
                sentences_ron.append(tagged_sentences[counter])
            if entity.lower() == 'ron weasley':
                sentences_ron.append(tagged_sentences[counter])
            if entity.lower() == 'hermione granger':
                sentences_her.append(tagged_sentences[counter])
            if entity.lower() == 'hermione':
                sentences_her.append(tagged_sentences[counter])
            if entity.lower() == 'granger':
                sentences_her.append(tagged_sentences[counter])
            if entity.lower() == 'hermione jean granger':
                sentences_her.append(tagged_sentences[counter])
        counter += 1

    print set(entity_names)
    print("sentences_har")
    pprint(sentences_har)
    print("sentences_ron")
    pprint(sentences_ron)
    print("sentences_her")
    pprint(sentences_her)

    # fuer SA Harry
    har_pos_tagged_sentences = postagger.pos_tag(sentences_har)
    print("pos_tagged_sentences")
    pprint(har_pos_tagged_sentences)
    har_dict_tagged_sentences = dicttagger.tag(har_pos_tagged_sentences)
    print("dict_tagged_sentences")
    pprint(har_dict_tagged_sentences)

    # fuer SA Ron
    ron_pos_tagged_sentences = postagger.pos_tag(sentences_ron)
    print("pos_tagged_sentences")
    pprint(ron_pos_tagged_sentences)
    ron_dict_tagged_sentences = dicttagger.tag(ron_pos_tagged_sentences)
    print("dict_tagged_sentences")
    pprint(ron_dict_tagged_sentences)

    # fuer SA Hermione
    her_pos_tagged_sentences = postagger.pos_tag(sentences_her)
    print("pos_tagged_sentences")
    pprint(her_pos_tagged_sentences)
    her_dict_tagged_sentences = dicttagger.tag(her_pos_tagged_sentences)
    print("dict_tagged_sentences")
    pprint(her_dict_tagged_sentences)

    print("CHAPTER" + str(chapter))

    # Sentiment-Scores ausgeben
    # Harry
    print("Score Harry Potter:")
    score_har = sentiment_score(har_dict_tagged_sentences)
    print(score_har)
    # Ron
    print("Score Ronald Weasley:")
    score_ron = sentiment_score(ron_dict_tagged_sentences)
    print(score_ron)
    # Hermione
    print("Score Hermione Granger:")
    score_her = sentiment_score(her_dict_tagged_sentences)
    print(score_her)
