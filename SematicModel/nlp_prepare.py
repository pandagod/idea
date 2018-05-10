#! /usr/bin/env python
# -*- coding: utf-8 -*-
import re
from bs4 import BeautifulSoup


def clean_str(string):
    string = BeautifulSoup(string, "html.parser").get_text()

    string = re.sub(r"[^A-Za-z0-9();.,!#?’'`¥$€@//\s]", "", string)
    string = re.sub(r"\s{1,}", " ", string)
    string = re.sub(r"\?{1,}", "?", string)
    string = re.sub(r"\.{1,}", ".", string)
    return string



class CustomizeTokenizer(object):
    def __replace_entity(self, span, replacement):
        i = 1
        for token in span:
            if i == span.__len__():
                token.lemma_ = replacement
            else:
                token.lemma_ = u''
            i += 1

    def __customize_rule(self, doc):
        for ent in doc.ents:
            if ent.label_ == u'PERSON':
                self.__replace_entity(ent, u'-PERSON-')
            if ent.label_ == u'DATE':
                self.__replace_entity(ent, u'-DATE-')
            if ent.label_ == u'TIME':
                self.__replace_entity(ent, u'-TIME-')
            if ent.label_ == u'MONEY':
                self.__replace_entity(ent, u'-MONEY-')

        for token in doc:
            if token.like_url:
                token.lemma_ = u'URL'
            if token.is_digit and token.lemma_ not in [u'-DATE-', u'-TIME-', u'-MONEY-', u'']:
                token.lemma_ = u'NUM'
        return doc

    def __punct_space(self, token):
        return token.is_space or token.is_punct

    def __init__(self, nlp):
        self.nlp = nlp
        self.nlp.add_pipe(self.__customize_rule, name='customize rule')

    def __call__(self, doc):
        return [token.lemma_ for token in self.nlp(doc) if not self.__punct_space(token)]
