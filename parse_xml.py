#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import xml.sax

import sys


class StreamHandler(xml.sax.handler.ContentHandler):
    lastEntry = None
    lastName = None

    def startElement(self, name, attrs):
        self.lastName = name
        if name == 'book':
            self.lastEntry = {}
        elif name != 'root':
            self.lastEntry[name] = {'attrs': attrs, 'content': ''}

    def endElement(self, name):
        if name == 'book':
            print({
                'a': self.lastEntry['a']['content'],
                'b': self.lastEntry['b']['attrs'].getValue('foo')
            })
            self.lastEntry = None
        elif name == 'root':
            raise StopIteration

    def characters(self, content):
        if self.lastEntry:
            self.lastEntry[self.lastName]['content'] += content


if __name__ == '__main__':
    # use default ``xml.sax.expatreader``
    parser = xml.sax.make_parser()
    parser.setContentHandler(StreamHandler())

    # feed the parser with small chunks to simulate
    # with open(sys.argv[1]) as f:
    #     while True:
    #         buffer = f.read(16)
    #         if buffer:
    #             try:
    #                 parser.feed(buffer)
    #             except StopIteration:
    #                 break
    #         else:
    #             time.sleep(2)

    # if you can provide a file-like object it's as simple as
    with open(sys.argv[1]) as f:
        parser.parse(f)
