# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:35:10 2016

@author: abshinova
"""
def remove_html_markup(s):
    tag = False
    quote = False
    out = ""
    for c in s:
        
        if c == '<' and not quote:
            tag = True
        elif c =='>' and not quote:
            tag = False
        elif (c =='"'  or c == "'") and tag:
            quote = not quote
        elif not tag:
            out = out +c
    return out
    
print remove_html_markup('<a href=">">foo</a>')
    
