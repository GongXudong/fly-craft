#!/usr/local/bin/python
# -- coding: utf-8 --
''' config management tools
Created by Weijia on 2017/06/01
Last modified by Weijia on 2019/10/12
'''
import sys
if sys.version_info.major == 2:   # modified wh on 20200221
    import ConfigParser as configparser
else:
    import configparser
import string

def loadSecItems(filename, sec):
    cf = configparser.ConfigParser()
    cf.read(filename)
    items = cf.items(sec)    
    return items

def loadSecMap(filename, sec):
    cf = configparser.ConfigParser()
    cf.read(filename)
    items = cf.items(sec)
    map = {}
    for it in items:
        map[it[0]] = it[1]
    return map
    
def loadConfig(filename):
    cf = configparser.ConfigParser()
    cf.read(filename)
    section = cf.sections()
    map = {}
    for s in section:
        map[s] = {}
        items = cf.items(s)
        for ps in items:  
            map[s][ps[0]]=ps[1]
    return map

def judgeTrue(word):
    if word == None: return False
    return word.lower() == "true"