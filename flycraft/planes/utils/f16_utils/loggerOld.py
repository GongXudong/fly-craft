#!/user/bin/python
# -*- coding: utf-8 -*-
'''LoggerOld is used to store texts into the indicated file on disk
Created by Weijia on 2017/06/01
Last modified by Weijia on 2018/08/04
'''
import os
loggerList = {}

class LoggerOld(object):
    def __init__(self, path, append = False):
        self.name = path
        if append:
            loggerList[self.name] = open(self.name ,'a')
        else:
            loggerList[self.name] = open(self.name ,'w')

    def write(self, str, end = "\n"):
        try:
            loggerList[self.name].write(str+end)
        except:
            writeLog(str, self.name)
            # print "logger "+self.name+" reinitialized"
    
    def continuousWrite(self, str):
        self.write(str,"")    
    
    def __del__(self):
        closeLog(self.name)

def hasLogger(name):
    return name in loggerList
    
def writeLog(str, name, end = "\n"):
    if hasLogger(name):
        loggerList[name].write(str+end)
    else:
        loggerList[name] = open(name ,'w')
        loggerList[name].write(str+end)
    loggerList[name].flush()

def closeLog(logName):
    if logName in loggerList:
        loggerList[logName].close()
        loggerList.pop(logName)