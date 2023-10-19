#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 15:19
# @Author  : LingJiaXiaoHu
# @File    : CalltTest.py
# @Software: win11 pytorch(GPU版本） python3.9.16
class Person:
    def __call__(self,name):
        print("__call__"+" Hello "+name)

    def hello(self,name):
        print("Hello "+name)

person = Person()
person("zhangshan")
person.hello("lisi")