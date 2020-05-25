#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def short_tag(tag):
    pieces = tag.split("_")
    for i, p in enumerate(pieces):
        if "-" in p:
            p, v = p.split("-")
        else:
            v = ""
        pieces[i] = p[:3] + v
    return "_".join(pieces)