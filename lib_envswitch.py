#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket

class EnvSwitcher(object):

    def __init__(self):
        self.var_map = {} # name, key -> val

    def who(self):
        hostname = socket.gethostname()
        if "abci"