#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.utils import OPTS
import socket

class EnvSwitcher(object):

    def __init__(self):
        self.var_map = {} # name, key -> val

    def who(self):
        hostname = socket.gethostname()
        if "abci" in hostname:
            return "shu"
        else:
            return "jason"

    def register(self, owner, key, val):
        self.var_map[(owner, key)] = val

    def load(self, key, default=None):
        # Load registered val for current owner
        # If not found, return default val
        owner = self.who()
        if (owner, key) in self.var_map:
            return self.var_map[(owner, key)]
        else:
            return default


if "envswitch" not in OPTS:
    OPTS.envswitch = EnvSwitcher()
envswitch = OPTS.envswitch