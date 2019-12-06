# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:44:49 2019

@author: Miklos Vecsei

This is the flap module for JET Lithium BES diagnostic
"""

import flap
import flap_jet_api as jetapi
from functools import partial

class ChopperConfig:
    '''
    Class for storing the Li-BES config data for a shot
    '''
    def __init__(self, exp_id, options={}):
        options_default = {
            "online": True}
        options = {**options_default, **options}
        self.exp_id = exp_id
        if options["online"] == True:
            self.get_chopper = partial(online_get_chopper, self_object=self)
        else:
            self.get_chopper = partial(offline_get_chopper, self_object=self)
        

def online_get_chopper(self_object, config=None, options={}):
    # The options keyword is passed on to the jetapi.getsignal() function
    # Looking at the flap_defaults.cfg if there are any signal locations defined
    # there
    if hasattr(self_object, "signal_loc") is False:
        try:
            self_object.signal_loc = flap.config.get("Module JET_API","Signal Location", evaluate=True)
        except ValueError:
            self_object.signal_loc = {}

    if config in self_object.signal_loc:
        # if the config is defined in the flap_defaults.cfg, use the location from there
        location = self_object.signal_loc[config]
    else:
        # if the config is not defined in the flap_defaults.cfg, use it as a location itself
        location = config
    
    # appending the result to the "data" dictionary
    data = jetapi.getsignal(self_object.exp_id, location,
                            options=options)
    self_object.data[config] = data        
    return data
    
def offline_get_chopper(self_object, config=None):
    raise NotImplementedError("Obtaining the chopper from offline files"+
                              " is not implemented yet")


if __name__ == "__main__":
    conf = LiBESConfig(96039)
    conf.get_config(config='KY6-Z', options={"UID": "KY6-team"})
    conf.get_config(config="KY6-CrossCalib", options={"UID": "KY6-team"})