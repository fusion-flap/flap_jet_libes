# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:44:49 2019

@author: Miklos Vecsei

This is the flap module for JET Lithium BES diagnostic
"""

import os
import warnings
import copy
import numpy as np
from functools import partial
import flap
import flap_jet_api as jetapi

class LiBESConfig:
    '''
    Class for storing the Li-BES config data for a shot
    '''
    def __init__(self, exp_id, options={}):
        options_default = {
            "online": True}
        options = {**options_default, **options}
        self.exp_id = exp_id
        self.data = {} # a dictionary of configuration information
        self.chopper_on = None # the dataobject used for getting the ontime of the ky6d beam
        self.chopper_off = None  # the dataobject used for getting the offtime of the ky6d beam
        self.energy_mean = None # the mean energy of the beam  in keV during the plasma
        self.apd_map = None # the map for storing the fiber names
        self.fibre_conf = None # an ordered list of the complete fibre configuration

        if options["online"] == True:
            self.get_config = partial(online_get_config, self_object=self)
            self.get_chopper = partial(online_get_chopper, self_object=self)
        else:
            self.get_config = partial(offline_get_config, self_object=self)
            self.get_chopper= partial(offline_get_chopper, self_object=self)

    def chopper_timing_data_object(self, options={}):
        """ Determine the chopper timing and return a data object with Time and Sample coordinates
    
            options:
            'Start delay' : Start delay relative to the phase start in microsec
            'End delay' : Delay at the end of the phase in microsec. Positive means later times.
        """
        # The distance of the chopper plates to the Z=0 plane
        beam_distance = float(flap.config.get("Module JET_LIBES","Chopper plate height"))
        mass_li = 1.1525801e-26
        beam_velocity = np.sqrt(2*self.energy_mean*1.60217662e-16/mass_li)
        time_delay = beam_distance/beam_velocity
        
        mode = flap.CoordinateMode(equidistant=False, range_symmetric=False)

        c_time = flap.Coordinate(name='Time',
                                 unit='Second',
                                 mode=mode,
                                 values = self.data['KY6-ChopOff'].data[:,0]+time_delay,
                                 value_ranges=self.data['KY6-ChopOff'].data[:,1]-\
                                              self.data['KY6-ChopOff'].data[:,0],
                                 dimension_list=[0])
        self.chopper_off = flap.DataObject(data_shape=[self.data['KY6-ChopOff'].shape[0]],
                                           data_unit=flap.Unit(name='Interval', unit='n.a.'),
                                           coordinates=[c_time],
                                           data_source = 'JET_LIBES',
                                           exp_id = self.exp_id
                                           )

        c_time = flap.Coordinate(name='Time',
                                 unit='Second',
                                 mode=mode,
                                 values = self.data['KY6-ChopOn'].data[:,0]+time_delay,
                                 value_ranges=self.data['KY6-ChopOn'].data[:,1]-\
                                              self.data['KY6-ChopOn'].data[:,0],
                                 dimension_list=[0])
        self.chopper_on = flap.DataObject(data_shape=[self.data['KY6-ChopOn'].shape[0]],
                                          data_unit=flap.Unit(name='Interval', unit='n.a.'),
                                          coordinates=[c_time],
                                          data_source = 'JET_LIBES',
                                          exp_id = self.exp_id
                                          )
    
    #-----------------------READING THE FIBRE SETUP FILES----------------------
    def read_apd_map(self):
        '''
        Read the apd_map file and the broken_fibres file and creates the fibre
        relevant to the current exp_id. The location of the files is defined 
        in flap_defaults.cfg
        INPUT: data is read from flap_defaults.cfg
        OUTPUT: The channel mapping for the apd
        '''
        # getting the apd_map
        filename = flap.config.get("Module JET_LIBES","APD map")
        filename = find_absolute_path(filename)
        with open(filename, "r") as apdmap:
            #finding the proper mapping based on exp_id
            all_apd_maps = apdmap.read().split("*")
            exp_id_limits = [[int(exp_id) for exp_id in apd_map.split("\n")[0].split("/")]
                             for apd_map in all_apd_maps[1:]]
            exp_id_limits = np.asarray(exp_id_limits)
            # in the apd map file the latest mapping has an upper exp_id limit 0
            # this uppwer exp_id limit will be set to self.exp_id+1 so that it is
            # definitely larger than the current exp_id
            last_map_index = np.where(exp_id_limits[:,1]==0)
            exp_id_limits[last_map_index, 1] = self.exp_id+1
            # the map for the current exp_id is given by
            map_index = np.where((exp_id_limits[:,0]-self.exp_id <= 0) *\
                                 (exp_id_limits[:,1]-self.exp_id >= 0))[0]
            
            # creating the multidimensional list for mapping
            apd_map = [channelID_variants.split(",") for \
                       channelID_variants in all_apd_maps[1+int(map_index)].split("\n")[1:]]
            #removing the empty strings
            self.apd_map = [channel for channel in apd_map if channel != [""]]

        # removing the channels defined in the broken channels file
        # work practically same way as for the apd map file, but since the
        # files are formatted differently, the code is slightly different
        filename = flap.config.get("Module JET_LIBES","Broken channels")
        filename = find_absolute_path(filename)
        with open(filename, "r") as broken_channels:
            #finding the proper data in the file based on exp_id
            all_broken_channels= broken_channels.read().split("*")
            exp_id_limits = [[int(exp_id) for exp_id in broken_channel.split(":")[0].split("/")]
                             for broken_channel in all_broken_channels[1:]]
            exp_id_limits = np.asarray(exp_id_limits)
            last_map_index = np.where(exp_id_limits[:,1]==0)
            exp_id_limits[last_map_index, 1] = self.exp_id+1
            exp_index = np.where((exp_id_limits[:,0]-self.exp_id <= 0) *\
                                 (exp_id_limits[:,1]-self.exp_id >= 0))[0]

            # getting the broken channels
            exp_broken_channels = [channels.split(",") for channels in \
                                   all_broken_channels[1+int(exp_index)].split("\n")[0].split(":")[1:]][0]
            exp_broken_channels = [channel for channel in exp_broken_channels if channel != [""]]
            
            # popping out the broken channels from self.apd_map
            if len(exp_broken_channels)>0:
                for channel in exp_broken_channels:
                    channel_index = np.where([(channel in channel_indices)
                                                  for channel_indices in self.apd_map])[0]
                    print("broken channel")    
                    if len(channel_index)>0:
                        channel_index = int(channel_index[0])
                        if channel_index < len(self.apd_map)-1:
                            self.apd_map = self.apd_map[:channel_index] + \
                                           self.apd_map[channel_index+1:]
                        else:
                            self.apd_map = self.apd_map[:channel_index]
        return self.apd_map
    
    def read_fibre_coords(self):
        '''
        Reads the fibre_coords file
        INPUT: data is read from flap_defaults.cfg
        OUTPUT: The fibre coords mapping for the apd
        '''
        # getting the fibre coordinates. almost the same as getting the apd_map
        # only some changes had to be implemented
        filename = flap.config.get("Module JET_LIBES","Fibre coordinates")
        filename = find_absolute_path(filename)
        with open(filename, "r") as fibre_coords:
            #finding the proper mapping based on exp_id
            all_fibre_coords = fibre_coords.read().split("*")
            exp_id_limits = [[int(exp_id) for exp_id in fibre_coord.split("\n")[0].split("/")]
                             for fibre_coord in all_fibre_coords[1:]]
            exp_id_limits = np.asarray(exp_id_limits)

            last_setup_index = np.where(exp_id_limits[:,1]==0)
            exp_id_limits[last_setup_index, 1] = self.exp_id+1
            # the map for the current exp_id is given by
            setup_index = np.where((exp_id_limits[:,0]-self.exp_id <= 0) *\
                                   (exp_id_limits[:,1]-self.exp_id >= 0))[0]
            
            # creating the multidimensional list for mapping
            fibre_coord = [channel_coord.split("£") for \
                           channel_coord in all_fibre_coords[1+int(setup_index)].split("\n")[1:]]
            
            # there are a number of lines with commens noted by "£" which should be removed
            # all lines starting with a "£" are removed, otherwise everything following a
            # "£" is removed from every line
            fibre_coord_orig = copy.deepcopy(fibre_coord)
            for channel in fibre_coord_orig:
                if channel[0] == "":
                    fibre_coord.remove(channel)
                    print("fibre_coord")
                elif len(channel)>1:
                    fibre_coord.remove(channel)
                    channel = channel[0]
                    fibre_coord.append([channel])
            
            fibre_coord = [channel[0].split(",") for channel in fibre_coord]
            #removing the empty strings
            for channel in fibre_coord:
                channel[1] = float(channel[1])
                channel[2] = float(channel[2])
                
        return fibre_coord

    def get_fibre_config(self):
        '''
            The main function for reading the fibre setup files.
            The lines of self.apd_map are read and ordered according to the
            location of the channels along the beam axis defined in self.fibre_coords.
            The order is with increasing distance
        '''
        # first a large matrix is created which comes from merging self.fibre_coord
        # into self.apd_map
        if self.apd_map is None:
            self.read_apd_map()
        fibre_coord = self.read_fibre_coords()

        # finding the data in fibre_coord corresponding to the channels in self.apd_map
        self.fibre_conf = []
        fibre_coord_names = [fibre_data[0] for fibre_data in fibre_coord]
        # one needs to take into account that there are some channels in apd_map
        # for which the location is missing in fibre_coord. These will be dropped
        # from self.apd_map as well
        for channel in self.apd_map:
            torus_hall_name = channel[2]
            try:
                fibre_loc_index = fibre_coord_names.index(torus_hall_name)
                self.fibre_conf.append(channel + fibre_coord[fibre_loc_index][1:])
            except ValueError:
                print("fibre_conf")

        
        # ordering the fibre_conf
        self.fibre_conf = sorted(self.fibre_conf, key = lambda x: x[5])
        self.apd_map = [channel[:4], channel in self.fibre_conf]

    #-----------------------FINDING CHANNEL DATA PER NAME----------------------
    
    def get_channel_id(self, input_channel, map_to = 4,
                       input_spectrometer_tracking=False):
        """
        Maps the input_channel to a different naming convention.
        INPUT: input_channel can be a string or a list of strings in the form:
                      BES-ADCxx
                      JPF/DH/KY6D-DOWN:xxx
                      JPF/DH/KY6D-FAST:xxx
                      BES-x-x
                      4AR/4BR...
                      KY6-16
                      x (spectrometer tracking number, see below)
               input_spectrometer_tracking: THE FUNCTION DOES NOT TAKE THE
                   SPECTROMETER TRACKING NUMBER PER DEFAULT AS AN INPUT. (This
                   may otherwise cause problems.) If the input is the spectrometer
                   tracking number, that should be noted with the
                   input_spectrometer_tracking=True keyword
                
        OUTPUT
            map_to = 0: input_channels name in BES-x-x form
                   = 1: input_channels name in 5AR, 4BL, etc. form
                   = 2: input_channels name in KY6-XX. form
                   = 3: input_channels name in spectrometer tracking number (string)
                   = 4: input_channels name in BES-ADCxx form
                   = 5: input_channels name in JPF/DH/KY6D-DOWN:xxx form
                   = 6: input_channels name in JPF/DH/KY6D-FAST:xxx form
        """
        if type(input_channel) == str:
            return self.get_channel_id_single(input_channel,
                                              map_to = map_to,
                                              input_spectrometer_tracking=input_spectrometer_tracking)
        elif type(input_channel) == list:
            output = [self.get_channel_id_single(channel, map_to = map_to,
                      input_spectrometer_tracking=input_spectrometer_tracking)
                      for channel in input_channel]
            return output
        raise ValueError("The input_channel should be either a string or a list of strings")

    def get_channel_id_single(self, input_channel, map_to = 4,
                              input_spectrometer_tracking=False):
        """
        The same as get_channel_id, but input_channel has to be a string
        """
        if input_channel[:7] == "BES-ADC":
            search_for = "0"+input_channel[7:]
            print(search_for)
            if len(search_for) != 3 or search_for.isdigit() is False:
                raise ValueError(input_channel+" is not a valid name")
        elif input_channel[:17] == "JPF/DH/KY6D-DOWN:":
            search_for = input_channel[17:]
            if len(search_for) != 3 or search_for.isdigit() is False:
                raise ValueError(input_channel+" is not a valid name")
        elif input_channel[:17] == "JPF/DH/KY6D-FAST:":
            search_for = input_channel[17:]
            if len(search_for) != 3 or search_for.isdigit() is False:
                raise ValueError(input_channel+" is not a valid name")
        else:
            search_for = input_channel

        output_starter = ""
        if map_to == 4:
            output_starter = "BES-ADC"
        elif map_to == 5:
            output_starter = "JPF/DH/KY6D-DOWN:"
        elif map_to == 6:
            output_starter = "JPF/DH/KY6D-FAST:"

        if self.apd_map is None:
            self.read_apd_map()
        if input_spectrometer_tracking is False:
            channel_index = np.where([(search_for in channel_indices[:3]+channel_indices[4:])
                                      for channel_indices in self.apd_map])
        else:
            channel_index = np.where([(search_for == channel_indices[3])
                                      for channel_indices in self.apd_map])

        if len(channel_index[0]) == 1:
            if map_to == 4:
                return output_starter + self.apd_map[int(channel_index[0])][min(map_to,4)][1:]
            else:
                return output_starter + self.apd_map[int(channel_index[0])][min(map_to,4)]
        elif len(channel_index[0]) > 1:
            matching_channels = [output_starter+self.apd_map[int(index)][min(map_to,4)] for index in channel_index[0]] 
            warnings.warn("Multiple matching channel names for "+ input_channel +
                             ":" + str(matching_channels))
            return matching_channels
        else:
            raise ValueError("No matching channel names for "+ input_channel +
                             ". This may happen if your input is a spectrometer tracking number. "+\
                             "Try input_spectrometer_tracking=True keyword if so.")


    def select_signals(self, chspec, options={}):
        '''Reads and returns the location of the signals defined in chspec
        INPUT: chspec the list of channels for which the location of the data is needed
                      Unix style regular expressions are allowed with * and []
                      Can also be a list of data names, eg. ['BES-1-1','BES-1-3']
                      The naming conventions for the channels can be in ONE of the following forms
                          BES-ADCxx
                          JPF/DH/KY6D-DOWN:xxx
                          JPF/DH/KY6D-FAST:xxx
                          BES-x-x
                          4AR/4BR...
                          KY6-16
                          x (spectrometer tracking number string)
               options:
                    Mode: Fast/Down: whether to get the fast or the downsampled data
                    Map to: a number defining the form of the returned strings
                            see map_to keyword of get_channel_id. Should not be
                            set together with "Mode"
        OUTPUT: 2d list of the signals and the map to their location/other names
        '''
        options_default = {'Mode': None,
                           'Map to': None}
        options = {**options_default, **options}
        if (options["Mode"] is None) and (options["Map to"] is None):
            raise ValueError("Either 'Mode' or 'Map to' should be set in the options keyword")
        elif (options["Mode"] is not None) and (options["Map to"] is not None):
            raise ValueError("'Mode' and 'Map to' should not both be set in the options" +
                             "keyword at the same time")
        
        if options["Mode"] == "Fast":
            options["Map to"] = 6
        elif options["Mode"] == "Down":
            options["Map to"] = 5
        
        #Tries to find the signals under one of the naming conventions
        for naming_convention in range(0, len(self.apd_map[0])):
            try:
                if naming_convention == 4:
                    # this is needed for the BES-ADCxx, JPF/DH/KY6D-DOWN:xxx,
                    # JPF/DH/KY6D-FAST:xxx naming conventions
                    signal_list_temp = [channel[naming_convention] for channel in self.apd_map]
                    try:
                        signal_list = self.get_channel_id(signal_list_temp, map_to = 4)
                        signal_proc, signal_index = flap.select_signals(signal_list, chspec)
                    except ValueError:
                        try:
                            signal_list = self.get_channel_id(signal_list_temp, map_to = 5)
                            signal_proc, signal_index = flap.select_signals(signal_list, chspec)
                        except ValueError:
                            signal_list = self.get_channel_id(signal_list_temp, map_to = 6)
                            signal_proc, signal_index = flap.select_signals(signal_list, chspec)
                else:
                    signal_list = [channel[naming_convention] for channel in self.apd_map] 
                    signal_proc, signal_index = flap.select_signals(signal_list, chspec)
                break
            except ValueError:
                pass
        if not("signal_proc" in locals()):
            raise ValueError("The channel definition "+str(chspec)+" did not match the naming conventions.")

        # Obtains the name of the channels in the form requested
        # in Options["Map to"] (or Options["Mode"])
        if naming_convention == 3:
            new_signal_proc = self.get_channel_id(signal_proc, map_to = options["Map to"],
                               input_spectrometer_tracking=True)
        else:
            new_signal_proc = self.get_channel_id(signal_proc, map_to = options["Map to"],
                               input_spectrometer_tracking=False)

        return (signal_proc, new_signal_proc)

    def amplitude_calib(self, signal_data, options={}):
        options_default = {'Amplitude calibration': flap.config.get("Module JET_LIBES","Amplitude calibration")}
        options = {**options_default, **options}

    def spatial_calib(self, signal_data, options={}):
            options_default = {'Spatial calibration': flap.config.get("Module JET_LIBES","Spatial calibration")}
            options = {**options_default, **options}

def online_get_config(self_object, config=None, options={}):
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
    
def offline_get_config(self_object, config=None):
    raise NotImplementedError("Obtaining the configuration from offline files"+
                              " is not implemented yet")

def online_get_chopper(self_object, options={}):
    #obtaining the beam energy
    self_object.get_config(config='KY6-AccVoltage')
    self_object.get_config(config='KY6-EmitterVoltage')
    # thedataobjects should have the same name to add them together
    self_object.data['KY6-AccVoltage'].data_unit.name = "Energy"
    self_object.data['KY6-AccVoltage'].data_unit.unit = "eV"
    self_object.data['KY6-EmitterVoltage'].data_unit.name = "Energy"
    self_object.data['KY6-EmitterVoltage'].data_unit.unit = "eV"
    energy = self_object.data['KY6-EmitterVoltage']+self_object.data['KY6-AccVoltage']
    
    # getting the measurement time during the plasma phase
    self_object.get_config(config='KY6-ChopOn', options={"UID": "KY6-team"})
    self_object.get_config(config='KY6-ChopOff', options={"UID": "KY6-team"})
    start_meas = np.min(self_object.data['KY6-ChopOn'].data)
    end_meas = np.max(self_object.data['KY6-ChopOff'].data)
    
    #slicing out the beam on times from the energy variable and averaging it
    energy_mean=np.mean(energy.slice_data(slicing={'Time':flap.Intervals(start_meas, end_meas)}).data)
    self_object.energy_mean = int(energy_mean/1000+0.5) # energy of the beam is rounded and defined to keV accuracy

    # generating the chopper dataobjects for slicing
    self_object.chopper_timing_data_object(options=options)

    # getting the calibration time for the beam-in-gase phase
    self_object.get_config(config='KY6-CalibTime', options={"UID": "KY6-team"})
    start_calib = np.min(self_object.data['KY6-CalibTime'].data)
    end_calib = np.max(self_object.data['KY6-CalibTime'].data)

    for channel in range(0,32):
        try:
            location = "JPF/DH/KY6D-DOWN:"+str(channel+1).zfill(3)
            data = jetapi.getsignal(self_object.exp_id, location)
            break
        except ValueError:
            data = None
    if data is None:
        raise ValueError("No JPF KY6D data found for shot "+\
                         str(self_object.exp_id))
    return data

def offline_get_chopper(self_object, config=None, options={}):
    pass

def jet_libes_get_data(exp_id=None, data_name=None, no_data=False,
                       options={}, coordinates=None, data_source=None):
    """ Data read function for the JET Li-BES diagnostic
    data_name: BES-ADCxx, JPF/DH/KY6D-DOWN:xxx, JPF/DH/KY6D-FAST:xxx,
    BES-x-x, 4AR/4BR..., KY6-16
    exp_id: Experiment ID
    Unix style regular expressions are allowed with * and []
                       Can also be a list of data names, eg. ['BES-1-1','BES-1-3']
    coordinates: List of flap.Coordinate() or a single flap.Coordinate
                 Defines read ranges
                     'Time': The read times
                     'Sample': To be implemented for reading from files
                     Only a single equidistant range is interpreted in c_range.
    options:
        'Amplitude calibration':
                       "Spectrometer PPF" - use the spectrometer calibration data available as ppf
                       "Spectrometer cal" - use the spectrometer calibration data to calculate on the fly
                       "APDCAM" - use APDCAM data for calibration
                       None - the data is not calibrated
        'Spatial calibration': 
                       "Spectrometer PPF" - use the spectrometer calibration data available as ppf
                       "Spectrometer cal" - use the spectrometer calibration data to calculate on the fly
                       None - the data is not calibrated                       
        'Start delay': delay the chopper interval start point,
        'End delay': delay the chopper interval end point
        'Mode': Fast/Down whether to get the data from the DH/FAST:xxx or
                DH/Down:xxx source
        'UID': The user id to get the data from, can be given as a team, if that
               team is defined in the flap_defaults.cfg
        'Cache Data': Whether to Cache the data when it is downloaded from the JET server
    OUTPUT: signal_data - flap DataObject with the data and coordinates 
            temporal data: ["Time", "Sample"] for the temporal data
            channel names: [Pixel Number", "Light Splitting Optics", "Torus Hall", 
                            "Spectrometer Track", "ADC", "JPF", "LPF"]
            spatial: "Device Z" if 'Spatial calibration' is set in the options
    """
    if (exp_id is None):
        raise ValueError('exp_id should be set for JET LiBES.')

    options_default = {'Datapath': flap.config.get("Module JET_LIBES","Datapath"),
                       'Amplitude calibration': flap.config.get("Module JET_LIBES","Amplitude calibration"),
                       'Spatial calibration': flap.config.get("Module JET_LIBES","Spatial calibration"),
                       'Start delay': flap.config.get("Module JET_LIBES","Start delay"),
                       'End delay': flap.config.get("Module JET_LIBES","End delay"),
                       "UID": "KY6-team",
                       "Cache Data": True}
    options = {**options_default, **options}

    configs = LiBESConfig(exp_id)

    # Ensuring that the data name is a list
    if type(data_name) is not list:
        chspec = [data_name]
    else:
        chspec = data_name

    # Finding read_range (timerange) and read_samplerange
    read_range = None
    if (coordinates is not None):
        if (type(coordinates) is not list):
             _coordinates = [coordinates]
        else:
            _coordinates = coordinates
        for coord in _coordinates:
            if (type(coord) is not flap.Coordinate):
                raise TypeError("Coordinate description should be flap.Coordinate.")
            if (coord.unit.name is 'Time'):
                if (coord.mode.equidistant):
                    read_range = np.asarray([float(coord.c_range[0]),float(coord.c_range[1])])
                    if (read_range[1] <= read_range[0]):
                        raise ValueError("Invalid read timerange.")
                else:
                    raise NotImplementedError("Non-equidistant Time axis is not implemented yet.")
                break

    # Finding the desired channels
    if options['Amplitude calibration'] == "Spectrometer PPF" or \
       options['Spatial calibration'] == "Spectrometer PPF":
        # in this case one has to take into account that there are dead channels
        # and there were mislabeled ones
        configs.get_fibre_config()
    else:
        # only the dead channels are omitted from channels that can be used
        configs.read_apd_map()
    signal_name, signal_loc = configs.select_signals(chspec, options=options)

    # Obtaining the data for all channels in signal_name
    signal_data = None
    for channel in signal_loc:
        if signal_data is None:
            signal_data = jetapi.getsignal(exp_id, channel, no_data = no_data, options=options)
        else:
            chan_dataobj = jetapi.getsignal(exp_id, channel, no_data = no_data, options=options)
            if len(signal_data.data.shape) == 1:
                signal_data.data=np.stack((signal_data.data, chan_dataobj.data), axis=1)
            else:
                signal_data.data=np.hstack((signal_data.data, np.expand_dims(chan_dataobj.data, axis=1)))
    signal_data.shape = signal_data.data.shape
    signal_data.data_unit.name = "Signal"
    signal_data.data_unit.unit = "Volt"
    signal_data.data_title = "JET APDCAM ("+data_name+")"


    # Adding a coordinate for all versions of the signal_naming
    signal_names = ["Pixel Number", "Light Splitting Optics", "Torus Hall",
                    "Spectrometer Track", "ADC", "JPF", "LPF"]
    for naming_convention in range(0, len(configs.apd_map[0])):
        new_signal_name = configs.get_channel_id(signal_name,
                                                 map_to = naming_convention)
        signal_name_coord = flap.Coordinate(name=signal_names[naming_convention],
                                            unit='n.a.',
                                            mode=flap.CoordinateMode(equidistant=False),
                                            values=new_signal_name,
                                            dimension_list=[1],
                                            shape=[len(new_signal_name)])
        signal_data.add_coordinate_object(signal_name_coord)
    
    # Adding a Sample coordinate for the time axis
    sample_coord = flap.Coordinate(name='Sample',
                                   unit='n.a.',
                                   mode=flap.CoordinateMode(equidistant=True),
                                   start=0,
                                   step=1,
                                   dimension_list=[0])
    signal_data.add_coordinate_object(sample_coord)


    # Amplitude calibration of the data
    if options['Amplitude calibration'] is not None:
            signal_data = configs.amplitude_calib(signal_data, options=options)  

    # Spatial calibration of the data
    if options['Spatial calibration'] is not None:
            signal_data = configs.spatial_calib(signal_data, options=options)  

    return d


def find_absolute_path(filename):
    if filename[:2] == "."+os.path.sep:
        location = os.getcwd()
        filename = location+os.path.sep+filename[2:]
    elif filename[:3] == ".."+os.path.sep:
        curr_path = os.getcwd()
        location = os.path.sep.join(curr_path.split(os.path.sep)[:-1])
        filename = location+os.path.sep+filename[3:]
    return filename

def add_coordinate():
    raise NotImplementedError("Adding coordinates is not JET implemented")

def register(data_source=None):
    flap.register_data_source('JET_LIBES', get_data_func=jet_libes_get_data, add_coord_func=add_coordinate)

if __name__ == "__main__":

    conf = LiBESConfig(95000)
#    conf = LiBESConfig(95000)
    conf.get_config(config="KY6-Z", options={"UID": "KY6-team"})
    conf.get_config(config="PPF/KY6I/BINV", options={"UID": "KY6-team"})
#    conf.data["KY6-Z"].plot()
#    conf.read_apd_map()
    conf.get_fibre_config()
#    conf.read_apd_map()
    print(conf.data["KY6-Z"].data.shape)
    print(len(conf.fibre_conf))
#    print(len(conf.apd_map))
#    out1, out2= conf.select_signals("KY6-[6-10]", options={"Mode": "Fast"})

#    jet_libes_get_data(exp_id=95000, data_name="BES-1-[4-6]", no_data=False, options={'Mode': 'Fast'})
#    conf.get_config(config="KY6-CrossCalib", options={"UID": "KY6-team"})
#    register()
#    
#    data = flap.get_data('W7X_ABES',
#                         name='ABES-'+str(channel),
#                         exp_id=shotID,
#                         object_name='ABES-'+str(channel)+' DATA',
#                         options={'Calibration': True,
#                                  'Calib. path': '/data/W7-X/APDCAM/cal/',
#                                  'Scaling': 'Volt'})
#    data=conf.get_chopper()