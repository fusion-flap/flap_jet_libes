# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:44:49 2019

@author: Miklos Vecsei

This is the flap module for JET Lithium BES diagnostic
"""

import os
import warnings
import copy
import gc
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
        self.spect_ppf_map = None # the spectrometer channels, their z coordinate and Beam-into-gas calibration factors

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
            "Time": For which time window the chopper data should be created
        """
        options_default = {"Start delay": 0, "End delay": 0, "Time": None}
        options = {**options_default, **options}

        # The distance of the chopper plates to the Z=0 plane
        beam_distance = float(flap.config.get("Module JET_LIBES","Chopper plate height"))
        mass_li = 1.1525801e-26
        beam_velocity = np.sqrt(2*self.energy_mean*1.60217662e-16/mass_li)
        time_delay = beam_distance/beam_velocity
        
        mode = flap.CoordinateMode(equidistant=False, range_symmetric=False)
        
        choff = self.data['KY6-ChopOff'].data
        chon = self.data['KY6-ChopOn'].data
        choff[:,0] = choff[:,0] + options["Start delay"]*1e-6
        choff[:,1] = choff[:,1] + options["End delay"]*1e-6
        chon[:,0] = chon[:,0] + options["Start delay"]*1e-6
        chon[:,1] = chon[:,1] + options["End delay"]*1e-6
        slicer = choff
        if len(chon)>len(choff):
            slicer = chon
        if options["Time"][0] is not None and options["Time"][1] is not None:
            indices = np.logical_and(slicer[:,0]>=options["Time"][0],
                                     slicer[:,0]<=options["Time"][1])
            choff = choff[indices[:choff.shape[0]],:]
            chon = chon[indices[:chon.shape[0]],:]           
            
        coff_time = flap.Coordinate(name='Time',
                                 unit='Second',
                                 mode=mode,
                                 values = choff[:,0]+time_delay,
                                 value_ranges=[np.zeros(choff.shape[0]),
                                               choff[:,1]-\
                                               choff[:,0]],
                                 shape = [choff.shape[0]],
                                 dimension_list=[0])
        self.chopper_off = flap.DataObject(data_shape=[choff.shape[0]],
                                           data_unit=flap.Unit(name='Interval', unit='n.a.'),
                                           coordinates=[coff_time],
                                           data_source = 'JET_LIBES',
                                           exp_id = self.exp_id
                                           )

        con_time = flap.Coordinate(name='Time',
                                 unit='Second',
                                 mode=mode,
                                 values = chon[:,0]+time_delay,
                                 value_ranges=[np.zeros(chon.shape[0]),
                                               chon[:,1]-\
                                               chon[:,0]],
                                 shape = [chon.shape[0]],
                                 dimension_list=[0])
        self.chopper_on = flap.DataObject(data_shape=[chon.shape[0]],
                                          data_unit=flap.Unit(name='Interval', unit='n.a.'),
                                          coordinates=[con_time],
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
            exp_id_limits[last_map_index, 1] = int(self.exp_id)+1
            # the map for the current exp_id is given by
            map_index = np.where((exp_id_limits[:,0]-int(self.exp_id) <= 0) *\
                                 (exp_id_limits[:,1]-int(self.exp_id) >= 0))[0]
            
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
            exp_id_limits[last_map_index, 1] = int(self.exp_id)+1
            exp_index = np.where((exp_id_limits[:,0]-int(self.exp_id) <= 0) *\
                                 (exp_id_limits[:,1]-int(self.exp_id) >= 0))[0]

            # getting the broken channels
            exp_broken_channels = [channels.split(",") for channels in \
                                   all_broken_channels[1+int(exp_index)].split("\n")[0].split(":")[1:]][0]
            exp_broken_channels = [channel for channel in exp_broken_channels if channel != [""]]
            
            # popping out the broken channels from self.apd_map
            if len(exp_broken_channels)>0:
                for channel in exp_broken_channels:
                    channel_index = np.where([(channel in channel_indices)
                                                  for channel_indices in self.apd_map])[0]
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
            exp_id_limits[last_setup_index, 1] = int(self.exp_id)+1
            # the map for the current exp_id is given by
            setup_index = np.where((exp_id_limits[:,0]-int(self.exp_id) <= 0) *\
                                   (exp_id_limits[:,1]-int(self.exp_id) >= 0))[0]
            
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
                elif len(channel)>1:
                    fibre_coord.remove(channel)
                    channel = channel[0]
                    fibre_coord.append([channel])
            
            fibre_coord = [channel[0].split(",") for channel in fibre_coord]
            #removing the empty strings
            for channel in fibre_coord:
                channel[1] = float(channel[1])
                channel[2] = float(channel[2])
                
        return 
    
    def get_fibre_config(self):
        '''
            The function for reading and merging all of the fibre setup files.
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
                pass # the channel is missing rom the fibre_coord file
        # ordering the fibre_conf
        self.fibre_conf = sorted(self.fibre_conf, key = lambda x: x[5])
        return self.fibre_conf

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
    
    def get_spect_ppf_map(self):
        if hasattr(self, "KY6-Z") is False:
            self.get_config(config="KY6-Z", options={"UID":"KY6-team"})
        if hasattr(self, "KY6-Bind") is False:
            self.get_config(config="KY6-Bind", options={"UID":"KY6-team"})
        if hasattr(self, "KY6-CrossCalib") is False:
            self.get_config(config="KY6-CrossCalib", options={"UID":"KY6-team"})
        self.spect_ppf_map = np.zeros((3,self.data["KY6-Bind"].data.shape[1]))
        self.spect_ppf_map[0,:]=self.data["KY6-Bind"].data+1
        self.spect_ppf_map[1,:]=self.data["KY6-Z"].data
        self.spect_ppf_map[2,:]=self.data["KY6-CrossCalib"].data/np.mean(self.data["KY6-CrossCalib"].data)       

    def amplitude_calib(self, signal_data, options={}):
        options_default = {'Amplitude calibration': flap.config.get("Module JET_LIBES","Amplitude calibration")}
        options = {**options_default, **options}

        if options['Amplitude calibration'] == "Spectrometer BIG":
            if self.spect_ppf_map is None:
                self.get_spect_ppf_map()
            channels = copy.deepcopy(signal_data.get_coordinate_object("Spectrometer Track"))
            if len(channels.dimension_list)>1:
                raise ValueError("Spectrometer BIG Amplitude calibration only works if the channels only vary along one dimension")
            chan_index=0
            missing_data_from_spect = False
            slicer = [slice(None)]*len(signal_data.shape)
            for channel in channels.values:
                slicer[channels.dimension_list[0]] = chan_index
                if channel =="0":
                    signal_data.data[tuple(slicer)] = 0
                    missing_data_from_spect = True
                else:
                    signal_data.data[tuple(slicer)] = signal_data.data[tuple(slicer)]*\
                        self.spect_ppf_map[2,np.where(self.spect_ppf_map[0,:]==int(channel))][0,0]
                chan_index = chan_index+1
            if missing_data_from_spect is True:
                    signal_data.info = signal_data.info + "Could not obtain Spectrometer PPF Amplitude calibration factor for a number of channels."+\
                                       " They were not plugged into the spectrometer./n"
            signal_data.data_unit.unit = "a.u."

        elif options['Amplitude calibration'] == "Spectrometer PPF":
            #Doing first a calibration with the BIG
            o = copy.deepcopy(options)
            o['Amplitude calibration'] = "Spectrometer BIG"
            signal_data = self.amplitude_calib(signal_data, options=o)
            # Ideally the previous part would be enough for the amplitude calibration
            # but the calibration coefficients of the apdcam and the spectrometer are
            # quite different, therefore a relative calibration between the two is also
            # necessary. This one should not vary too much between shots as long the
            # components of the diagnostic do not move.
            if self.spect_ppf_map is None:
                self.get_spect_ppf_map()
            channels = copy.deepcopy(signal_data.get_coordinate_object("Spectrometer Track"))

            time_coord = signal_data.coordinate("Time")[0]
            time_window = [np.min(time_coord), np.max(time_coord)]
            
            #getting the spectrometer light profiles:
            if 'PPF/KY6I/AINT' not in self.data:
                self.get_config(config="KY6-AINT", options={"UID":"KY6-team"})
                spect_track_coord_val = np.asarray([str(int(tracknum)) for tracknum in self.spect_ppf_map[0,:]])
                spect_track_coord = flap.Coordinate(name="Spectrometer Track",
                                                unit='n.a.',
                                                mode=flap.CoordinateMode(equidistant=False),
                                                values=spect_track_coord_val,
                                                dimension_list=[1],
                                                shape=spect_track_coord_val.shape)
                self.data["KY6-AINT"].add_coordinate_object(spect_track_coord)
            spectrometer_data = self.data["KY6-AINT"].slice_data(slicing={"Time":flap.Intervals(time_window[0],time_window[1])})
            
            # in case that the spectrometer time window is shorter
            spectrometer_time = spectrometer_data.coordinate("Time")[0]
            spect_time_window = [np.min(spectrometer_time), np.max(spectrometer_time)]
            signal_data_calib = signal_data.slice_data(slicing={"Time":flap.Intervals(spect_time_window[0],spect_time_window[1])})
            
            slicer = [slice(None)]*len(signal_data.shape)
            chan_index=0
            for channel in channels.values:
                slicer[channels.dimension_list[0]] = chan_index
                if not (channel == "0"):
                    apdcam_chan_mean = np.mean(signal_data_calib.slice_data(slicing={"Spectrometer Track":channel}).data)
                    spectrometer_chan_mean = np.mean(spectrometer_data.slice_data(slicing={"Spectrometer Track":channel}).data)
                    signal_data.data[tuple(slicer)] = signal_data.data[tuple(slicer)]*spectrometer_chan_mean/apdcam_chan_mean
                chan_index = chan_index+1
        return signal_data 
                    

    def spatial_calib(self, signal_data, options={}):
            options_default = {'Spatial calibration': flap.config.get("Module JET_LIBES","Spatial calibration"),
                               'Sort data with Device Z': True}
            options = {**options_default, **options}
            if options['Spatial calibration'] == "Spectrometer PPF":
                if self.spect_ppf_map is None:
                    self.get_spect_ppf_map()

                channels = copy.deepcopy(signal_data.get_coordinate_object("Spectrometer Track"))
                if len(channels.dimension_list)>1:
                    raise ValueError("Spectrometer PPF Spatial calibration only works if the channels only vary along one dimension")
                coord_val = np.zeros(len(channels.values))
                missing_data_from_spect = False
                index=0
                for channel in channels.values:
                    if channel == "0":
                        missing_data_from_spect = True
                        coord_val[index] = None # the channel is not connected to the spectrometer
                    else:
                        coord_val[index] = self.spect_ppf_map[1,np.where(self.spect_ppf_map[0,:]==int(channel))]
                    index = index+1
                if missing_data_from_spect is True:
                    signal_data.info = signal_data.info + "Could not obtain Spectrometer PPF Z coordinate for a number of channels."+\
                                       " They were not plugged into the spectrometer./n"
                    coord_val[np.where(coord_val==0)] = None
                channels.unit.name =  "Device Z"
                channels.unit.unit =  "m"
                channels.values = coord_val
                signal_data.add_coordinate_object(channels)
            
            if options['Sort data with Device Z'] is True:
                z_coord = signal_data.get_coordinate_object("Device Z")
                if len(z_coord.dimension_list)>1:
                    raise ValueError ("Can not sort data along Device Z, coordinate varies over multiple dimensions.")
                for coordinate in signal_data.coordinates:
                    if z_coord.dimension_list[0] in coordinate.dimension_list and \
                    not (z_coord.dimension_list == coordinate.dimension_list):
                        raise ValueError("Can not sort along Device Z, "+coordinate.name+" varies along"+
                                         "multiple dimensions of the data and also varies with Device Z")
            
                slicer = [slice(None)]*len(signal_data.shape)
                newz_dataindex = np.zeros([len(z_coord.values),2])
                newz_dataindex[:,0] = copy.deepcopy(z_coord.values)
                newz_dataindex[:,1] = np.arange(len(z_coord.values))
                newz_dataindex = newz_dataindex[np.argsort(newz_dataindex[:,0])]
                signal_data.get_coordinate_object("Device Z").values = newz_dataindex[:,0]
                slicer[z_coord.dimension_list[0]] = tuple(newz_dataindex[:,1].astype("int"))
                signal_data.data = signal_data.data[tuple(slicer)]
                # need to change the order of every other coordinate that is 
                for coordinate in signal_data.coordinates:
                    if z_coord.dimension_list == coordinate.dimension_list:
                        coordinate.values = coordinate.values[newz_dataindex[:,1].astype("int")]
                    
            return signal_data
                

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
    o = copy.deepcopy(options)
    o["UID"] = "KY6-team"
    self_object.get_config(config='KY6-ChopOn', options=o)
    self_object.get_config(config='KY6-ChopOff', options=o)
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

def get_signal_data(exp_id=None, data_name=None, no_data=False,
                       options={}, coordinates=None, data_source="JET_LIBES"):
    """ Obtains the signal data fo a JET channel, further info in jet_libes_get_data
    """
    
    configs = LiBESConfig(exp_id)
    
    # Finding the desired channels
    configs.read_apd_map()
    signal_name, signal_loc = configs.select_signals(data_name, options=options)

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
    signal_data.data_title = "JET APDCAM ("+str(data_name)+")"


    # Adding a coordinate for all versions of the signal_naming for multidimensional data
    if len(signal_data.shape) > 1:
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
    
    signal_data.get_coordinate_object("Time").unit.unit = "Second"


    # Amplitude calibration of the data
    if options['Amplitude calibration'] is not None:
            signal_data = configs.amplitude_calib(signal_data, options=options)  

    # Spatial calibration of the data
    if options['Spatial calibration'] is not None:
            signal_data = configs.spatial_calib(signal_data, options=options)
    
    return signal_data

def proc_chopsignals_single(exp_id=None,timerange=None,signals='KY6-1', test=None, on_options={},
                             off_options={},  dataobject=None, options={}):
    """ Calculate signals in beam on and beam/off phases of the measurement of a single channel and
        correct the beam-on phases with the beam-off. The result is "LIBES" and "LIBES_back" data object
        in the FLAP storage.
        INPUT:
            There are two modes of this function:
                The first one obtains the data directly from a measurement file
                    exp_id: exp_id (no default)
                    signals: List of measurement signals. Default is KY6-*
                The second one takes a dataobject as an input
                    dataobject: the input dataobject file
            timerange: Time range to process. Default is all times.
            test: Plot test plots if True
            options:
                Average Chopping Period: Whether the data should be averaged for a single chopper time period. Per 
                                         default this is True. If this option is False then all measurement time points
                                         of the beam on state is saved. This could be useful if slow beam chopping is
                                         used and the background signal is reasonably constant relative to the analyzed
                                         process.
        OUTPUT: The background subtracted Li-BES data
    """
    # Obtaining the chopper data
    if dataobject is not None:
        exp_id = dataobject.exp_id

    o = copy.deepcopy(on_options)
    o["State"] = {"Chop":0}
    d_beam_on=flap.get_data('JET_LIBES',
                            exp_id=exp_id,
                            name='Chopper_time',
                            coordinates={'Time':timerange},
                            options=o,
                            object_name='Beam_on',
                            )
    o = copy.deepcopy(off_options)
    o["State"] = {"Chop":1}
    d_beam_off=flap.get_data('JET_LIBES',
                             exp_id=exp_id,
                             name='Chopper_time',
                             coordinates={'Time':timerange},
                             options=o,
                             object_name='Beam_off',
                             )

    # Background subtraction
    # in this case the passed dataobject is used and the only the copper data is obtained from file
    # The code can only handle dataobjects with the data of a single channel
    if timerange is not None:
        dataobject = dataobject.slice_data(slicing={'Time': flap.Intervals(timerange[0],timerange[1])})
    dataobject_beam_on = dataobject.slice_data(slicing={'Time': d_beam_on})
    # For dataobject_beam_on.data the 0 dimension is along a constant 'Start Time in int(Time)' and 
    # "Rel. Time in int(Time)" varies. For dimension 1 it is 'Start Time in int(Time)' that varies
    dataobject_beam_on = process_chopped_dataobject(dataobject_beam_on, options=options)

    dataobject_beam_off = dataobject.slice_data(slicing={'Time': d_beam_off}, options={'Partial intervals':False})
    dataobject_beam_off = process_chopped_dataobject(dataobject_beam_off, options={'Average Chopping Period': True})

    dataobject_background = dataobject_beam_off.slice_data(slicing={'Time': dataobject_beam_on},
                                                           options={'Inter': 'Linear'})
    if (test):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.scatter(dataobject_beam_on.get_coordinate_object("Time").values, dataobject_beam_on.data)
        plt.scatter(dataobject_beam_off.get_coordinate_object("Time").values, dataobject_beam_off.data)
        plt.plot(dataobject.get_coordinate_object("Time").values, dataobject.data)
        
    
    dataobject_beam_on.data -= dataobject_background.data.reshape(np.shape(dataobject_beam_on.data))

    # calculating the error for the beam off part
    dataobject_beam_off_error = copy.deepcopy(dataobject_beam_off)
    dataobject_beam_off_error.data = np.sqrt(dataobject_beam_off.error)
    background_error = dataobject_beam_off_error.slice_data(slicing={'Time': dataobject_beam_on}, options={'Inter': 'Linear'})
    background_error = background_error.data.reshape(np.shape(dataobject_beam_on.data))
    dataobject_beam_on.error = np.asarray(np.sqrt(dataobject_beam_on.error**2 + background_error**2))

    if (test):
        plt.plot(dataobject_beam_on.get_coordinate_object("Time").values,
                 dataobject_beam_on.data)

    return dataobject_beam_on


def proc_chopsignals(dataobject=None, timerange=None, test=None, on_options={},
                     off_options={},  options={}):
    """ Calculate signals in beam on and beam/off phases of the measurement and
        correct the beam-on phases with the beam-off. Further information in the comments of 
        proc_chopsignals_single
    """
    # checking, whether dataobject has the data for multiple channels
    naming_conventions = ["Pixel Number", "Light Splitting Optics", "Torus Hall",
                        "Spectrometer Track", "ADC", "JPF", "LPF"]
    channel_naming = []
    for name in naming_conventions:
        if name in dataobject.coordinate_names():
            channel_naming.append(name)
    
    if len(channel_naming)==0:
        processed_data = proc_chopsignals_single(dataobject=dataobject, timerange=timerange,
                                          test=test, on_options=on_options,
                                          off_options=off_options,  options=options)
    else:
        channels = dataobject.get_coordinate_object(channel_naming[0]).values
        for channel in channels:
            channel_data = dataobject.slice_data(slicing={channel_naming[0]: channel})
            if not ("processed_data" in locals()):
                processed_data = proc_chopsignals_single(dataobject=channel_data,
                                                         timerange=timerange,
                                                         test=test, on_options=on_options,
                                                         off_options=off_options,  options=options)
            else:
                partial_processed_data = proc_chopsignals_single(dataobject=channel_data, timerange=timerange,
                                          test=test, on_options=on_options,
                                          off_options=off_options,  options=options)
                if len(processed_data.data.shape) == 1:
                    processed_data.data=np.stack((processed_data.data, partial_processed_data.data), axis=1)
                    processed_data.error=np.stack((processed_data.error, partial_processed_data.error), axis=1)
                else:
                    processed_data.data=np.hstack((processed_data.data, np.expand_dims(partial_processed_data.data, axis=1)))
                    processed_data.error=np.hstack((processed_data.error, np.expand_dims(partial_processed_data.error, axis=1)))

#                del partial_processed_data
#                gc.collect()
        processed_data.shape = processed_data.data.shape
        # adding the channel coordinates
        channel_dimension = dataobject.get_coordinate_object(channel_naming[0]).dimension_list
        for coordinate in dataobject.coordinates:
            if coordinate.dimension_list == channel_dimension:
                naming_coord = copy.deepcopy(coordinate)
                naming_coord.dimension_list = [1]
                processed_data.add_coordinate_object(naming_coord)
    return processed_data

def process_chopped_dataobject(dataobject, options={}):
    ''' Processes the input channel data dataobject which has been already
    sliced with a chopper dataobject. It flattens the data and adds the proper
    Time coordinate. If requested it only averages the data over the chopping
    time periods.
    dataobject - the channel data sliced with a chopper DataObject
    options 'Average Chopping Period' - boolean,  whether to average the date
                                        over a chopping interval
    '''
    options_default = {'Average Chopping Period': True}
    options = {**options_default, **options}
    if options['Average Chopping Period'] is True:
        #calculating the error of the beam on part
        reltime_coord = dataobject.get_coordinate_object("Rel. Time in int(Time)")
        reltime_size = dataobject.data.shape[reltime_coord.dimension_list[0]]
        average = np.nanmean(dataobject.data, axis=reltime_coord.dimension_list[0], keepdims=True)
        beam_on_error = np.nansum((dataobject.data-average)**2, axis = reltime_coord.dimension_list[0])/(reltime_size-1)

        #the averaged density profile - #flap is currently not capable of doing this correctly
        # need to set up a time coordinate correctly
        regenerate_time_sample(dataobject)
        times = copy.deepcopy(dataobject.coordinate("Time")[0])[0]
        dataobject.coordinates = [dataobject.get_coordinate_object("Time")]

        #taking the average over the Rel.Time in int(Time) coordinate
        dataobject.data = average[0]
        dataobject.shape = dataobject.data.shape
        dataobject.error = np.sqrt(beam_on_error)
    else:
        add_absolute_time(dataobject)
        times = copy.deepcopy(dataobject.coordinate("Time")[0]).flatten()
        dataobject.data = dataobject.data.flatten()
        times = times[np.logical_not(np.isnan(dataobject.data))] # removing nans due to the padding of the slicer
        dataobject.data = dataobject.data[np.logical_not(np.isnan(dataobject.data))]
        dataobject.shape = dataobject.data.shape
        dataobject.error = np.zeros(dataobject.data.shape)
        dataobject.coordinates = [dataobject.get_coordinate_object("Time")]

    dataobject.get_coordinate_object("Time").values = times
    dataobject.get_coordinate_object("Time").shape = times.shape
    dataobject.get_coordinate_object("Time").dimension_list = [0]
    dataobject.get_coordinate_object("Time").mode.equidistant = False
    dataobject.get_coordinate_object("Time").start=None
    dataobject.get_coordinate_object("Time").step=None
    
    #ordering the data along the time coordinate
    time_data = np.asarray(sorted(zip(dataobject.get_coordinate_object("Time").values,
                                      dataobject.data)))
    dataobject.get_coordinate_object("Time").values = time_data[:,0]
    dataobject.data = time_data[:,1]
    
    return dataobject

def add_absolute_time(dataobject):
    """ Creates a coordinate 'Time' to the input dataobject from proc_chopsignals. This can be used for slicing the
            data
        INPUT:
            d - dataobject, with coordinates Start Time in int(Sample) and Rek. Time in int(Sample)
        OUTPUT: None, the coordinate is added to the original dataobject
    """
    # Finding the coordinates for the dataobject
    coords = [coord.unit.name for coord in dataobject.coordinates]
    coord_index = 0
    dimension_list_time = []
    for coordinate in coords:
        # will need to set for the time index
        if coordinate == 'Start Time in int(Time)':
            dimension_list_time = list(dataobject.coordinates[coord_index].dimension_list)+dimension_list_time
        elif coordinate == 'Rel. Time in int(Time)':
            dimension_list_time = list(dataobject.coordinates[coord_index].dimension_list)+dimension_list_time
        coord_index = coord_index+1
    dimension_list_time = list(np.unique(np.asarray(dimension_list_time)))

    name = 'Time'
    time_coord_value = dataobject.coordinate('Start Time in int(Time)')[0] +\
                       dataobject.coordinate('Rel. Time in int(Time)')[0]
    time_coord = flap.Coordinate(name=name, unit='Second', values=time_coord_value, shape=np.shape(time_coord_value),
                 mode=flap.CoordinateMode(equidistant=False), dimension_list=dimension_list_time)
    dataobject.add_coordinate_object(time_coord)

def regenerate_time_sample(d):
    """ Regenerate Time and Sample coordinate after chopper slicing aon Sample coordinate
        and averaging for chopper interval.
    """
    try:
        # Trying to get Time coordinate. If not present regenerating it
        d.get_coordinate_object('Time')
    except ValueError:
        ct = d.get_coordinate_object('Start Time in int(Time)')
        c_shift = d.get_coordinate_object('Rel. Time in int(Time)')
        if (not ct.mode.equidistant):
            try:
                ct.values += np.nanmean(c_shift.values, axis=c_shift.dimension_list[0])
            except IndexError:
                ct.values += c_shift.values
            #check if new coordinate is equidistant
            if len(ct.dimension_list) == 1:
                steps = ct.values[1:]-ct.values[:-1]
                accuracy = np.max(steps)/np.min(steps)
                if accuracy-1 < 1e-10:
                    ct.start = ct.values[0]
                    ct.step = np.mean(steps)
                    ct.mode.equidistant = True
        else:
            try:
                ct.start += c_shift.values[0]
            except IndexError:
                ct.start += c_shift.values
        ct.unit.name='Time'
        ct.unit.unit='Second'
        
        d.del_coordinate('Rel. Time in int(Time)')
#    try:
#        # Trying to get Sample coordinate. If not present regenerating it
#        d.get_coordinate_object('Sample')
#    except ValueError:
#        ct = d.get_coordinate_object('Start Sample in int(Sample)')
#        c_shift = d.get_coordinate_object('Rel. Sample in int(Sample)')
#        if (c_shift.dimension_list != []):
#            raise ValueError("Rel Sample in int(Sample) is not constant.")
#        if (not ct.mode.equidistant):
#            try:
#                ct.values += c_shift.values[0]
#            except IndexError:
#                ct.values += c_shift.values
#            #check if new coordinate is equidistant
#            if len(ct.dimension_list) == 1:
#                steps = ct.values[1:]-ct.values[:-1]
#                accuracy = np.max(steps)/np.min(steps)
#                if accuracy-1 < 1e-10:
#                    ct.start = ct.values[0]
#                    ct.step = np.mean(steps)
#                    ct.mode.equidistant = True
#        else:
#            try:
#                ct.start += c_shift.values[0]
#            except IndexError:
#                ct.start += c_shift.values
#        ct.unit.name='Sample'
#        d.del_coordinate('Rel. Sample in int(Sample)')
#    try:
#        d.del_coordinate('Interval(Sample)')
#        d.del_coordinate('Interval(Sample) sample index')
#    except ValueError:
#        pass

def jet_libes_get_data(exp_id=None, data_name=None, no_data=False, 
                       options={}, coordinates=None, data_source="JET_LIBES"):
    """ Data read function for the JET Li-BES diagnostic
    data_name: channel data can be reached under: "BES-ADCxx", "JPF/DH/KY6D-DOWN:xxx",
                                     "JPF/DH/KY6D-FAST:xxx", "BES-x-x", "4AR"/"4BR"..., "KY6-16"
               "Chopper_time" : To read the chopper state (will return intervals in Time and Sample ccoordinate)

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
        'State': Need it if "Chopper_time" is given {'Chop': 0} -beam on {'Chop': 1} - beam off
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
                       'Start delay': float(flap.config.get("Module JET_LIBES","Start delay (us)")),
                       'End delay': float(flap.config.get("Module JET_LIBES","End delay (us)")),
                       "UID": "KY6-team",
                       "Cache Data": True}
    options = {**options_default, **options}

    # Ensuring that the data name is a list
    if type(data_name) is not list:
        chspec = [data_name]
    else:
        chspec = data_name

#    # Finding read_range (timerange) and read_samplerange
#    read_range = None
#    if (coordinates is not None):
#        if (type(coordinates) is not list):
#             _coordinates = [coordinates]
#        else:
#            _coordinates = coordinates
#        for coord in _coordinates:
#            if (type(coord) is not flap.Coordinate):
#                raise TypeError("Coordinate description should be flap.Coordinate.")
#            if (coord.unit.name is 'Time'):
#                if (coord.mode.equidistant):
#                    read_range = np.asarray([float(coord.c_range[0]),float(coord.c_range[1])])
#                    if (read_range[1] <= read_range[0]):
#                        raise ValueError("Invalid read timerange.")
#                else:
#                    raise NotImplementedError("Non-equidistant Time axis is not implemented yet.")
#                break
    
    if not (chspec == ["Chopper_time"]):
        # In this case the actual channel data is obtained
        signal_data = get_signal_data(exp_id=exp_id, data_name=chspec, no_data=no_data,
                       options=options, coordinates=coordinates, data_source=data_source)
        return signal_data
    
    else:
        o=copy.deepcopy(options)
        if coordinates is not None:
            for coordinate in coordinates:
                if 'Time' == coordinate.unit.name:
                    o["Time"] = coordinate.c_range
        if "State" not in o:
            raise ValueError("State should be given in the options keyword if"+
                             " chopper dataobject is requested it.")
            return None
        elif not ((o["State"] == {'Chop': 0}) or (o["State"] == {'Chop': 1})):
            raise ValueError("options['State'] should be either {'Chop': 0} (beam on)"+
                             " or {'Chop': 1} (beam off).")
            return None
        configs = LiBESConfig(exp_id)
        configs.get_chopper(options=o)
        if options["State"]["Chop"]==0:
            return configs.chopper_off
        else:
            return configs.chopper_on   


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
