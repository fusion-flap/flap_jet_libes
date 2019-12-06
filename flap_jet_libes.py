# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:44:49 2019

@author: Miklos Vecsei

This is the flap module for JET Lithium BES diagnostic
"""

import os
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
        self.apd_map = None # the map for storing the fiber configurations

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
    def read_apd_map(self):
        filename = flap.config.get("Module JET_LIBES","APD map")
        if filename[:2] == "."+os.path.sep:
            location = os.path.dirname(os.path.abspath(__file__))
            filename = location+os.path.sep+filename[2:]
        elif filename[:3] == ".."+os.path.sep:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            location = os.path.sep.join(curr_path.split(os.path.sep)[:-1])
            filename = location+os.path.sep+filename[3:]

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
            map_index = np.where((exp_id_limits[:,0]-self.exp_id < 0) *\
                                 (exp_id_limits[:,1]-self.exp_id > 0))[0]
            
            # creating the multidimensional list for mapping
            apd_map = [channelID_variants.split(",") for \
                       channelID_variants in all_apd_maps[1+int(map_index)].split("\n")[1:]]
            #removing the empty strings
            self.apd_map = [channel for channel in apd_map if channel != [""]]

    def get_channel_id(self, input_channel, map_to = 4, input_spectrometer_tracking=False):
        """
        Maps the input_channel to a different naming convention.
        INPUT: input_channel can be in the form BES-ADCxx
                                                JPF/DH/KY6D-DOWN:xxx
                                                JPF/DH/KY6D-FAST:xxx
                                                BES-x-x
                                                4AR/4BR...
                                                KY6-16
               input_spectrometer_tracking: if the input is the spectrometer
                tracking number, that should be noted
        OUTPUT
            map_to = 0: input_channels name in BES-x-x form
                   = 1: input_channels name in 5AR, 4BL, etc. form
                   = 2: input_channels name in KY6-XX. form
                   = 3: input_channels name in spectrometer tracking number (string)
                   = 4: input_channels name in BES-ADCxx form
                   = 5: input_channels name in JPF/DH/KY6D-DOWN:xxx form
                   = 6: input_channels name in JPF/DH/KY6D-FAST:xxx form
        """
        input_starter = ""
        if input_channel[:7] == "BES-ADC":
            input_channel = "0"+input_channel[7:]
            input_starter = "BES-ADC"
            if len(input_channel[1:]) != 2 or input_channel.isdigit() is False:
                raise ValueError(input_starter+input_channel[1:]+" is not a valid name")
        elif input_channel[:17] == "JPF/DH/KY6D-DOWN:":
            input_channel = input_channel[17:]
            input_starter = "JPF/DH/KY6D-DOWN:"
            if len(input_channel) != 3 or input_channel.isdigit() is False:
                raise ValueError(input_starter+input_channel+" is not a valid name")
        elif input_channel[:17] == "JPF/DH/KY6D-FAST:":
            input_channel = input_channel[17:]
            input_starter = "JPF/DH/KY6D-FAST:"
            if len(input_channel) != 3 or input_channel.isdigit() is False:
                raise ValueError(input_starter+input_channel+" is not a valid name")

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
            channel_index = np.where([(input_channel in channel_indices)
                                      for channel_indices in self.apd_map])
        else:
            a = [channel_indices[3] for channel_indices in self.apd_map]
            print(a)
            channel_index = np.where([(input_channel == channel_indices[3])
                                      for channel_indices in self.apd_map])

        if len(channel_index[0]) == 1:
            if map_to == 4:
                return output_starter + self.apd_map[int(channel_index[0])][min(map_to,4)][1:]
            else:
                return output_starter + self.apd_map[int(channel_index[0])][min(map_to,4)]
        elif len(channel_index[0]) > 1:
            matching_channels = [output_starter+self.apd_map[int(index)][min(map_to,4)] for index in channel_index[0]] 
            raise ValueError("Multiple matching channel names for "+input_starter + input_channel +":" + str(matching_channels))
        else:
            raise ValueError("No matching channel names for "+input_starter + input_channel)
            
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
                       Can also be a list of data names, eg. ['BES-1-1','ABES-1-3']
    coordinates: List of flap.Coordinate() or a single flap.Coordinate
                 Defines read ranges
                     'Time': The read times
                     'Sample': To be implemented for reading from files
                     Only a single equidistant range is interpreted in c_range.
    options:
        'Calibration': "Spectrometer" - use the spectrometer calibration data
                       "APDCAM" - use APDCAM data for calibration
        'Start delay': delay the chopper interval start point,
        'End delay': delay the chopper interval end point
    """
    if (exp_id is None):
        raise ValueError('exp_id should be set for JET LiBES.')

    options_default = {'Datapath': flap.config.get("Module JET_LIBES","Datapath"),
                       'Calibration': flap.config.get("Module JET_LIBES","Calibration"),
                       'Start delay': flap.config.get("Module JET_LIBES","Start delay"),
                       'End delay': flap.config.get("Module JET_LIBES","End delay")}
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
    signal_list = configs.get_config(config='KY6-SignalList')
    ADC_list = config['ADC_list']
    try:
        signal_proc, signal_index = flap.select_signals(signal_list,chspec)
    except ValueError as e:
        raise e
    ADC_proc = []
    for i in signal_index:
        ADC_proc.append(ADC_list[i])


    scale_to_volts = False
    dtype = np.int16
    data_unit = flap.Unit(name='Signal',unit='Digit')
    if _options is not None:
        try:
            if (_options['Scaling'] == 'Volt'):
                scale_to_volts = True
                dtype = float
                data_unit = flap.Unit(name='Signal',unit='Volt')
        except (NameError, KeyError):
            pass

    try:
        offset_timerange = _options['Offset timerange']
    except (NameError, KeyError):
        offset_timerange = None

    if (offset_timerange is not None):
        if (type(offset_timerange) is not list):
            raise ValueError("Invalid Offset timerange. Should be list or string.")
        if ((len(offset_timerange) != 2) or (offset_timerange[0] >= offset_timerange[1])) :
            raise ValueError("Invalid Offset timerange.")
        offset_samplerange = np.rint((np.array(offset_timerange) - float(config['APDCAM_starttime']))
                                   / float(config['APDCAM_sampletime']))
        if ((offset_samplerange[0] < 0) or (offset_samplerange[1] >= config['APDCAM_samplenumber'])):
            raise ValueError("Offset timerange is out of measurement time.")
        offset_data = np.empty(len(ADC_proc), dtype='int16')
        for i_ch in range(len(ADC_proc)):
            fn = os.path.join(datapath, "Channel_{:03d}.dat".format(ADC_proc[i_ch] - 1))
            try:
                f = open(fn,"rb")
            except OSError:
                raise OSError("Error opening file: " + fn)
            try:
                f.seek(int(offset_samplerange[0]) * 2, os.SEEK_SET)
                d = np.fromfile(f, dtype=np.int16, count=int(offset_samplerange[1]-offset_samplerange[0])+1)
            except Exception:
                raise IOError("Error reading from file: " + fn)
            offset_data[i_ch] = np.int16(np.mean(d))
        if (scale_to_volts):
            offset_data = ((2 ** config['APDCAM_bits'] - 1) - offset_data) \
                        / (2. ** config['APDCAM_bits'] - 1) * 2
        else:
            offset_data = (2 ** config['APDCAM_bits'] - 1) - offset_data


    ndata = int(read_samplerange[1] - read_samplerange[0] + 1)

    if (no_data is False):
        if (len(ADC_proc) is not 1):
            data_arr = np.empty((ndata, len(ADC_proc)), dtype=dtype)
        for i in range(len(ADC_proc)):
            fn = os.path.join(datapath, "Channel_{:03d}.dat".format(ADC_proc[i] - 1))
            try:
                f = open(fn,"rb")
            except OSError:
                raise OSError("Error opening file: " + fn)

            try:
                f.seek(int(read_samplerange[0]) * 2, os.SEEK_SET)
            except Exception:
                raise IOError("Error reading from file: " + fn)

            if (len(ADC_proc) is 1):
                try:
                    data_arr = np.fromfile(f, dtype=np.int16, count=ndata)
                except Exception:
                    raise IOError("Error reading from file: " + fn)
                if (scale_to_volts):
                    data_arr = ((2 ** config['APDCAM_bits'] - 1) - data_arr) \
                                / (2. ** config['APDCAM_bits'] - 1) * 2
                else:
                    data_arr = (2 ** config['APDCAM_bits'] - 1) - data_arr
                if (offset_timerange is not None):
                        data_arr -= offset_data[i]
            else:
                try:
                    d = np.fromfile(f, dtype=np.int16, count=ndata)
                except Exception:
                    raise IOError("Error reading from file: " + fn)
                if (scale_to_volts):
                    d = ((2 ** config['APDCAM_bits'] - 1) - d) \
                                / (2. ** config['APDCAM_bits'] - 1) * 2
                else:
                    d = (2 ** config['APDCAM_bits'] - 1) - d
                if (offset_timerange is not None):
                        d -= offset_data[i]
                data_arr[:,i] = d
        f.close

        try:
            data_arr = calibrate(data_arr, signal_proc, read_range, exp_id=exp_id, options=_options)
        except Exception as e:
            raise e
        data_dim = data_arr.ndim    
    else:
        if (len(ADC_proc) is not 1):
            data_dim = 2
        else:
            data_dim = 1

    coord = [None]*data_dim*2
    c_mode = flap.CoordinateMode(equidistant=True)
    coord[0] = copy.deepcopy(flap.Coordinate(name='Time',
                                             unit='Second',
                                             mode=c_mode,
                                             start=read_range[0],
                                             step=config['APDCAM_sampletime'],
                                             dimension_list=[0])
                             )
    coord[1] = copy.deepcopy(flap.Coordinate(name='Sample',
                                             unit='n.a.',
                                             mode=c_mode,
                                             start=read_samplerange[0],
                                             step=1,
                                             dimension_list=[0])
                             )
    if (data_dim > 1):
        ch_proc = []
        for ch in signal_proc:
            if (ch[0:4] != 'ABES'):
                ch_proc = []
                break
            ch_proc.append(int(ch[5:]))
        if (ch_proc != []):
            c_mode = flap.CoordinateMode(equidistant=False)
            coord[2] = copy.deepcopy(flap.Coordinate(name='Channel',
                                                     unit='n.a.',
                                                     mode=c_mode,
                                                     shape=len(ch_proc),
                                                     values=ch_proc,
                                                     dimension_list=[1])
                                 )
        coord[3] = copy.deepcopy(flap.Coordinate(name='Signal name',
                                                 unit='n.a.',
                                                 mode=c_mode,
                                                 shape=len(signal_proc),
                                                 values=signal_proc,
                                                 dimension_list=[1])
                                 )

    data_title = "W7-X ABES data"
    if (data_arr.ndim == 1):
        data_title += " (" + signal_proc[0] + ")"
    d = flap.DataObject(data_array=data_arr,
                        data_unit=data_unit,
                        coordinates=coord,
                        exp_id=exp_id,
                        data_title=data_title,
                        info={'Options':_options},
                        data_source="W7X_ABES")
    return d


def add_coordinate():
    raise NotImplementedError("Adding coordinates is not JET implemented")

def register(data_source=None):
    flap.register_data_source('JET_LIBES', get_data_func=jet_libes_get_data, add_coord_func=add_coordinate)

if __name__ == "__main__":
    conf = LiBESConfig(95939)
    conf.get_config(config='KY6-AccVoltage')
    conf.get_config(config='KY6-EmitterVoltage')
#    conf.get_config(config="KY6-CrossCalib", options={"UID": "KY6-team"})
    conf.read_apd_map()
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