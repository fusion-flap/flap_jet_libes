import flap
import flap_jet_api as dataapi
import flap_jet_libes as jet_libes
from matplotlib import pyplot as plt
import numpy as np

def plot_time(tw, all_chan_data):
    from scipy import signal as sci_sign
    b, a = sci_sign.butter(3, 0.01)
    off_time = all_chan_data.slice_data(slicing={"Time":flap.Intervals(tw[0], tw[1])})
    off_time = all_chan_data
    start_at = 0
    average_val = np.mean(off_time.data,axis=0, keepdims=True)
    steps = np.mean(np.mean(np.abs(off_time.data-average_val),axis=0),axis=0)*15
    for channel_name in off_axis_channels:
        channel = off_time.slice_data(slicing={"Pixel Number":channel_name})
        y = channel.data
        y = sci_sign.filtfilt(b, a, channel.data)
        if channel_name in append_on_channel:
            plot_label = channel_name+" (on axis)"
        else:
            plot_label = channel_name+" (off axis)"
        plt.plot(channel.get_coordinate_object("Time").values, y+start_at,\
                 label=plot_label)
        start_at = start_at-steps
    plt.title(str(all_chan_data.exp_id))
    plt.show()
    plt.legend(loc="upper right")

def get_fft(dataobject):
    signal = dataobject.data[1::2]
    fourier = np.fft.rfft(signal)
    time = dataobject.get_coordinate_object("Time").values[1::2]
    timestep = np.mean(time[1:]-time[:-1])
    n = signal.size+2
    freq = np.fft.fftfreq(n, d=timestep)
    freq = freq[np.where(freq>=0)]
    return fourier, freq

def check_chopper(pulse_id, channel, start_delay=0, end_delay=0, time_window = None):
    if time_window is not None:
        data = flap.get_data('JET_LIBES',
                            name=[str(channel)],
                            exp_id=pulse_id,
                            coordinates={'Time':time_window},
                            object_name=str(channel)+' DATA',
                            options={'Mode': 'Fast',
                                             'Amplitude calibration': None,
                                             'Spatial calibration': None})
    else:
        data = flap.get_data('JET_LIBES',
                             name=str(channel),
                             exp_id=str(pulse_id),
                             object_name=str(channel)+' DATA',
                             options={'Mode': 'Fast',
                                              'Amplitude calibration': None,
                                              'Spatial calibration': None})
    if time_window is not None:
        d_beam_on=flap.get_data('JET_LIBES',exp_id=data.exp_id, name='Chopper_time',
                             options={'State':{'Chop': 0}, 'Start delay': start_delay, 'End delay': end_delay}, object_name='Beam_on', coordinates={'Time':time_window})
        d_beam_off=flap.get_data('JET_LIBES',exp_id=data.exp_id, name='Chopper_time',
                             options={'State':{'Chop': 1}, 'Start delay': start_delay, 'End delay': end_delay}, object_name='Beam_off', coordinates={'Time':time_window})
    else:
        d_beam_on=flap.get_data('JET_LIBES',exp_id=data.exp_id,name='Chopper_time',
                             options={'State':{'Chop': 0}, 'Start delay': start_delay, 'End delay': end_delay}, object_name='Beam_on')
        d_beam_off=flap.get_data('JET_LIBES',exp_id=data.exp_id,name='Chopper_time',
                             options={'State':{'Chop': 1}, 'Start delay': start_delay, 'End delay': end_delay}, object_name='Beam_off')

    plt.plot(data.coordinate('Time')[0], data.data)
    plt.scatter(d_beam_off.get_coordinate_object("Time").values, np.ones(d_beam_off.get_coordinate_object("Time").values.shape)*np.min(data.data))
    d_beam_off.plot(axes=['Time', min(data.data)], plot_type='scatter')
#    d_beam_on.plot(axes=['Time', min(data.data)], plot_type='scatter')
    plt.show()
    del data
    del d_beam_off
    del d_beam_on

def remove_background(data, options={}):
    options_default={'Average Chopping Period': False}
    options = {**options_default, **options}
    if data.time_window is not None:
        beam_on = jet_libes.proc_chopsignals(dataobject=data,
                            timerange=data.time_window, options=options, test=False)
    else:
        beam_on = jet_libes.proc_chopsignals(dataobject=data,
                            options=options, test=False)
    del data
    return beam_on

if __name__ == "__main__":
    
    jet_libes.register()

    pulse_id = 96371
#    pulse_id = 96040
    conf = jet_libes.LiBESConfig(pulse_id)

    conf.read_apd_map()
    conf.get_fibre_config()
    
    #getting the off_axis_channels
#    on_axis_channels = []
    append_on_channel = ["BES-1-1", "BES-1-8",  "BES-4-2"]
    off_axis_channels = []
    for channel in conf.fibre_conf:
        if np.abs(channel[6])>0.1:
            off_axis_channels.append(channel[0])
    all_channels = off_axis_channels+append_on_channel
#    off_axis_channels = off_axis_channels[0]
#     Getting of axis channel data
#    all_chan_data=jet_libes.jet_libes_get_data(exp_id=pulse_id, data_name=all_channels,
#                                options={'Mode': 'Down', 'Amplitude calibration': "Spectrometer PPF", 'Spatial calibration': "Spectrometer PPF"})
#    
#    
#    # plotting the relevant time window
#    tw = [48,52]
#    plot_time(tw, all_chan_data)


#    # Calculating and plotting the fourier transform
#    plt.figure()
#    signal_name = "BES-4-2"
#    tw= [49.9, 50.0]
#    channel = all_chan_data.slice_data(slicing={"Pixel Number":signal_name})
#    on_fast_mod = channel.slice_data(slicing={"Time":flap.Intervals(tw[0],tw[1])})
#    fourier, freq = get_fft(on_fast_mod)
#    plt.plot(freq,np.abs(fourier), label = "fast modulation on "+str(tw)+"s")
#    tw= [50.0, 50.1]
#    off_fast_mod = channel.slice_data(slicing={"Time":flap.Intervals(tw[0],tw[1])})
#    fourier, freq = get_fft(off_fast_mod)
#    plt.plot(freq,np.abs(fourier), label = "fast modulation off "+str(tw)+"s")
#    plt.title(signal_name+" at "+str(pulse_id))
#    plt.xlim(10,1600)
#    plt.xlabel("Frequency (Hz)")
#    plt.ylim(0, np.max(fourier[np.logical_and(freq>=10,freq<=800)]))
#    plt.legend(loc="upper right")
#    plt.show()
    
    # comparing background corrected off and on axis channels
    all_chan_data=jet_libes.jet_libes_get_data(exp_id=pulse_id, data_name="KY6-*",
                                options={'Mode': 'Down', 'Amplitude calibration': "Spectrometer PPF", 'Spatial calibration': "Spectrometer PPF"})
    all_chan_data.time_window = None
    beam_on = remove_background(all_chan_data, options={'Average Chopping Period': True})
    temp = beam_on.slice_data(slicing={"Time":49.05})
    temp.data =np.nanmean(beam_on.data, axis=0)
    temp.plot(axes=["Device Z"], plot_type="scatter")
#    pairs = {"BES-3-4": "BES-1-1",
#             "BES-1-4": "BES-1-1",
#             "BES-2-5": "BES-1-8",
#             "BES-1-5": "BES-1-8",
#             "BES-3-5": "BES-4-2"}
#
#    index=1
#    plt.title(str(pulse_id))
#    for off_chan in off_axis_channels:
#        all_chan_data.slice_data(slicing={"Pixel Number": off_chan})
#        off_chan_data = beam_on.slice_data(slicing={"Pixel Number": off_chan})
#        on_chan_data = beam_on.slice_data(slicing={"Pixel Number": pairs[off_chan]})
#        res= off_chan_data.data/on_chan_data.data
#        plt.subplot(510+index)
#        plt.scatter(off_chan_data.coordinate("Time")[0], res,
#                 label = off_chan+" (off axis)/"+pairs[off_chan]+" (on axis)")
#        plt.legend(loc="upper right")
#        lim = np.nanmean(res[np.logical_and(res>0, abs(res)<10000)])
#        plt.ylim((-0.1*lim,lim*3))
#        plt.show()
#        index=index+1