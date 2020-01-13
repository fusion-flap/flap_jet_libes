import flap
import flap_jet_api as dataapi
import flap_jet_libes as jet_libes
from matplotlib import pyplot as plt
import unittest
import numpy as np


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

def test_register():
    jet_libes.register()
    return 'JET_LIBES' in flap.list_data_sources()


def test_read_files():
    pulse_id = 96371
    conf = jet_libes.LiBESConfig(pulse_id)
    conf.read_apd_map()
    conf.get_fibre_config()
    return True

def test_get_data():
    pulse_id = 96371
    jet_libes.jet_libes_get_data(exp_id=pulse_id, data_name="KY6-*",
                                 options={'Mode': 'Down', 'Amplitude calibration': "Spectrometer PPF",
                                          'Spatial calibration': "Spectrometer PPF"})
    return True

def test_remove_background():
    pulse_id = 96371
    all_chan_data=jet_libes.jet_libes_get_data(exp_id=pulse_id, data_name="KY6-*",
                                options={'Mode': 'Down', 'Amplitude calibration': "Spectrometer PPF",
                                         'Spatial calibration': "Spectrometer PPF"})
    all_chan_data.time_window = [49.0,49.1]
    beam_on = remove_background(all_chan_data, options={'Average Chopping Period': False})
    averaged = beam_on.slice_data(summing={'Time':'Mean'})
    averaged.plot(axes=["Device Z"])
    return True

class LiBESTest(unittest.TestCase):
    def test_register(self):
        self.assertTrue(test_register())
    def test_read_files(self):
        self.assertTrue(test_read_files())
    def test_get_data(self):
        self.assertTrue(test_get_data())
    def test_remove_background(self):
        self.assertTrue(test_remove_background())

if __name__ == '__main__':
    unittest.main()