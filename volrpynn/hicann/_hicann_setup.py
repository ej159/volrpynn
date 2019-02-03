from pyhalbe import HICANN
import pyhalbe.Coordinate as C
from pysthal.command_line_util import init_logger

import pyhmf as pynn
from pymarocco import PyMarocco, Defects
from pymarocco.runtime import Runtime
from pymarocco.coordinates import LogicalNeuron
from pymarocco.results import Marocco

init_logger("WARN", [
    ("guidebook", "DEBUG"),
    ("marocco", "INFO"),
    ("Calibtic", "INFO"),
    ("sthal", "INFO")
])

def setup_marocco(wafer=37):
    """Setup hardwarm mapping on a wafer. Defaults to 37"""
    marocco = PyMarocco()
    marocco.neuron_placement.default_neuron_size(4)
    marocco.neuron_placement.minimize_number_of_sending_repeaters(False)
    marocco.merger_routing.strategy(marocco.merger_routing.one_to_one)

    marocco.bkg_gen_isi = 125
    marocco.pll_freq = 125e6

    marocco.backend = PyMarocco.Hardware
    marocco.calib_backend = PyMarocco.XML
    marocco.defects.path = marocco.calib_path = "/wang/data/calibration/brainscales/default-2017-09-26-1"
    marocco.defects.backend = Defects.XML
    marocco.default_wafer = C.Wafer(int(os.environ.get("WAFER", wafer)))
    marocco.param_trafo.use_big_capacitors = True
    marocco.input_placement.consider_firing_rate(True)
    marocco.input_placement.bandwidth_utilization(0.8)
    return marocco

def set_sthal_params(wafer, gmax, gmax_div):
    """
    synaptic strength:
    gmax: 0 - 1023, strongest: 1023
    gmax_div: 1 - 15, strongest: 1
    """

    # for all HICANNs in use
    for hicann in wafer.getAllocatedHicannCoordinates():

        fgs = wafer[hicann].floating_gates

        # set parameters influencing the synaptic strength
        for block in C.iter_all(C.FGBlockOnHICANN):
            fgs.setShared(block, HICANN.shared_parameter.V_gmax0, gmax)
            fgs.setShared(block, HICANN.shared_parameter.V_gmax1, gmax)
            fgs.setShared(block, HICANN.shared_parameter.V_gmax2, gmax)
            fgs.setShared(block, HICANN.shared_parameter.V_gmax3, gmax)

        for driver in C.iter_all(C.SynapseDriverOnHICANN):
            for row in C.iter_all(C.RowOnSynapseDriver):
                wafer[hicann].synapses[driver][row].set_gmax_div(
                    C.left, gmax_div)
                wafer[hicann].synapses[driver][row].set_gmax_div(
                    C.right, gmax_div)

        # don't change values below
        for ii in xrange(fgs.getNoProgrammingPasses()):
            cfg = fgs.getFGConfig(C.Enum(ii))
            cfg.fg_biasn = 0
            cfg.fg_bias = 0
            fgs.setFGConfig(C.Enum(ii), cfg)

        for block in C.iter_all(C.FGBlockOnHICANN):
            fgs.setShared(block, HICANN.shared_parameter.V_dllres, 275)
            fgs.setShared(block, HICANN.shared_parameter.V_ccas, 800)
