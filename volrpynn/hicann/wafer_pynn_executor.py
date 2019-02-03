
from collections import defaultdict
import json
import sys
import argparse
import os
import numpy as np

init_logger("WARN", [
    ("guidebook", "DEBUG"),
    ("marocco", "INFO"),
    ("Calibtic", "INFO"),
    ("sthal", "INFO")
])

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

def create_wafer_edge(nodes, edge, marocco):
    projection_type = edge["projection_type"]["kind"]
    if ("type" in nodes[edge["output"]["id"]] and nodes[edge["output"]["id"]]["type"] == "output"):
        print("Not wiring output")
    elif  (projection_type == "all_to_all"):
        # only support static connectivity for now
        assert(edge['projection_target']['kind'] == 'static')
        connector = pynn.AllToAllConnector(weights=edge["projection_type"]["weight"])
        target_effect = edge['projection_target']['effect']
        pynn.Projection(nodes[edge["input"]["id"]],
                        nodes[edge["output"]["id"]],
                        connector,
                        target=str(target_effect))
    else:
        print "not yet supported"

def spikes_to_json(spikes):
    spiking_neurons = defaultdict(list)
    for spike in spikes:
        spiking_neurons[int(spike[0])].append(spike[1])

    return spiking_neurons.values()

def execute(model, hicann_number=297):
    # Create and setup runtime
    execution_target = model["execution_target"]
    marocco = instrument_marocco(execution_target)
    runtime = Runtime(marocco.default_wafer)

    hicann = C.HICANNOnWafer(C.Enum(hicann_number))

    pynn.setup(marocco=marocco, marocco_runtime=runtime)

    # Build network
    net = model["network"]
    block = net["blocks"][0] # only support one block
    nodes = {}
    outputs = {}
    stimulus = ""

    # Create nodes
    for node in block["nodes"]:
        n = create_wafer_node(node, marocco)
        nodes[node["id"]] = n
        if node["type"] == "population" and node["record_spikes"]:
            outputs[node["label"]] = nodes[node["id"]]
        elif node["type"] == "spike_source_array":
            stimulus = n

        if node["type"] == "population" or node["type"] == "spike_source_array":
            marocco.manual_placement.on_hicann(n, hicann)

    for proj in block["edges"]:
        create_wafer_edge(nodes, proj, marocco)

    # Setup recording
    for label in outputs:
        outputs[label].record()

    # Run mapping
    marocco.skip_mapping = False
    marocco.backend = PyMarocco.None
    pynn.reset()
    pynn.run(model["simulation_time"])

    # Set hardware wafer parameters
    set_sthal_params(runtime.wafer(), gmax=1023, gmax_div=1)

    marocco.skip_mapping = True
    marocco.backend = PyMarocco.Hardware
    # Full configuration during first step
    marocco.hicann_configurator = PyMarocco.ParallelHICANNv4Configurator

    pynn.run(model["simulation_time"])

    # Extract spikes
    for label in outputs:
        outputs[label] = spikes_to_json(outputs[label].getSpikes())
    pynn.reset()

    return json.dumps(outputs, separators=(',', ':'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='wafer pynn executor')
    args = parser.parse_args()
    conf = json.load(sys.stdin)
    result = execute(conf)
    print(result)
