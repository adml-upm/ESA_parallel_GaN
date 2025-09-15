from PyLTSpice import SimRunner
from PyLTSpice import AscEditor
import os

# Force another simulatior
# simulator = r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe"

# select spice model
LTC = SimRunner(output_folder='temp', simulator=None)

sim_path = './02 Simulations/01 Examples/Batch_Test.asc'
netlist = AscEditor(sim_path)

# set default arguments
netlist.set_parameters(res=0, cap=100e-6)
netlist.set_component_value('R2', '2k')  # Modifying the value of a resistor
netlist.set_component_value('R1', '4k')
netlist.set_component_value('V3', "SINE(0 1 3k 0 0 0)")

netlist.add_instructions(
    "; Simulation settings",
    ";.param run = 0"
)
netlist.set_parameter('run', 0)

for opamp in ('AD712', 'AD820_new'):
    netlist.set_element_model('U1', opamp)
    for supply_voltage in (5, 10, 15):
        netlist.set_component_value('V1', supply_voltage)
        netlist.set_component_value('V2', -supply_voltage)
        print("simulating OpAmp", opamp, "Voltage", supply_voltage)
        LTC.run(netlist)

for raw, log in LTC:
    print("Raw file: %s, Log file: %s" % (raw, log))
    # do something with the data
    # raw_data = RawRead(raw)
    # log_data = LTSteps(log)
    # ...

# Sim Statistics
print('Successful/Total Simulations: ' + str(LTC.okSim) + '/' + str(LTC.runno))

enter = input("Press enter to delete created files")
if enter == '':
    LTC.file_cleanup()
