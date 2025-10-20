import numpy as np
import matplotlib.pyplot as plt

class GaNDevice:
    def __init__(self, model='EPC2305', T_j=25):
        self.model = model
        if T_j is None:
            self.T_j = 25  # Default junction temperature in Celsius
        else:   
            self.T_j = T_j  # Initial junction temperature in Celsius
        if model == 'EPC2305':
            self.Rds_on_25C = 0.0022  # Ohm
            self.Vgs_th = 1.2  # V
            self.V_drive = 5  # V
            self.R_gate_total_on = 0.5 + 2.2 + 2  # Ohm, R_gate + R_driver + Rg_on
            self.R_gate_total_off = 0.5 + 1 + 1  # Ohm, R_gate + R_driver + Rg_on
            self.Qgate = 22e-9  # Coulombs
        else:
            raise ValueError("Unknown GaN model")
        
    def gfs(self, type='simple'):  # Transconductance
        if self.model == 'EPC2305':
            if type == 'simple':
                return -0.5941*self.T_j + 155.66  # Sievert
            elif type == 'advanced':
                return (self.V_plateau(Id=30)-self.V_plateau(Id=30.1)) / (0.1)  # Sievert
            else:
                raise ValueError("Unknown gfs type")
        else:
            raise ValueError("Unknown GaN model")
        
    def Coss(self, V_ds):  # Polynomial fit from datasheet (R^2=0.9908)
        if self.model == 'EPC2305':
            return 1e-12*(-4e-7*V_ds**5 + 2e-4*V_ds**4 - 0.0354*V_ds**3 + 3.0353*V_ds**2 - 129.84*V_ds + 3192.6)
        else:
            raise ValueError("Unknown GaN model")
        
    def Qoss_energy_eq(self, V_ds):  # Polynomial fit from datasheet (R^2=0.9999)
        if self.model == 'EPC2305':
            return 1e-12*(9e-6*V_ds**5 - 4e-3*V_ds**4 + 0.6591*V_ds**3 - 53.214*V_ds**2 + 3033.5*V_ds + 518.03)
        else:
            raise ValueError("Unknown GaN model")

    def Rds_on(self):  # Polynomial fit from datasheet (R^2=0.9997)
        if self.model == 'EPC2305':
            return self.Rds_on_25C*(1e-5*self.T_j**2 + 0.0055*self.T_j + 0.8535)  # Ohm
        else:
            raise ValueError("Unknown GaN model")

    def Cgd(self, V_ds):  # Polynomial fit from datasheet (R^2=0.9923)
        if self.model == 'EPC2305':
            return 1e-12*(-5e-10*V_ds**5 + 2e-7*V_ds**4 - 3e-5*V_ds**3 + 0.0025*V_ds**2 - 0.0945*V_ds + 2.5196)  # nF
        else:
            raise ValueError("Unknown GaN model")
        
    def Qgd_energy_eq(self, V_ds):  # Polynomial fit from datasheet (R^2=0.9951)
        if self.model == 'EPC2305':
            return 1e-12*(-2e-8*V_ds**6 + 8e-6*V_ds**5 - 0.0015*V_ds**4 + 0.1456*V_ds**3 - 7.239*V_ds**2 + 185.06*V_ds + 101.67)
        else:
            raise ValueError("Unknown GaN model")
        
    def Vsd(self, I_sd):  # Linear interpolation from 2 datasheet points
        if self.model == 'EPC2305':
            return 1.862738 + (0.007263 - 5.01e-5 * (self.T_j - 25)) * max(0, I_sd)  # V
        else:
            raise ValueError("Unknown GaN model")
        
    def Qgs2(self, Id=30):  # Linear interpolation from 2 datasheet points
        if self.model == 'EPC2305':
            Qgs = 6.6e-9 # Coulombs
            Qgs_th = 4.6e-9 # Coulombs
            Qgs2_spec = Qgs - Qgs_th
            Id_spec = 30 # Amps

            Qgs2 = Qgs2_spec *(Id/Id_spec)
            return Qgs2
        else:
            raise ValueError("Unknown GaN model")

    def V_plateau(self, Id=30):
        if self.model == 'EPC2305':  # Fitted with smooth MOSFET model. Fitted data to a softplus + power law
            k_25 = 161.369
            Vth_25 = 1.8588
            n_25 = 1.046
            s_25 = 10.880
            k_125 = 80.8
            Vth_125 = 1.7269
            n_125 = 1.12
            s_125 = 12.012
            V_plateau_25 = Vth_25 + (1 / s_25) * np.log(np.exp(s_25 * (Id / k_25) ** (1 / n_25)) - 1)
            V_plateau_125 = Vth_125 + (1 / s_125) * np.log(np.exp(s_125 * (Id / k_125) ** (1 / n_125)) - 1)
            return V_plateau_25 + (V_plateau_125 - V_plateau_25) / (125 - 25) * (self.T_j - 25)
        else:
            raise ValueError("Unknown GaN model")

    def calculate_switching_losses(self, Isw_on, Isw_off, V_sw, f_sw, t_deadtime, analysis_type='simple'):
        """Calculates the total switching losses of the GaN device.
        I_rms: RMS current through the device (A)"""
        # Turn-on energy losses per switching event (J)
        I_G_cr = (self.V_drive-(self.Vgs_th + self.V_plateau())/2) / self.R_gate_total_on  # I_G during plateau
        t_cr = self.Qgs2(Id=Isw_on) / I_G_cr
        if analysis_type == 'simple':
            I_G_vf = (self.V_drive - self.V_plateau()) / self.R_gate_total_on  # I_G during voltage fall
            t_vf = self.Qgd_energy_eq(V_sw) / I_G_vf
        else:
            delta_Vgs_vf = (self.V_drive - self.V_plateau())/(0.5+(self.R_gate_total_on*self.gfs()*self.Cgd(0))/(self.Coss(V_sw)+self.Coss(0)))
            t_vf = 2*(self.Qoss_energy_eq(V_sw)+self.Qoss_energy_eq(0))/(self.gfs()*delta_Vgs_vf)
        
        E_on = 0.5 * V_sw * abs(Isw_on) * (t_cr + t_vf)
        P_on = E_on * f_sw  # Turn-on power loss (W)

        # Turn-off energy losses per switching event (J)
        
        I_G_cf = (((self.V_plateau() + self.Vgs_th)/2) - self.V_drive) / self.R_gate_total_off  # I_G during current fall
        t_cf = self.Qgs2(Id=Isw_off) / abs(I_G_cf)
        t_vr = (self.Qoss_energy_eq(0) + self.Qoss_energy_eq(V_sw)) / Isw_off - t_cf/2
        delta_vds_cf = 1/2 * t_cf * Isw_off / (self.Coss(V_sw) + self.Coss(0))
        E_off = 1/6 * t_cf * delta_vds_cf * abs(Isw_off)
        P_off = E_off * f_sw  # Turn-off power loss (W)

        # Gate drive loss:
        E_gate = self.Qgate * self.V_drive  # Energy lost in driving the gate (J)
        Pgate = E_gate * f_sw * 1/3  # Gate drive power loss (W), assuming 1/3 power dissipated in GaN device (Rdrv=Rg_on=Rgate)

        # Reverse conduction loss:
        t_on_SR = self.Qgs2(Id=Isw_on) * self.R_gate_total_on / (self.V_drive - (self.Vgs_th-0)/2)  # Time in reverse conduction during turn-on
        t_off_SR = 2 * self.Qgs2(Id=Isw_off) * self.R_gate_total_off / (self.Vgs_th - 0)  # Time in reverse conduction during turn-off
        t_sd_off = max(0, t_deadtime - t_vf - t_cr/2 - t_off_SR/2)
        t_sd_on = max(0, t_deadtime - t_cf - t_vr/2 - t_on_SR/2)
        # print(f"t_deadtime: {1e12*t_deadtime:.3f} ns")
        # print(f"t_cf: {1e12*t_cf:.3f} ns")
        # print(f"t_vr: {1e12*t_vr:.3f} ns")
        # print(f"t_on_SR: {1e12*t_on_SR:.3f} ns")
        # print(f"t_deadtime - t_cf - t_vr/2 - t_on_SR/2: {1e12*(t_deadtime - t_cf - t_vr/2 - t_on_SR/2):.3f} ns")
        E_sd = Isw_on * self.Vsd(Isw_on) * t_sd_on + Isw_off * self.Vsd(Isw_off) * t_sd_off

        # Simple implementation of E_sd, due to previous equations returning negative times sometimes
        E_sd = (Isw_on * self.Vsd(Isw_on) + Isw_off * self.Vsd(Isw_off)) * (t_deadtime * 0.75)
        P_sd = E_sd * f_sw  # Reverse conduction power loss (W)

        # Coss related loss:
        E_coss = self.Qoss_energy_eq(V_sw)  # Energy lost in output capacitance (J)
        P_coss = E_coss * f_sw  # Coss related power loss (W)

        P_sw_tot = max(0, P_on) + max(0, P_off) + max(0, Pgate) + max(0, P_sd) + max(0, P_coss)  # Total power loss (W)
        return P_sw_tot, P_on, P_off, Pgate, P_sd, P_coss
    
    def calculate_conduction_losses(self, I_rms):
        """Calculates the conduction losses of the GaN device.
        I_rms: RMS current through the device (A)"""
        # Conduction related loss:
        P_cond = I_rms**2 * self.Rds_on()  # Conduction loss (W)
        return P_cond


if __name__ == "__main__":

    TheDevice = GaNDevice(model='EPC2305', T_j=80)
    P_tot, P_on, P_off, Pgate, P_sd, P_coss = TheDevice.calculate_switching_losses(Isw_on=10,
                                                                                   Isw_off=10,
                                                                                   V_sw=100,
                                                                                   f_sw=100e3,
                                                                                   t_deadtime=5e-9)
    P_cond = TheDevice.calculate_conduction_losses(I_rms=80.0/6)
    P_tot += P_cond
    print(f"Total loss: {P_tot*1e3:.3f} mW, Turn-on loss: {P_on*1e3:.3f} mW, Turn-off loss: {P_off*1e3:.3f} mW, Gate loss: {Pgate*1e3:.3f} mW, Reverse conduction loss: {P_sd*1e3:.3f} mW, Conduction loss: {P_cond*1e3:.3f} mW, Coss loss: {P_coss*1e3:.3f} mW")
    
    # Plotting test
    gan = GaNDevice()

    # Define current range (e.g., 0 to 60A)
    currents = np.linspace(0, 60, 100)

    # Prepare lists for each loss component
    switch_on_losses = []
    switch_off_losses = []
    coss_losses = []
    isd_losses = []
    Pgate_losses = []
    conduction_losses = []

    for Isw in currents:
        # Calculate switching losses (returns tuple: (switch_on, switch_off, coss, driver, ...))
        _, P_on, P_off, Pgate, P_sd, P_coss = gan.calculate_switching_losses(Isw, Isw, 100, 100e3, 5e-9, analysis_type='simple')
        switch_on_losses.append(P_on)
        switch_off_losses.append(P_off)
        coss_losses.append(P_coss)
        isd_losses.append(P_sd)
        Pgate_losses.append(Pgate)
        # Calculate conduction losses (returns a single value)
        conduction = gan.calculate_conduction_losses(Isw)
        conduction_losses.append(conduction)

    # Stack the losses for area plotting
    loss_arrays = np.array([
        switch_on_losses,
        switch_off_losses,
        coss_losses,
        Pgate_losses,
        conduction_losses
    ])

    labels = [
        'Switch On Loss',
        'Switch Off Loss',
        'Coss Loss',
        'Driver Loss',
        'Conduction Loss'
    ]

    colors = ["#1b7cc2", '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    plt.figure(figsize=(10, 6))
    plt.stackplot(currents, loss_arrays, labels=labels, colors=colors, alpha=0.8)
    plt.xlabel('Switching Current (A)')
    plt.ylabel('Loss (W)')
    plt.title('GaN Device Losses vs Switching Current (Stacked)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
