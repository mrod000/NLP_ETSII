# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:50:29 2021

@author: mrod
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,ode,complex_ode
import math

inletT_bol=False
pidbol=False

#Corriente de entrada
F =54.0 # Flowrate l/hr
T0=298.13 # Inlet temperature feed (K)
ca0=2.1641 # Concentration A inlet (mol/l)
#Volumenes de los reactores
V1=15.0  # Volume vessel l
V2=40.0  # Volume vessel 2
V3=40.0  # Volume vessel 3
#Cinetica de la reaccion
Eac= 76480.0 # Activation energy Arrhenius equation (J/mol)
k0=1.25*10**14 # Reaction constant (l/mol/hr)
H=-500000 # Reaction enthalpy (J/mol)
R=8.3144 # Gas constant (J/mol/K)
#Calores de las camisas de refrigeracion
Q1=-28146100.0 # Cooling duty vessel 1 (J/hr)
Q2=-11187000.0 # Cooling duty vessel 2 (J/hr)
Q3=-5593500.0  # Cooling duty vessel 3 (J/hr)
#Calor específico
dens_cp=30000.0 # Specific heat capacity per liter  (J/l/K)

class PID:

       def __init__(self, P=0.0, I=1.0, D=0.0, Derivator=0, Integrator=0, Windup_max=500, Windup_min=-500,Bias=0,Action=1,Resint=0):

             self.Kp=P #Proportional gain
             self.Ki=I # Integral Time
             self.Kd=D # Derivative Time
             self.Derivator=Derivator
             self.Integrator=Integrator
             self.Windup_max=Windup_max
             self.Windup_min=Windup_min
             self.Bias=Bias
             self.Action=Action
             self.Resint=Resint
             self.set_point=0.0
             self.error=0.0

       def update(self,current_value):

          self.error = self.set_point - current_value
          self.P_value = self.Kp * self.error*self.Action
          self.D_value = self.Kd * self.Derivator*self.Action*self.Kp		
          self.I_value = self.Integrator * self.Kp/self.Ki*self.Action
          PID = self.P_value + self.I_value + self.D_value +self.Bias
          
          if PID > self.Windup_max:
             self.Ki=10000
             PID=self.Windup_max
             print("wind up ",PID)
          elif PID < self.Windup_min:
             self.Ki=10000
             PID=self.Windup_min
             print("wind up ",PID)
          else:
             self.Ki=self.Resint


          return PID
      
        
ca3_set=0.313 #Set point inicial
pidcontr_ca3=PID() #Creamos una instancia (objeto) de la clase PID     

def modelo(y, t):
     global T0,ca3_set,kd,ki,qnew,pidcontr_ca3,cas_contr_ca3  
     if not inletT_bol:
        if t>step:      
           pidcontr_ca3.set_point=0.4
        else:
           pidcontr_ca3.set_point=ca3_set
     else:
        if t>step:
           T0=TO_list[1]
        else:
           T0=TO_list[0]
     

     #Assignment of state variable values
     ca_1 = y[0]
     ca_2 = y[2]
     ca_3 = y[4]

     T_1 = y[1]
     T_2 = y[3]
     T_3 = y[5]

      
     Contr_error=y[6]
     Int_Contr_error=y[7] # integral action
     
     # Reaction kinetics   
     k1=k0*math.exp(-Eac/T_1/R)
     k2=k0*math.exp(-Eac/T_2/R)
     k3=k0*math.exp(-Eac/T_3/R)
 
     # Mass and energy balance to CSTR 3
     dT3_dt=(F*dens_cp*(T_2-T_3)-H*k3*ca_3*ca_3*V3+Q3)/dens_cp/V3
     dca3_dt=F*(ca_2-ca_3)/V3-k3*ca_3*ca_3
         
     #This indicates if the PID is used (pidbol = True) or not
     if pidbol:
        e=pidcontr_ca3.set_point-ca_3
        derr_dt=-dca3_dt
        intpid_err=e
        pidcontr_ca3.Integrator=Int_Contr_error
        pidcontr_ca3.Derivator=derr_dt
        qnew=pidcontr_ca3.update(ca_3)
     else:
        qnew=Q1
        e=ca3_set-ca_3
        derr_dt=-dca3_dt
        intpid_err=e
     
     # Mass and energy balance to CSTR 1 and 2    
     dT1_dt=(F*dens_cp*(T0-T_1)-H*k1*ca_1*ca_1*V1+qnew)/dens_cp/V1
     dca1_dt=F*(ca0-ca_1)/V1-k1*ca_1*ca_1
     dca2_dt=F*(ca_1-ca_2)/V2-k2*ca_2*ca_2
     dT2_dt=(F*dens_cp*(T_1-T_2)-H*k2*ca_2*ca_2*V2+Q2)/dens_cp/V2
     
 
 
 
     return [dca1_dt, dT1_dt, dca2_dt, dT2_dt, dca3_dt, dT3_dt,derr_dt,intpid_err]
 
#Flags indicando que simulamos
inletT_bol=True #salto de temperatura
pidbol=False #no hay controlador

#Tiempo para la perturbacion y su valor
step=3.0
TO_list=[298.13,289.1] #valor inicial y valor de la perturbacion

#initial conditions
T1=304.615 # Temperature vessel 1 (K)
T2=305.814 # Temperature vessel 2 (K)
T3=304.306 # Temperature vessel 3 (K)
ca1=0.7325 # Concentration A vessel 1 (mol/l)
ca2= 0.2464 # Concentration A vessel 2 (mol/l)
ca3=0.13  # Concentration A vessel 3 (mol/l)

ylist=[ca1,T1,ca2,T2,ca3,T3,0.0,0.0]
# Tiempo de simulacion
t  = np.linspace(0.0, 10, 1000) 	# time grid

# Resolución de las ecuaciones diferenciales

soln = odeint(modelo, ylist, t)
ca3_r = soln[:, 4]
T3=soln[:,5]

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.xlabel('Tiempo (horas)')
plt.ylabel('Concentracion Ca3 sin PID')
plt.plot(t,ca3_r)
plt.legend(['Concentracion A en CSTR3'],loc='best',frameon=False)

plt.subplot(2,1,2)
plt.xlabel('Tiempo (horas)')
plt.ylabel('Temperatura T3')
plt.plot(t,T3)
plt.legend(['Temperatura en CSTR3 sin PID'],loc='best',frameon=False)


# optimal pid control
inletT_bol=True
pidbol=True
pidcontr_ca3.set_point=ca3_set
pidcontr_ca3.Windup_max=-14*10**5
pidcontr_ca3.Windup_min=-154*10**6
pidcontr_ca3.Bias=Q1
pidcontr_ca3.Action=-1

if not inletT_bol:
    pidcontr_ca3.Kp=6.58523370e+07 #sintonizacion para False/true, PID  cambio de set point
    pidcontr_ca3.Ki=2.03234814e+00 #sintonizacion para False/true, PID cambio de set point
else:
    pidcontr_ca3.Kp=20.58523370e+07 #sintonizacion para True/true, PID perturbacion temperatura
    pidcontr_ca3.Ki=0.5603234814e+00 #sintonizacion para True/true, PID perturbacion temperatura
pidcontr_ca3.Kd=4.77114600e-01

#inicialización de las acciones integral y derivativa
pidcontr_ca3.Integrator=0.0
pidcontr_ca3.Derivator=0.0
pidcontr_ca3.Resint=pidcontr_ca3.Ki # not windup reset i action

#Tiempo de la pertubacion 
step=18.0


#initial conditions
T1=304.615 # Temperature vessel 1 (K)
T2=305.814 # Temperature vessel 2 (K)
T3=304.306 # Temperature vessel 3 (K)
ca1=0.7325 # Concentration A vessel 1 (mol/l)
ca2= 0.2464 # Concentration A vessel 2 (mol/l)
ca3=0.13  # Concentration A vessel 3 (mol/l)


ylist=[ca1,T1,ca2,T2,ca3,T3,0.0,0.0]
#Tiempo de simulacion
t  = np.linspace(0.0, 40, 1000) 	# time grid


# Resolucion de las ecuaciones diferenciales

soln = odeint(modelo, ylist, t)
ca3_r_2 = soln[:, 4]
T3_2 = soln[:, 5]

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.xlabel('Tiempo (horas)')
plt.ylabel('Concentracion Ca3')
plt.plot(t,ca3_r_2)
plt.legend(['Concentracion A en CSTR3 con PID'],loc='best',frameon=False)

plt.subplot(2,1,2)
plt.xlabel('Tiempo (horas)')
plt.ylabel('Temperatura T3')
plt.plot(t,T3_2)
plt.legend(['Temperatura en CSTR3 con PID'],loc='best',frameon=False)
plt.show()

  
        