# -*- coding: utf-8 -*-

'''
pathSim (c) University of Manchester 2019

pathSim is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  Pablo Carbonell
@description: Basic pathway simulation
'''

import tellurium as te
import numpy as np

def modelTemplate(nsteps):
    """ Nsteps basic linear pathway defined using tellurium """
    antinom = """
    // Created by libAntimony v2.9.4
        function Constant_flux__irreversible(v)
          v;
        end

        function Henri_Michaelis_Menten__irreversible(substrate, Km, V)
          V*substrate/(Km + substrate);
        end

        function Hill_Cooperativity(substrate, Shalve, V, h)
          V*(substrate/Shalve)^h/(1 + (substrate/Shalve)^h);
        end


        model New_Model()

          // Compartments and Species:
          compartment Cell;
          species Substrate in Cell, Product in Cell, Enzyme in Cell;
          species Inducer in Cell, Activated_promoter in Cell;

          // Reactions:
          Induction: Inducer => Activated_promoter; Copy_number*Cell*Hill_Cooperativity(Inducer, Induction_Shalve, Induction_Vi, Induction_h);
          Expression: Activated_promoter => Enzyme; Cell*Expression_k1*Activated_promoter;
          Leakage:  => Enzyme; Cell*Constant_flux__irreversible(Leakage_vl);
          Degradation: Enzyme => ; Cell*Degradation_k2*Enzyme;
          Catalysis: Substrate => Product; Cell*Henri_Michaelis_Menten__irreversible(Substrate, Catalysis_Km, Catalysis_V);

         // Species initializations:
          Substrate = 0.5;
          Product = 0;
          Enzyme = 0;
          Inducer = 1;
          Activated_promoter = 0;
          Copy_number = 5;

          // Compartment initializations:
          Cell = 1;

          // Variable initializations:
          Induction_Shalve = 0.01;
          Induction_Vi = 10000;
          Induction_h = 4;
          Expression_k1 = 100;
          Leakage_vl = 0.0001;
          Degradation_k2 = 10;
          Catalysis_Km = 0.1;
          Catalysis_V = 0.1;

          // Other declarations:
          const Cell;

        end
        
        """
    antinom += "model *Big_Model()"+"\n"
    for i in np.arange(nsteps):
            antinom += "\t"+"m%d: New_Model();" % (i+1,)
            antinom += "\n"
    for i in np.arange(nsteps-1):
            antinom += "\t"+"m%d.Product is m%d.Substrate;" % (i+1, i+2)
            antinom += "\n"
    antinom += "end\n"
    return te.loada(antinom)

def ranges():
    """ Define global parameter ranges """
    param = {
        'Catalysis': {
                'Km': [0.001, 1e2],
                'V': [0.01, 10] 
                },
        'Degradation': {
                'k2': [10, 1e3]
                },
        'Expression': {
                'k1': [10, 1e3]
                },       
        'Induction': {
                'Shalve': [0.0001, 1],
                'Vi': [1e3, 1e5],
                'h': [2, 6]
                },
        'Leakage': {
                'vl': [1e-6, 1e-4]
                }
        }
    return param
    
def instance():
    """ Generate an instance mean, std """
    par = ranges()
    vals = {}
    for group in par:
        vals[group] = {}
        for x in par[group]:
            xmax = par[group][x][1]
            xmin = par[group][x][0]
            mean = np.random.uniform( xmin, xmax )
            std = mean/100.0 + np.random.rand()*(xmax-xmin)/100.0
            vals[group][x] = ( mean,std )
    return vals
            
def initModel(model, nr=5):
    """ Each step in the pathway requires the following parameter definitions:
            - Induction: Shalve, Vi, h
            - Expression: k1
            - Degradation: k2
            - Leakage: vl
            - Catalysis: Km, V
            - Initial substrate concentration
            - Inducer concentrations
    """
    # Each step consists of nr reactions
    nsteps = int( model.getNumReactions()/nr )
    # Init all species to 0
    for step in np.arange(0,nsteps):
        model['m'+str(step+1)+'_Substrate'] = 0
        model['m'+str(step+1)+'_Enzyme'] = 0
        model['m'+str(step+1)+'_Activated_promoter'] = 0
    model['m'+str(nsteps)+'_Product'] = 0
    

class metPath:
    def __init__(self, steps):
        """ Init model and parameters """
        self.steps = steps
        self.model = modelTemplate( steps )
        self.vals = []
        for i in np.arange( steps ):
            self.vals.append( instance() )
    
    def sample(self, initSubstrate=1.0):
        """ Create a sample of the model 
            with given inital substrate concentration.
            Start inducers.
        """
        for i in np.arange( self.steps ):
            v = self.vals[i]
            for group in v:
                for x in v[group]:
                    (mean, std) = v[group][x]
                    p = np.random.normal( mean, std )
                    param = 'm{}_{}_{}'.format( i+1, group, x )
                    self.model[ param ] = p      
        initModel( self.model )
        self.model[ 'm1_Substrate' ] = initSubstrate
        for i in np.arange( self.steps ):
            self.model[ 'm{}_Inducer'.format(i+1) ] = 1.0


















