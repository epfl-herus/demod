======================
Initialization
======================

At the beginning of each run, the LibraryNAME-based model generates a single 
or multiple instances of households.
Depending on the purpose of the simulation and the modules selected, 
each household can be associated with a set of devices including appliances, 
water fixtures, light bulbs, heating and control systems, 
and electric vehicles. 

The modules available for initialization of different devices are presented 
below. 


Households 
----------

The modeler can select the number of instances of households to be generated 
for the simulation and the number of residents that compose them. 

According to the chosen modules for the simulation, it can be necessary 
to specify additional parameters, such as household typology
or age, gender and employment status of the members. 



Installed appliances
---------------------
The initialization of appliances can be done manually if the goal is to 
simulate the behavior of a home with a predefined configuration, 
otherwise stochastic methods can be applied.

Stochastic appliance initialization: Germany

    For the German case, appliance stochastic initialization makes use of [Destatis2020]_, 
    which allows for the consideration of dependence on socio-demographic data 
    such as household size, typology, income and employment status. 
    
    Table
    
    
Stochastic appliance initialization: UK

    For the UK, appliances can be initialized using the CREST_ approach 
    that considers household size to estimate the probability of the 
    presence of one or more of the 33 available types. 
    
    Table


Lighting bulbs
--------------

As for the appliances, the initializatione of the light bulbs can be done
manually or in a stochastic way.

Stochastic light bulbs initialization: Germany

    Frondel2019
    
    
    
Stochastic light bulbs initialization: UK

    CREST



Dwelling
--------------

Currently dwelling modules are provided with UK specific parameters (continue)

    Table


Heating system
----------------

    


Electric vehicle
----------------


Data requirements summary
--------------------------


References
----------

.. [Destatis2020]
    Statistisches Bundesamt - Destatis (2020) Laufende wirtschaftsrechnungen 
    ausstattung privater haushaltemit ausgewahlten gebrauchsgutern.       
 
 
 .. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LINKs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _German-time-use: https://www.forschungsdatenzentrum.de/de/haushalte/zve

.. _CREST: https://www.lboro.ac.uk/research/crest/demand-model/ 