======================
Occupancy and activity
======================

This module allows to simulate household occupancy and activity behaviour
with a 10 min time resolution.
It uses a Markov chain technique to create stochastic profiles using
transition probability matrices based on time use survey data.


Two alternative modules are currently available in demod:

.. _overview_4_States:

4-states occupancy simulator
-----------------------------

It is based on a first-order non-homogeneous Markov chain model,
developed by [McKenna2016]_.
According to this approach, the occupancy status of each resident is defined
by the activity status (*active* or *asleep*)
and location (*home* or *away from home*).

It follows then that there can be :math:`2^2 = 4` different states:
(i) at home and active, (ii) at home and asleep, (iii) away from home and active,
or (iv) away from home and asleep.

The model is non-homogeneous because the coefficients of the transition
probability matrix (TPM) change throughout the day with a timestep
of 10 min.
This approach has the advantage of ensuring greater accuracy
in simulating shared family activities such as mealtime.
However as the number of family members increases the size of TPMs
grow exponentially making proper parameterization difficult.
For a household with N members,
the TPM size can be calculated as :math:`(N + 1)^2`.
Moreover, this approach does not allow for tracking behavioral profiles
of individual residents,
as household occupancy data are provided at the aggregate level.

For illustrative purposes, a graphic representation of the matrix in
the case of a two-person household is shown in :numref:`TPM`.

.. figure:: OverviewFigures/TPM.PNG
    :width: 400
    :alt: Transition probability matrices
    :align: center
    :name: TPM

    Transition probability matrices for a two-person household

.. _overview_transit_occupancy:

Transit occupancy simulator
---------------------------

This approach extend the 4-state occupancy simulator by distinguishing
between 'away for work' and 'away for other'.
This new version of the model complicates the parameterization of TPMs
as it consider three location alternatives.
However, it may be more appropriate for integrating driving
and charging modules for electric vehicles.

In this case the size of TPMs is equal to
:math:`(N + 1) \cdot {3 + N - 1 \choose N}`.
The first term of the product stands for the number of active/asleep people,
while the second term corresponds to their location and
is calculated as the combination with repetition of class N and
a set of 3 alternatives (i.e., 'home', 'away for work' and 'away for other'),
:math:`C^{'}_{(3,N)}={3 + N - 1 \choose N}`.



Other occupancy/activity simulators
------------------------------------

Demod's modular structure allows new simulation modules to be introduced and
tested for performance, such as explit activity simulation
(e.g., see [Yamaguchi2020]_ ).

.. warning:: In this case, however, it is important to consider the
             compatibility of new modules of simulation of the occupancy
             and activity, with those of simulation of the electric and
             thermal demand (see :doc:`API References </api/index>`
             for additional info).



Data requirements summary
--------------------------

======================

+-----------+-----------------------------+-------------------------+------------------+-----------+
| Function  | Model                       | Data                    | Source DE        | Source UK |
+===========+=============================+=========================+==================+===========+
| Occupancy | 4-state occupancy simulator | number of residents     | User             | User      |
|           |                             +-------------------------+------------------+-----------+
|           |                             | week/weekend day        | User             | User      |
|           |                             +-------------------------+------------------+-----------+
|           |                             | 4-states occupancy TPMs | German-time-use_ | CREST_    |
|           +-----------------------------+-------------------------+------------------+-----------+
|           | Transit occupancy simulator | number of residents     | User             | n.a.      |
|           |                             +-------------------------+------------------+           |
|           |                             | week/weekend day        | User             |           |
|           |                             +-------------------------+------------------+           |
|           |                             | transit occupancy TPMs  | German-time-use_ |           |
+-----------+-----------------------------+-------------------------+------------------+-----------+
| activity  | Not yet implemented         | n.a.                    | n.a.             | n.a.      |
+-----------+-----------------------------+-------------------------+------------------+-----------+





References
----------

.. [Destatis2017]
    Statistisches Bundesamt - Destatis (2017) Laufende wirtschaftsrechnungen
    ausstattung privater haushaltemit ausgewahlten gebrauchsgutern.

.. [McKenna2016]
    E. McKenna, M. Thomson (2016) High-resolution stochastic integrated
    thermal-electrical domestic demand model

.. [Yamaguchi2020]
    Y. Yamaguchi, N. Prakash, Y. Simoda (2020) Activity-Based Modeling
    for Integration of Energy Systems for House and Electric Vehicle


 .. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LINKs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _German-time-use: https://www.forschungsdatenzentrum.de/de/haushalte/zve

.. _CREST: https://www.lboro.ac.uk/research/crest/demand-model/