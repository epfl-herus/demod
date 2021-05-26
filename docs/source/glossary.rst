.. _glossary:


Glossary
========

.. glossary::

   bottom-up model
      "The bottom-up approach encompasses all models which use
      input data from a hierarchal level less than that of the sector as a
      whole. Models can account for the energy consumption of
      individual end-uses, individual houses, or groups of houses and
      are then extrapolated to represent the region or nation based on
      the representative weight of the modeled sample."
      [Swan2009]_

   pdf
      probability density function.

   TPMs
      Transition Probablity Matrixs. It defines the probability 
      of a simulated component in a Markov chain of channging 
      from on state to another.
      In demod, TPMs are a set of TPM that remebers 
      how the matrices evolve with time.
      By convention, the i-eth TPM define the probabilities for transition from
      state i-1 to i.

   subgroup
      Represent a set of social, technical, or economical information of
      a specific kind of the population.
      See how demod implement subgroups:
      :py:attr:`~demod.utils.cards_doc.Params.subgroup`.