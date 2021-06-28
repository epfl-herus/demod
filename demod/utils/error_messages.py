"""Different formatted error messages that can be used in Demod.

You can use these messages like this::

    raise ValueError(UNKOWN_POPULATION_TYPE.format(
        population_type='resident_number',
        dataset=YouDataLoader
    ))


* UNKOWN_POPULATION_TYPE(population_type, dataset)
* ALGO_REQUIRES_LOADING_METHOD(algo, simulator, loading_method, dataset)
* NOT_IMPLEMENTED_IN_DATASET_FOR_VERSION(not_implemented, dataset, version)
* UNIMPLEMENTED_ALGO_IN_METHOD(algo, method)

"""


ALGO_REQUIRES_LOADING_METHOD = (
    "Algorithm '{algo}' in '{simulator}' requires '{loading_method}' "
    " from dataset. Could not find '{loading_method}' in '{dataset}'."
)

USE_OTHER_ALGOS_FOR_ALGONAME = (
    "Try using {other_algos} as {algo_name} instead"
)

UNKOWN_POPULATION_TYPE = (
    "Unkown population type '{population_type}' for dataset"
    " {dataset}."
)

NOT_IMPLEMENTED_IN_DATASET_FOR_VERSION = (
    "{not_implemented} is not implemented for {dataset} "
    "version '{version}'."
)

UNIMPLEMENTED_ALGO_IN_METHOD = (
    "Algorithm '{algo}' is not implemeted in {method}"
)

DATASET_CANNOT_DISTINGUISH_ON_SUBGROUPS = (
    "{dataset} cannot distinguish {not_distinguishable}  based on subgroups.\n"
    "{not_distinguishable} will be the same for all subgroups."
)
