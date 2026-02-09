import pathlib

import datasets


def thermoml_dataset_from_xml(
    path: str | pathlib.Path,
) -> datasets.Dataset:
    """Load a ThermoML dataset from an XML file


    Parameters
    ----------
    path : str | pathlib.Path
        The path to the ThermoML XML file to load.


    Returns
    -------
    datasets.Dataset
        A ThermoML HuggingFace Dataset containing the data from the ThermoML XML file,
        with the schema defined by :py:attr:`descent.targets.thermoml.DATA_SCHEMA`.
    """
    ...


def thermoml_dataset_from_doi(doi: str) -> datasets.Dataset:
    """Download a ThermoML dataset from a DOI


    Parameters
    ----------
    doi : str | pathlib.Path
        The DOI of the ThermoML dataset to load.


    Returns
    -------
    datasets.Dataset
        A ThermoML HuggingFace Dataset containing the data from the ThermoML XML file,
        with the schema defined by :py:attr:`descent.targets.thermoml.DATA_SCHEMA`.
    """
    ...


def download_thermoml_to_dataset(
    url: str = "https://data.nist.gov/od/ds/mds2-2422/ThermoML.v2020-09-30.tgz",
) -> datasets.Dataset:
    """Download the ThermoML dataset from the given URL and
    load it into a HuggingFace Dataset.

    Akin to
    openff.evaluator.datasets.curation.components.thermoml.ImportThermoMLDataSchema

    Parameters
    ----------
    url : str, optional
        The URL to download the ThermoML dataset from, by default
        "https://data.nist.gov/od/ds/mds2-2422/ThermoML.v2020-09-30.tgz"

    Returns
    -------
    datasets.Dataset
        A ThermoML HuggingFace Dataset containing the data from the ThermoML XML file,
        with the schema defined by :py:attr:`descent.targets.thermoml.DATA_SCHEMA`.
    """
    ...
