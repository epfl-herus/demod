"""Setup file for uploading to https://pypi.org/ .

This file was created thanks to:
https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56

When a next release is to come, you need to do the following:
    - change version in this file
    - in github:
        click on the tab “releases” and then on “Create a new release”
        define a Tag verion
        Add a release title (v0.x) and a description (not that important),
        then click on “publish release”
        Now you see a new release and under Assets a link to Source Code (tar.gz)
        Right-click on this link and chose Copy Link Address.
        Paste this link into the download_url field in this file(setup.py).
    - Once you have changed the versions in this file and in Github
    - open a terminal in the folder where setup.py is and run from terminal:
    - if not installed : " pip install twine "
    - " python setup.py sdist "
    - " twine upload dist/* "

"""
from setuptools import PackageFinder, setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="demod",
    version="0.1-3",
    download_url = 'https://github.com/epfl-herus/demod/archive/refs/tags/v0.1.tar.gz',
    description="Domestic Energy Demand Modelling Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GNU General Public License v3',
    url="https://github.com/epfl-herus/demod",
    author="Matteo Barsanti, Lionel Constantin, HERUS, Ecole Polytechnique Fédérale de Lausanne",
    author_email="demod@groupes.epfl.ch",
    keywords=[
        'energy', 'demand', 'simulation', 'modelling', 'load', 'electricity',
        'power', 'appliance', 'heating', 'household', 'DSM', 'dataset'
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.19",
        "scipy>=1.4",
        "pandas>=1.2",
        "openpyxl>=2.6",
    ],
    classifiers=[
        #  "4 - Beta" or "5 - Production/Stable"
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        # Specify which pyhton versions are supported
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
  ],
)


