#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Current version
version_major = 1
version_minor = 0
version_micro = 0

# Define default SPM STANDALONE path for the package
DEFAULT_SPM_STANDALONE_PATH = "/i2bm/local/bin/spm12"

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
[pyFunTK]
"""
long_description = """
==================================
pyFunTK: Python Functional ToolKit
==================================


[pyfuntk] A Python project that provides a wrapping over common libraries to
process functional volumes.
"""

# Main setup parameters
NAME = "pyFunTK"
ORGANISATION = "CEA"
MAINTAINER = "Antoine Grigis"
MAINTAINER_EMAIL = "antoine.grigis@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/neurospin/pyfuntk"
DOWNLOAD_URL = "https://github.com/neurospin/pyfuntk"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = "pyFunTK developers"
AUTHOR_EMAIL = "antoine.grigis@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["pyfuntk"]
REQUIRES = [
    "numpy>=1.6.1",
    "nibabel>=2.0.2"
]
EXTRA_REQUIRES = {}
