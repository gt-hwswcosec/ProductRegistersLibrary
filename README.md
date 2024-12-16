# ProductRegistersLibrary
A collection of tools to help work with and investigate product registers and their applications. Compatible with Python>=3.12. Tested on Windows 10, Windows 11, and Red Hat Enterprise Linux (RHEL) 7.

**Dependencies**

`memoization numba numpy galois python-sat`

**Installation Note for GATech ECE RedHat Servers:** The ECE servers have several versions of Python installed, but our library is compatible with Python 3.12 and newer. So, use the `python3.12` command when using the library on the RHEL machines.

Moreover, to install the dependencies on the RHEL machines, you must (i) install the packages as a user (due to limited permissions) and (ii) install them for use with Python 3.12. To do so, use the command `python3.12 -m pip install --user <package name>`

**Imports**
```
# Basic constructs:
from ProductRegisters.FeedbackRegister import FeedbackRegister
from ProductRegisters.FeedbackFunctions import *

# Boolean logic and chaining templates
from ProductRegisters.BooleanLogic import *
from ProductRegisters.BooleanLogic.ChainingGeneration.Templates import *
import ProductRegisters.BooleanLogic.ChainingGeneration.TemplateBuilding

# Berlekamp-Massey and variants
from ProductRegisters.Tools.RegisterSynthesis.lfsrSynthesis import *
from ProductRegisters.Tools.RegisterSynthesis.fcsrSynthesis import *
from ProductRegisters.Tools.RegisterSynthesis.nlfsrSynthesis import *

# Tools and other extraneous files
import ProductRegisters.Tools.ResolventSolving as ResolventSolving
from ProductRegisters.Tools.RootCounting.MonomialProfile import *

# Cryptanalysis:
from ProductRegisters.Cryptanalysis.cube_attacks import *
from ProductRegisters.Cryptanalysis.utility import *
```

**Working with the Library**

A comprehensive set of examples can be found at https://github.gatech.edu/ProductRegistersGroup/ProductRegistersImplementations

**Importing the Library**

The library can be imported using pip in three different ways.

Note 1: Currently the ability to update pip installs of the library is untested; however, new installations should install the latest version. Version numbers must be manually updated in the pyproject.toml file, at the moment. 

Note 2: Pip Install Options 1 and 2 tested on MacOS 11.6 with recent Miniconda versions (22.11.1 – 24.7.1)

1. Miniconda environment YAML file (from GitHub):

    ```
    dependencies: 
        - pip
        - pip:
            - "git+https://github.gatech.edu/ProductRegistersGroup/ProductRegistersLibrary.git"
    ```

2. Miniconda environment YAML file (from local after cloning repo):

    ```
    dependencies: 
        - pip
        - pip:
            - "<local path to cloned repo>"
    ```

3. Standard pip install (NOT tested):

    `pip install git+https://github.gatech.edu/ProductRegistersGroup/ProductRegistersLibrary.git`
    
