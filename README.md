# ProductRegistersLibrary
A collection of tools to help work with and investigate product registers and their applications. Compatible with Python>=3.12. Tested on Windows 10, Windows 11, and Red Hat Enterprise Linux (RHEL) 7.

**Dependencies**

`memoization numba numpy galois python-sat`

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
    
[![Github All Releases](https://img.shields.io/github/downloads/gt-hwswcosec/ProductRegistersLibrary/total.svg)]()
