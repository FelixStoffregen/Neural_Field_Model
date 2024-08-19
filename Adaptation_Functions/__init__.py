import pkgutil
import importlib

# Initialize an empty __all__ list
__all__ = []

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # Import the module
    module = importlib.import_module('.' + module_name, package=__name__)
    
    # Iterate over all attributes in the module
    for attr_name in dir(module):
        # Get the attribute
        attribute = getattr(module, attr_name)
        
        # Check if the attribute is a callable and not a dunder method
        if callable(attribute) and not attr_name.startswith('__'):
            # Append the attribute name to __all__
            __all__.append(attr_name)
            
            # Import the attribute into the package's namespace
            globals()[attr_name] = attribute

# Now __all__ contains all callable names from all modules in the package
# and they are imported into the package namespace.



# from .A0Basic_Functions import *
# from .A1Points_of_equilibrium import *
# from .A2Local_Model_Stability import *
# from .A3Full_Model_Stability import *
# from .A4Last_Plot import *
# from .A5Simulation import *
# from .A6Plots import *
# from .A7Slow_Oscillations import *

from .Parameters import *