import sys
if int(sys.version[0]) == 3:
    from easypy._py3 import *
    PY3 = True
    PY2 = False
else:
    from easypy._py2 import *
    PY3 = False
    PY2 = True
