from easypy.exceptions import TException
from easypy.bunch import Bunch


class T(TException):
    template = "The happened: {what}"


def test_pickle_texception():
    import pickle

    t1 = T(what="happened", a=1, b=Bunch(x=[1, 2, 3], y=range(5)))
    t2 = pickle.loads(pickle.dumps(t1))

    assert t1.render() == t2.render()
    assert t1._params == t2._params
