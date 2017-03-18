class Predicate(object):

    def test(self, obj):
        raise NotImplementedError()

    def __call__(self, obj):
        return self.test(obj)

    def __eq__(self, obj):
        return self.test(obj)

    def __ne__(self, obj):
        return not self == obj

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __str__(self):
        return self.describe()

    def __repr__(self):
        return "<%s>" % self

    def describe(self, variable="X"):
        return self._describe(variable) + "?"

    def _describe(self, variable):
        raise NotImplementedError()


class FunctionPredicate(Predicate):

    def __init__(self, func, description=None):
        super(FunctionPredicate, self).__init__()
        self.func = func
        if description is None:
            description = func.__doc__
        self.description = description

    def test(self, obj):
        if isinstance(obj, FunctionPredicate):
            return obj.func is self.func
        else:
            return self.func(obj)

    def _describe(self, variable):
        if self.description:
            return self.description % dict(var=variable)
        else:
            return "%s(%s)" % (self.func,variable)


class Equality(Predicate):

    def __init__(self, value):
        super(Equality, self).__init__()
        self.value = value

    def test(self, obj):
        if isinstance(obj, Equality):
            return obj.value == self.value
        else:
            return obj == self.value

    def _describe(self, variable):
        return "%s==%s" % (variable, str(self.value))


Inequality = lambda value: Not(Equality(value))


class Or(Predicate):

    def __init__(self, *preds):
        super(Or, self).__init__()
        self.preds = map(make_predicate, preds)

    def test(self, obj):
        return any(p == obj for p in self.preds)

    def _describe(self, variable):
        return " OR ".join("(%s)" % pred._describe(variable) for pred in self.preds)


class And(Predicate):

    def __init__(self, *preds):
        super(And, self).__init__()
        self.preds = map(make_predicate, preds)

    def test(self, obj):
        return all(p == obj for p in self.preds)

    def _describe(self, variable):
        return " AND ".join("(%s)" % pred._describe(variable) for pred in self.preds)


class Not(Predicate):

    def __init__(self, pred):
        super(Not, self).__init__()
        self.pred = make_predicate(pred)

    def test(self, obj):
        return not self.pred == obj

    def _describe(self, variable):
        return "NOT(%s)" % (self.pred._describe(variable),)


class _Dummy(Predicate):

    def __init__(self, retval, description=""):
        self.retval = retval
        self.description = description

    def test(self, other):
        return self.retval

    def _describe(self, variable):
        return self.description

IGNORE = _Dummy(True, "ANYTHING")
FAIL = _Dummy(False, "NOTHING")


def make_predicate(expr):
    """
    common utility for making various expressions into predicates
    """
    if isinstance(expr, Predicate):
        return expr
    elif isinstance(expr, type):
        return FunctionPredicate(lambda obj, type=expr: isinstance(obj, type))
    elif callable(expr):
        return FunctionPredicate(expr)
    else:
        return Equality(expr)

P = make_predicate
