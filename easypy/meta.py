import inspect
from abc import ABCMeta
from functools import wraps
from collections import OrderedDict

from .misc import kwargs_resilient
from .collections import as_list


class EasyMeta(ABCMeta):
    """
    This class helps in implementing various metaclass-based magic.
    Implement the ``before_subclass_init`` and/or ``after_subclass_init`` (see defined in :class:``EasyMetaHooks``)
    to modify the class spec, register subclasses, etc.

    Each hook method can be defined more than once (with the same name), and methods will all be invoked sequentially.

    Important: the hooks are not invoked on the class that implements the hooks - only on it subclasses.
    """

    class EasyMetaHooks:
        """
        The class defines the available EasyMeta hooks (slots), and registers handlers for EasyMeta derivatives
        """

        HOOK_NAMES = []

        def _define_hook(dlg, HOOK_NAMES=HOOK_NAMES):
            HOOK_NAMES.append(dlg.__name__)

            @wraps(dlg)
            def hook(self, *args, **kwargs):
                kwargs_resilience = kwargs_resilient(negligible=self.class_kwargs.keys())
                kwargs.update((k, v) for k, v in self.class_kwargs.items() if k not in kwargs)

                for hook in self._em_hooks[dlg.__name__]:
                    kwargs_resilience(hook)(*args, **kwargs)

            return hook

        @_define_hook
        def after_subclass_init(self, cls):
            """
            Invoked after a subclass is being initialized

            >>> class PrintTheName(metaclass=EasyMeta):
            ...     @EasyMeta.Hook
            ...     def after_subclass_init(cls):
            ...         print('Declared', cls.__name__)
            ...
            ...
            >>> class Foo(PrintTheName):
            ...     pass
            Declared Foo
            """

        @_define_hook
        def before_subclass_init(self, name, bases, dct):
            """
            Invoked after a subclass is being initialized

            >>> class AddMember(metaclass=EasyMeta):
            ...     @EasyMeta.Hook
            ...     def before_subclass_init(name, bases, dct):
            ...         dct['foo'] = 'bar'
            ...
            ...
            >>> class Foo(AddMember):
            ...     pass
            >>> Foo.foo
            'bar'
            """

        def __init__(self, class_kwargs={}):
            self._em_hooks = {name: [] for name in self.HOOK_NAMES}
            self.class_kwargs = class_kwargs

        def add(self, hook):
            self._em_hooks[hook.__name__].append(hook)

        def extend(self, other):
            for k, v in other._em_hooks.items():
                slot = self._em_hooks[k]
                slot.extend(hook for hook in v if hook not in slot)

    class EasyMetaDslDict(OrderedDict):
        """
        This class is used as the namespace for the user's class.
        Any member decorated as an EasyMeta hook gets removed from the namespace and
        put into a special 'hooks' collection. The EasyMeta metaclass then invokes the
        delegates registered under those hooks
        """
        def __init__(self, class_name):
            super().__init__()
            self._class_name = class_name
            self._em_hooks = EasyMeta.EasyMetaHooks()

        def __setitem__(self, name, value):
            if isinstance(value, EasyMeta.Hook):
                self._em_hooks.add(value.dlg)
            else:
                return super().__setitem__(name, value)

    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        dsl = metacls.EasyMetaDslDict(class_name=name)
        return dsl

    class Hook(object):
        def __init__(self, dlg):
            self.dlg = dlg

    def __init__(cls, name, bases, dct, **kwargs):
        super().__init__(name, bases, dct)

    def __new__(mcs, name, bases, dct, **kwargs):
        aggregated_hooks = mcs.EasyMetaHooks(class_kwargs=kwargs)

        bases = list(bases)  # allow the hook to modify the base class list

        for base in bases:
            for sub_base in reversed(inspect.getmro(base)):
                if isinstance(sub_base, EasyMeta):
                    aggregated_hooks.extend(sub_base._em_hooks)

        ns = dict(dct)
        aggregated_hooks.before_subclass_init(name, bases, ns)
        new_type = super().__new__(mcs, name, tuple(bases), ns)
        aggregated_hooks.after_subclass_init(new_type)

        new_type._em_hooks = dct._em_hooks

        return new_type


class GetAllSubclasses(metaclass=EasyMeta):
    """
    Meta-magic mixin for registering subclasses

    The ``get_all_subclasses`` class method will return a list of all subclasses
    of the class it was called on. The class it was called on is not included in
    the list.

    >>> class Foo(GetAllSubclasses):
    ...     pass
    ...
    ...
    >>> class Bar(Foo):
    ...     pass
    ...
    ...
    >>> class Baz(Foo):
    ...     pass
    ...
    ...
    >>> class Qux(Bar):
    ...     pass
    ...
    ...
    >>> Foo.get_all_subclasses()
    [<class 'easypy.meta.Bar'>, <class 'easypy.meta.Qux'>, <class 'easypy.meta.Baz'>]
    >>> Bar.get_all_subclasses()
    [<class 'easypy.meta.Qux'>]
    >>> Baz.get_all_subclasses()
    []
    >>> Qux.get_all_subclasses()
    []
    """

    @EasyMeta.Hook
    def after_subclass_init(cls):
        cls.__direct_subclasses = []
        for base in cls.__bases__:
            if base is not GetAllSubclasses and issubclass(base, GetAllSubclasses):
                base.__direct_subclasses.append(cls)

    @classmethod
    @as_list
    def get_subclasses(cls):
        """
        List immediate subclasses of this class
        """
        yield from cls.__direct_subclasses

    @classmethod
    def iter_all_subclasses(cls, level=0) -> (int, type):
        """
        walk all subclasses of this class
        """
        for subclass in cls.__direct_subclasses:
            yield level, subclass
            yield from subclass.iter_all_subclasses(level=level + 1)

    @classmethod
    @as_list
    def get_all_subclasses(cls):
        """
        List all subclasses of this class
        """
        for level, subclass in cls.iter_all_subclasses():
            yield subclass
