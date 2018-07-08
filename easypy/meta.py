from abc import ABCMeta, abstractmethod
from functools import wraps
from collections import OrderedDict
from enum import Enum

from .decorations import kwargs_resilient


class EasyMeta(ABCMeta):
    """
    Base class for various meta-magic mixins.

    Use the hooks in :class:`EasyMetaHooks`, decorated with `@EasyMeta.Hook`,
    to add functionality. Multiple hooks with the same name can be defined,
    and they will all be invoked sequentially.
    """

    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        dsl = EasyMetaDslDict()
        return dsl

    class Hook(object):
        def __init__(self, dlg):
            self.dlg = dlg

    def __init__(cls, name, bases, dct, **kwargs):
        super().__init__(name, bases, dct)

    def __new__(mcs, name, bases, dct, **kwargs):
        hooks = EasyMetaHooks(class_kwargs=kwargs)

        for base in bases:
            if isinstance(base, EasyMeta):
                hooks.extend(base._em_hooks)

        if hooks.hooks['before_subclass_init']:
            bases = list(bases)
            hooks.before_subclass_init(name, bases, dct)
            bases = tuple(bases)

        new_type = super().__new__(mcs, name, bases, dct)

        new_type._em_hooks = hooks

        hooks.after_subclass_init(new_type)

        hooks.extend(dct.hooks)

        return new_type


class EasyMetaHooks:
    """
    Hooks for ``EasyMeta``
    """

    HOOK_NAMES = []

    def hook(dlg, HOOK_NAMES=HOOK_NAMES):
        HOOK_NAMES.append(dlg.__name__)

        @wraps(dlg)
        def hook(self, *args, **kwargs):
            kwargs_resilience = kwargs_resilient(negligible=self.class_kwargs.keys())
            kwargs.update((k, v) for k, v in self.class_kwargs.items() if k not in kwargs)

            for hook in self.hooks[dlg.__name__]:
                kwargs_resilience(hook)(*args, **kwargs)

        return hook

    def __init__(self, class_kwargs={}):
        self.hooks = {name: [] for name in self.HOOK_NAMES}
        self.class_kwargs = class_kwargs

    def add(self, hook):
        self.hooks[hook.__name__].append(hook)

    def extend(self, other):
        for k, v in other.hooks.items():
            self.hooks[k].extend(v)

    @hook
    def before_subclass_init(self, name, bases, dct):
        """
        Invoked before a subclass is being initialized

        :param name: The name of the class. Immutable.
        :param list bases: The bases of the class. A list, so it can be changed.
        :param dct: The body of the class.

        >>> class NoB(metaclass=EasyMeta):
        >>>     @EasyMeta.Hook
        >>>     def before_subclass_init(name, bases, dct):
        >>>         dct.pop('b', None)
        >>>
        >>> class Foo(NoB):
        >>>     a = 1
        >>>     b = 2
        >>>
        >>> Foo.a
        1
        >>> Foo.b
        AttributeError: type object 'Foo' has no attribute 'b'
        """

    @hook
    def after_subclass_init(self, cls):
        """
        Invoked after a subclass is being initialized

        >>> class PrintTheName(metaclass=EasyMeta):
        >>>     @EasyMeta.Hook
        >>>     def after_subclass_init(cls):
        >>>         print('Declared', cls.__name__)
        >>>
        >>>
        >>> class Foo(PrintTheName):
        >>>     pass
        Declared Foo
        """


class EasyMetaDslDict(OrderedDict):
    def __init__(self):
        super().__init__()
        self.hooks = EasyMetaHooks()

    def __setitem__(self, name, value):
        if isinstance(value, EasyMeta.Hook):
            self.hooks.add(value.dlg)
        else:
            return super().__setitem__(name, value)


class EasyMixinStage(Enum):
    BASE_MIXIN_CLASS = 1
    MIXIN_GENERATOR = 2
    ACTUAL_MIXIN_SPECS = 3


class EasyMixinMeta(ABCMeta):
    def __new__(mcs, name, bases, dct, **kwargs):
        try:
            stage = dct['_easy_mixin_stage_']
        except KeyError:
            stage = min(b._easy_mixin_stage_ for b in bases if hasattr(b, '_easy_mixin_stage_'))
            stage = EasyMixinStage(stage.value + 1)
        if stage == EasyMixinStage.ACTUAL_MIXIN_SPECS:
            base, = bases
            return base(name, bases, dct)._generate_class()
        else:
            dct['_easy_mixin_stage_'] = stage
            return super().__new__(mcs, name, bases, dct)


class EasyMixin(metaclass=EasyMixinMeta):
    """
    Create mixins creators.

    Direct subclasses (hereinafter "mixin creators") of this class will be
    created normally, but subclasses of these subclasses (hereinafter "mixins")
    will be new classes that are not subclasses of neither the mixin creator nor
    ``EasyMixin`` nor their other base classes, and will not necessarily contain
    the content of their bodies.  Instead, the mixin class' body and its other
    base classes will be passed to methods of the mixin creator, which will be
    able to affect the resulting mixin class.

    >>> class Foo(EasyMixin):  # the mixin creator
    >>>     def prepare(self):
    >>>         # `orig_dct` contains the original body - to affect the new one
    >>>         # we use `dct`
    >>>         self.dct['value_of_' + self.name] = self.orig_dct['value_of_name']
    >>>
    >>> class Bar(Foo):  # the mixin
    >>>     value_of_name = 'Baz'
    >>>
    >>> Bar.value_of_Bar
    'Baz'
    """

    _easy_mixin_stage_ = EasyMixinStage.BASE_MIXIN_CLASS
    metaclass = EasyMeta
    """The metaclass for the mixin"""

    def __init__(self, name, bases, dct):
        self.name = name
        """The name of the to-be-created mixin. Can be changed."""
        self.bases = ()
        """The bases of the to-be-created mixin. Can be changed."""
        self.orig_bases = bases
        """The bases of the mixin's body. Not carried to the created mixin."""
        self.orig_dct = dct
        """The the mixin's body. Not carried to the created mixin."""

    @abstractmethod
    def prepare(self):
        """
        Override this to control the mixin creation.

        * Alter ``self.name``, ``self.bases`` and ``self.dct``.
        * Use ``self.add_hook`` to add easymeta hooks.
        * Access the declaration of the mixin with ``self.orig_bases`` and self.orig_dct``.
        """

    def _generate_class(self):
        self.dct = EasyMetaDslDict()
        self.dct.update(__module__=self.orig_dct['__module__'], __qualname__=self.orig_dct['__qualname__'])
        self.prepare()
        return self.metaclass(self.name, self.bases, self.dct)

    def add_hook(self, fn):
        """
        Add EasyMeta hooks to the created mixins. Use inside ``prepare``.

        >>> class Foo(EasyMixin):
        >>>     def prepare(self):
        >>>         @self.add_hook
        >>>         def after_subclass_init(cls):
        >>>             # Note that the hook will not run on Bar - only on Baz
        >>>             print(self.orig_dct['template'].format(cls))
        >>>
        >>> class Bar(Foo):
        >>>     template = 'Creating subclass {0.__name__}'
        >>>
        >>> class Baz(Bar):
        >>>     pass
        Creating subclass Baz
        """
        self.dct[fn.__name__] = EasyMeta.Hook(fn)
