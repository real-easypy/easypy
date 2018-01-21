from copy import deepcopy

from .exceptions import TException
from .tokens import MANDATORY
from .collections import ListCollection, PythonOrderedDict, iterable
from .bunch import Bunch, bunchify


class InvalidFieldType(TException):
    template = 'Bad field type {field_type} - {reason}'


class NotFields(TException, AttributeError):
    template = '{field_names} are not fields of {typed_struct.__name__}'


class NotAField(TException, AttributeError):
    template = '{field_name} is not a field of {typed_struct.__name__}'


class NotAKey(TException, KeyError):
    template = '{field_name} is not a key of {typed_struct.__name__}'


class MissingField(TException, TypeError):
    template = '{typed_struct.__name__} is missing mandatory field {field.name}'


class FieldTypeMismatch(TException, TypeError):
    template = '{field.name} is {expected_type}, not {received_type}'


class FieldKeyTypeMismatch(TException, TypeError):
    template = 'keys of {field.name} are {expected_type}, not {received_type}'


class FieldCollectionTypeMismatch(TException, TypeError):
    template = '{field.name} is {field.collection_type.__name__} - not {received_type.__name__}'


class Field(object):
    """
    Define a single field in a TypedStruct

        type: The type of the field
              * Use [<type>] for a typechecked list
              * Use {<key_type>: <type>} for a typechecked dict
              * Use {str: <type>} for a typechecked Bunch
        default: Default value for the field
        preprocess: A function for preprocessing the values assigned to the field:
                    * If the field can be converted from different types,
                      should do the conversion and return the converted value.
                    * If there are validation checks, should perform them and
                      raise an exception if they fail.
                    * Use the field.add_validation() and field.add_conversion()
                      helpers to modify the preprocess function without
                      manually setting this parameter.
                    * Must return a value of the field's type.
        meta: Custom metadata. Can be queried by tools that use reflection on
              the TypedStruct object.
    """

    def __init__(self, type, *, default=MANDATORY, preprocess=None, meta=Bunch()):
        if isinstance(type, list):
            if len(type) != 1:
                raise InvalidFieldType(field_type=type,
                                       reason='Collection-style fields must be defined with a single item collection')
            type, = type
            self.collection_type = TypedList
            self.key_type = None
        elif isinstance(type, dict):
            if len(type) != 1:
                raise InvalidFieldType(field_type=type,
                                       reason='Collection-style fields must be defined with a single item collection')
            (self.key_type, type), = type.items()
            if self.key_type is str:
                self.collection_type = TypedBunch
            else:
                self.collection_type = TypedDict
        else:
            self.collection_type = None
            self.key_type = None

        import builtins
        if not isinstance(type, builtins.type):
            raise InvalidFieldType(field_type=type,
                                   reason='Not a type')
        self.type = type

        self.default = default
        # NOTE: _validate_type() will be also be called in _process_new_value()
        # in case people implement their own preprocess function, but we still
        # want to call it before the validatiors added with add_validation() so
        # that they won't throw a generic TypeError that will prevent the more
        # specific FieldTypeMismatch.
        self.preprocess = preprocess or self._validate_type
        self.meta = Bunch(meta)
        self.name = None

        if issubclass(self.type, TypedStruct):
            if self.default is MANDATORY:
                try:
                    self.default = self.type()
                except MissingField:
                    pass  # keep it mandatory

            orig_preprocess = self.preprocess

            def preprocess(obj):
                if isinstance(obj, dict) and not isinstance(obj, TypedStruct):
                    obj = self.type.from_dict(obj)
                return orig_preprocess(obj)
            self.preprocess = preprocess

    def add_validation(self, predicate, ex_type, *ex_args, **ex_kwargs):
        """
        Add a validation on values assigned to the field

            predicate: The validation function. Should return True if the value is OK.
            ex_type: The exception to raise if the predicate returns False on a value.
            ex_args, ex_kwargs: Arguments for the exception.

        Calling this method will modify the preprocess function.
        """
        orig_preprocess = self.preprocess

        def new_preprocess(obj):
            obj = orig_preprocess(obj)
            if not predicate(obj):
                raise ex_type(*ex_args, **ex_kwargs)
            return obj
        self.preprocess = new_preprocess

    def add_conversion(self, predicate, conversion):
        """
        Add a validation on values assigned to the field

            predicate: Only perform the conversion if this returns True.
            conversion: A conversion function. The returned value must be of
                        the type of the field.

        Calling this method will modify the preprocess function.
        """
        if isinstance(predicate, type):
            typ = predicate

            def predicate(obj):
                return isinstance(obj, typ)

        orig_preprocess = self.preprocess

        def new_preprocess(obj):
            if predicate(obj):
                obj = conversion(obj)
                assert isinstance(obj, self.type), 'Conversion %s converted to %s - expected %s' % (
                    conversion, type(obj), self.type)
            return orig_preprocess(obj)
        self.preprocess = new_preprocess

    def _validate_type(self, value):
        if not isinstance(value, self.type):
            raise FieldTypeMismatch(field=self,
                                    expected_type=self.type,
                                    received_type=type(value)) from None
        return value

    def _named(self, name):
        assert self.name is None, '%s is already named' % self
        self.name = name
        return self

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        return obj[self.name]

    def __set__(self, obj, value):
        if self.collection_type is not None:
            try:
                existing = obj[self.name]
            except KeyError:  # First time - called from TypedStruct c'tor
                assert isinstance(value, self.collection_type), '%s is not a %s' % (value, self.collection_type)
            else:  # Collection already exists - let it handle the assignment
                if existing is not value:
                    existing._set(value)
                return

        else:
            value = self._process_new_value(value)

        return super(TypedStruct, obj).__setitem__(self.name, value)

    def _process_new_value(self, value):
        value = self.preprocess(value)
        self._validate_type(value)
        return value


class TypedCollection(object):
    def __init__(self, owner, field):
        self._owner = owner
        self._field = field
        super().__init__()


class TypedList(TypedCollection, ListCollection):
    def _set(self, values):
        if not iterable(values):
            raise FieldCollectionTypeMismatch(field=self._field, received_type=type(values))
        # Must eagerly verify them all before clearing the list
        values = [self._field._process_new_value(value) for value in values]
        self.clear()
        # Use super to avoid calling _process_new_value twice
        super().extend(values)

    def __setitem__(self, index, value):
        value = self._field._process_new_value(value)
        return super().__setitem__(index, value)

    def append(self, value):
        value = self._field._process_new_value(value)
        return super().append(value)

    def insert(self, index, value):
        value = self._field._process_new_value(value)
        return super().insert(index, value)

    def __iadd__(self, other):
        other = map(self._field._process_new_value, other)
        return super().__iadd__(other)

    def extend(self, iterable):
        iterable = map(self._field._process_new_value, iterable)
        return super().extend(iterable)

    # TODO: We may decide to override these methods in the future. They don't
    # need to validate members, but do change the collection - so if we want to
    # add events we need to override them:
    #       __delitem__(self, key)
    #       __imul__(self, other)
    #       clear(self)
    #       remove(self, value)
    #       reverse(self)
    #       sort(self, key=None, reverse=False)
    #       pop(self, index=None)


class TypedDict(TypedCollection, dict):
    def _set(self, values):
        if not isinstance(values, dict):
            raise FieldCollectionTypeMismatch(field=self._field, received_type=type(values))
        # Must eagerly verify them all before clearing the list
        values = {self._process_new_key(k): self._field._process_new_value(v) for k, v in values.items()}
        self.clear()
        # # Use super to avoid calling _process_new_value twice
        super().update(values)

    def _process_new_key(self, key):
        if not isinstance(key, self._field.key_type):
            raise FieldKeyTypeMismatch(field=self._field,
                                       expected_type=self._field.key_type,
                                       received_type=type(key))
        return key

    def __setitem__(self, key, value):
        key = self._process_new_key(key)
        value = self._field._process_new_value(value)
        return super().__setitem__(key, value)

    def setdefault(self, key, default=None):
        key = self._process_new_key(key)
        default = self._field._process_new_value(default)
        return super().setdefault(key, default)

    def update(self, dct={}, **kwargs):
        try:
            items = dct.items
        except AttributeError:
            items = dct
        else:
            items = items()

        for k, v in items:
            self[k] = v

        for k, v in kwargs.items():
            self[k] = v

    # TODO: We may decide to override these methods in the future. They don't
    # need to validate members, but do change the collection - so if we want to
    # add events we need to override them:
    #       __delitem__(self, key)
    #       clear(self)
    #       pop(k[,d])
    #       popitem(self)


class TypedBunch(TypedDict, Bunch):
    def __init__(self, owner, field):
        # Hack around Bunch's __getattr__()
        self.__dict__.update(_field=field, _owner=owner)


TypedBunch.__name__ = 'Bunch'  # make it look like a Bunch when formatted


class TypedStructMeta(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        return PythonOrderedDict()

    def __new__(cls, name, bases, dct):
        def altered_dct_gen():
            fields = []
            yield '_fields', fields
            for k, v in dct.items():
                if isinstance(v, Field):
                    v = v._named(k)
                    fields.append(v)
                else:
                    try:
                        v = Field(v)._named(k)
                        fields.append(v)
                    except InvalidFieldType:
                        pass
                yield k, v
        return super().__new__(cls, name, bases, dict(altered_dct_gen()))


class TypedStruct(dict, metaclass=TypedStructMeta):
    """
    Define typechecked-at-runtime classes.

    * Define fields with types, defaults values, validators etc.
    * Nesting of TypedStructs.
    * Typechecked collections.

    Example:

        import easypy.typed_struct as ts

        class Foo(ts.TypedStruct):
            a = ts.Field(int, default=14)
            b = ts.Field([str], default=['1', '2', '3'])  # list of strings

            # Shorter style - less customizable but you get completions
            c = bool
            d = [float]
    """

    @classmethod
    def _get_field(cls, name, error_variant):
        field = getattr(cls, name, None)
        if not isinstance(field, Field):
            raise error_variant(typed_struct=cls,
                                field_name=name)
        return field

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def items(self):
        for field in self._fields:
            yield field.name, field.__get__(self)

    def to_dict(self):
        return {name: value.to_dict() if isinstance(value, TypedStruct) else value
                for name, value in self.items()}

    def to_bunch(self):
        return bunchify(self)

    def __init__(self, **kwargs):
        for field in self._fields:
            if field.collection_type is not None:
                value = field.collection_type(self, field)
                try:
                    raw_value = kwargs.pop(field.name)
                except KeyError:
                    if field.default is not MANDATORY:
                        value._set(deepcopy(field.default))
                else:
                    value._set(raw_value)
            else:
                try:
                    value = kwargs.pop(field.name)
                except KeyError:
                    if field.default is MANDATORY:
                        raise MissingField(typed_struct=type(self),
                                           field=field) from None
                    value = deepcopy(field.default)

            field.__set__(self, value)
        if kwargs:
            raise NotFields(typed_struct=type(self), field_names=', '.join(kwargs.keys()))

    def __delitem__(self, key):
        assert False, 'deletion is not allowed'

    def __setitem__(self, key, value):
        self._get_field(key, NotAKey).__set__(self, value)

    def __setattr__(self, name, value):
        self._get_field(name, NotAField).__set__(self, value)

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join('%s=%r' % pair for pair in self.items()))

    def _repr_pretty_(self, *args, **kwargs):
        from easypy.humanize import ipython_mapping_repr
        return ipython_mapping_repr(self, *args, **kwargs)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return super().__eq__(other)

    def __ne__(self, other):
        return not self == other
