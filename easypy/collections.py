"""
This module helps you with handling and generating collections of items.
"""


from __future__ import absolute_import
import collections
from numbers import Integral
from itertools import chain, islice
from functools import partial
import inspect
import random
from .predicates import make_predicate
from .tokens import UNIQUE
from .decorations import parametrizeable_decorator
from .exceptions import TException

import sys

from collections import OrderedDict as PythonOrderedDict
SUPPORT_GET_SIBLING = True

if sys.version_info[:2] >= (3, 5):
    # In order to support 'get_prev' and 'get_next', we need access to OrderedDict's internal .__map,
    # which we don't have in the C-implementation of the class in Python3.5
    # This hack allows us to get to the pythonic implemenation of OrderedDict
    try:
        from test.support import import_fresh_module
        PythonOrderedDict = import_fresh_module('collections', blocked=['_collections']).OrderedDict
    except ImportError:
        # this happens on 3.6.7-8 due to a bug (https://bugzilla.redhat.com/show_bug.cgi?id=1651245)
        SUPPORT_GET_SIBLING = False


def _format_predicate(pred):
    args = inspect.formatargspec(*inspect.getargspec(pred))
    return "<lambda>" if pred.__name__ == "<lambda>" else "{pred.__name__}{args}".format(**locals())


def _format_filter_string(preds, filters):
    returned = []
    returned.extend(map(_format_predicate, preds))
    returned.extend("{!s}={!r}".format(*item) for item in filters.items())
    return ", ".join(returned)


class ObjectNotFound(TException, LookupError):
    template = "Object not found in {collection!r}, filtered by {_filters}"

    def __init__(self, collection, preds=(), filters={}, **kw):
        preds = getattr(collection, "preds", ()) + preds
        filters = dict(getattr(collection, "filters", {}), **filters)
        filter_str = _format_filter_string(preds, filters)
        super().__init__(collection=collection, _filters=filter_str, **kw)


class TooManyObjectsFound(ObjectNotFound):
    template = "Too many objects ({_found}) found in {collection!r}, {_filters}"


class NotEnoughObjects(ObjectNotFound):
    template = "Not enough objects found in {collection!r}, {_filters}"


class Repr(str):
    """
    A string that represents itself without quotes
    """
    __repr__ = str.__str__


class defaultlist(list):
    """
    Like defaultdict, but for a list.
    Any access to an index that isn't populated will fill it with the value returned by the default factory:

    >>> d = defaultlist(int)
    >>> d[3] += 1
    >>> d
    [0, 0, 0, 1]
    """

    def __init__(self, default_factory):
        self.default_factory = default_factory
        super().__init__()

    def _fill(self, index):
        while len(self) <= index or (len(self) == 0 and index == -1):
            self.append(self.default_factory())

    def __setitem__(self, index, value):
        self._fill(index)
        list.__setitem__(self, index, value)

    def __getitem__(self, index):
        self._fill(index)
        return list.__getitem__(self, index)


def _get_value(obj, attr):
    value = getattr(obj, attr)
    if callable(value):
        value = value()
    return value


def _validate(obj, key, value):
    assert value is not UNIQUE, "Can't specify %s except with '.sample()'" % UNIQUE
    actual = _get_value(obj, key)
    return actual == value


def filters_to_predicates(filters):
    """
    Converts the given dict to a list of predicate callables.
    Used by the various *Collection classes in this module.

    >>> from .bunch import Bunch
    >>> [pred] = filters_to_predicates(dict(foo='bar'))

    >>> obj1 = Bunch(foo='bar')
    >>> pred(obj1)  # checks!
    True

    >>> obj2 = Bunch(bar='foo', foo='bad')
    >>> pred(obj2)  # checks!
    False
    """
    return [make_predicate(partial(_validate, key=key, value=value))
            for key, value in filters.items()]


def filtered(objects, preds, filters):
    """
    Yields the object that pass the predicates (callables), and filters (dict).
    Used by the various *Collection classes in this module.

    >>> from .bunch import Bunch
    >>> l = [Bunch(a=a, b=b) for a in range(3) for b in range(3)]
    >>> f = list(filtered(l, [lambda o: o.a==o.b], dict(a=1)))
    >>> f
    [Bunch(a=1, b=1)]
    """
    preds = [make_predicate(p) for p in preds]
    preds += filters_to_predicates(filters)

    # don't collapse into one line so easier to debug
    for obj in objects:
        first_exception = None
        for pred in preds:
            try:
                if not pred(obj):
                    break
            except Exception as exc:
                if first_exception is None:
                    first_exception = exc
        else:
            if first_exception is not None:
                raise first_exception
            yield obj


def uniquify(objects, num, attrs):
    """
    Find ``num`` objects (in random order) that are unique in their `attrs` attributes.
    Returns empty list if no objects are found to satisfy the uniqueness condition.

    >>> from .bunch import Bunch

    >>> l = [
    ...     Bunch(first='Adam', last='Brody', age='n/a'),
    ...     Bunch(first='Adam', last='Levine', age='n/a'),
    ...     Bunch(first='Adam', last='Sandler', age='n/a'),
    ...     Bunch(first='Amy', last='Adams', age='n/a'),
    ...     Bunch(first='Amy', last='Poehler', age='n/a'),
    ...     Bunch(first='Amy', last='Winehouse', age='n/a')]

    >>> o1, o2 = uniquify(l, 2, ["first", "last"])
    >>> (o1.first != o2.first) and (o1.last != o2.last)
    True

    >>> list(uniquify(l, 3, ["first", "last"]))
    []

    >>> list(uniquify(l, 2, ["first", "age"]))
    []
    """

    object_map = {}
    partitions = collections.defaultdict(lambda: collections.defaultdict(set))

    for object in objects:
        k = id(object)
        object_map[k] = object
        for attr in attrs:
            value = _get_value(object, attr)
            partitions[attr][value].add(k)

    if not all(len(group.values()) >= num for group in partitions.values()):
        return []

    def gen(available, collected=[]):
        if len(collected) == num:
            yield [object_map[k] for k in collected]
            return

        for k in shuffled(available):
            excluded = {i for attr in attrs for i in partitions[attr][_get_value(object_map[k], attr)]}
            remain = available - {k} - excluded
            yield from gen(remain, collected=collected + [k])

    try:
        return next(gen(set(object_map.keys())))
    except StopIteration:
        return []


# Inspired by code written by Rotem Yaari

class ObjectCollectionBase(object):

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        returned = 0
        for _ in self:
            returned += 1
        return returned

    def __nonzero__(self):
        for _ in self:
            return True
        return False

    def _new(self, items):
        return self.__class__(items)

    def _getitem(self, index):
        if isinstance(index, Integral):
            if index < 0:
                return list(self)[index]
            try:
                return next(islice(self, index, index+1))
            except StopIteration:
                raise LookupError(index)
        elif isinstance(index, slice):
            if any(i < 0 for i in (index.start, index.stop, index.step) if i is not None):
                # can't use islice when indices are negative
                items = list(self)
                return self._new(items[slice(index.start, index.stop, index.step)])
            try:
                return self._new(islice(self, index.start, index.stop, index.step))
            except StopIteration:
                raise LookupError(index)
        else:
            return self.get(index)

    def __getitem__(self, index):
        return self._getitem(index)

    def __repr__(self):
        if not self.name:
            items = all_items = list(self)
            if len(items) > 3:
                items = items[:2] + [Repr("...")] + items[-1:]
            return "{0.__class__.__name__}({items}, size={size})".format(self, items=items, size=len(all_items))
        return "<{0.__class__.__name__} '{0.name}'>".format(self)

    def __str__(self):
        items = ", ".join(map(str, self))
        if not self.name:
            return "<{}>".format(items)
        return "<'{}': {}>".format(self.name, items)

    def _repr_pretty_(self, p, cycle):
        from easypy.colors import DARK_CYAN
        # used by IPython
        class_name = self.__class__.__name__
        if cycle:
            p.text('%s(...)' % DARK_CYAN(class_name))
            return
        prefix = '%s(' % DARK_CYAN(class_name)
        with p.group(len(class_name) + 1, prefix, ')'):
            for idx, v in enumerate(self):
                if idx:
                    p.text(',')
                    p.breakable()
                p.pretty(v)

    def get_by_key(self, key):
        return self.get(key=key)

    def get_by_uid(self, uid):
        return self.get_by_key(uid)

    def safe_get_by_key(self, key):
        try:
            return self.get_by_key(key)
        except ObjectNotFound:
            return None

    def safe_get_by_uid(self, uid):
        return self.safe_get_by_key(uid)

    def get_by_keys(self, keys):
        gen = map(self.get_by_key, keys)
        try:
            return type(keys)(gen)  # return something of the same type as keys
        except TypeError:  # keys is generator type - return a generator
            return gen

    def iter_filtered(self, *preds, **filters):
        _shuffle = filters.pop("_shuffle", False)
        if _shuffle:
            objects = list(self)
            random.shuffle(objects)
        else:
            objects = self
        return filtered(objects, preds, filters)

    def select(self, *preds, **filters):
        return ListCollection(self.iter_filtered(*preds, **filters))

    def safe_get(self, *preds, **filters):
        # this python fu avoids filtering the entire list, while still checking for one item
        matching = [obj for obj, _ in zip(self.iter_filtered(*preds, **filters), range(2))]
        if len(matching) > 1:
            raise TooManyObjectsFound(self, preds, filters, _found=len(matching))
        return matching[0] if matching else None

    def get(self, *preds, **filters):
        # this python fu avoids filtering the entire list, while still checking for one item
        matching = [obj for obj, _ in zip(self.iter_filtered(*preds, **filters), range(2))]
        if len(matching) == 0:
            raise ObjectNotFound(self, preds, filters)
        if len(matching) != 1:
            raise TooManyObjectsFound(self, preds, filters, _found=len(matching))
        return matching[0]

    def choose(self, *preds, **filters):
        for obj in self.iter_filtered(_shuffle=True, *preds, **filters):
            return obj
        raise ObjectNotFound(self, preds, filters)

    def safe_choose(self, *preds, **filters):
        try:
            return self.choose(*preds, **filters)
        except ObjectNotFound:
            return None

    def menu(self, multiselect=False):
        """Open termenu to interactively select item(s) from the collection"""
        from termenu import show_menu
        return show_menu(self.name, list(self), multiselect=multiselect)

    def filtered(self, *preds, **filters):
        return FilterCollection(self, preds, filters)

    def shuffled(self):
        return self.sample(len(self))

    def sorted(self, key=None):
        return self._new(sorted(self, key=key))

    def sample(self, num, *preds, **filters):
        """
        If 'num' is negative, leaves 'num' out of the sample:

            >>> collection = ListCollection([1, 2, 3])
            >>> sample = collection.sample(-1)
            >>> len(sample)
            2
            >>> all(i in collection for i in sample)
            True
        """
        if num < 0:
            num += len(self)

        if isinstance(num, float):
            num, d = divmod(num, 1)
            assert d == 0, "Can't sample a non-integer number of items"
            num = int(num)

        if not num:
            return self._new([])

        uniquifiers = []
        for k, v in list(filters.items()):
            if v is UNIQUE:
                filters.pop(k)
                uniquifiers.append(k)

        if uniquifiers:
            if (preds or filters):
                matching = self.iter_filtered(_shuffle=True, *preds, **filters)
            else:
                matching = self

            matching = uniquify(matching, num, uniquifiers)

        else:
            # use 'zip' as a limiter
            matching = [obj for _, obj in zip(range(num), self.iter_filtered(_shuffle=True, *preds, **filters))]

        if len(matching) < num:
            raise NotEnoughObjects(self, preds, filters, needed=num)

        return self._new(matching)

    def _choose_sampling_size(self, minimum=0, maximum=None):
        if minimum < 0:
            minimum += len(self)
        if maximum is None:
            maximum = len(self)
        elif maximum < 0:
            maximum += len(self)
        maximum = min(maximum, len(self))
        return random.randint(minimum, maximum)

    def sample_some(self, minimum=0, maximum=None):
        sample_size = self._choose_sampling_size(minimum, maximum)
        return self.sample(sample_size)

    def without(self, *items_to_exclude):
        def filter(item):
            return item not in items_to_exclude
        return self.filtered(filter)

    @property
    def L(self):
        return list(self)

    @property
    def M(self):
        from .concurrency import MultiObject
        return MultiObject(self)

    def __add__(self, collection):
        return AggregateCollection([self, collection])


class AggregateCollection(ObjectCollectionBase):
    """
    Dynamically aggregate other collections into one chained collection
    """

    def __init__(self, collections, name=None):
        super().__init__()
        self._collections = collections
        self.name = name or "+".join((c.name or 'unnamed') for c in collections)

    def __iter__(self):
        return chain(*self._collections)

    def __add__(self, collection):
        return AggregateCollection(self._collections + [collection])

    def _new(self, items):
        return ListCollection(items)


class ListCollection(list, ObjectCollectionBase):
    """
    A simple list-based implementation of the ObjectCollection protocol
    """
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name', None)
        super().__init__(*args, **kwargs)

    def pop_some(self, minimum=0, maximum=None):
        sample_size = self._choose_sampling_size(minimum, maximum)
        return self._new([self.pop() for i in range(sample_size)])

    def __getitem__(self, index):
        return self._getitem(index)

    def __repr__(self):
        if not self.name:
            return ObjectCollectionBase.__repr__(self)
        return "<'{0.name}', size={size}>".format(self, size=len(self))


class SimpleObjectCollection(ObjectCollectionBase):
    """
    An ObjectCollection with fast lookup based on an ID attribute found on the objects.
    """

    ID_ATTRIBUTE = 'uid'

    class Collectable():
        """
        Objects can inherit this mixin class and will then get a ``.collection`` attribute
        that points back to the collection they were added to
        """
        collection = None

    def __init__(self, objs=(), ID_ATTRIBUTE=None, backref=False, name=None):
        super().__init__(name=name)
        if ID_ATTRIBUTE:
            self.ID_ATTRIBUTE = ID_ATTRIBUTE
        self._objects = PythonOrderedDict()
        for obj in objs:
            self.add(obj, backref=backref)

    def __repr__(self):
        if not self.name:
            return ObjectCollectionBase.__repr__(self)
        return "<'{0.name}', size={size}>".format(self, size=len(self))

    def _new(self, items):
        return self.__class__(items, self.ID_ATTRIBUTE)

    def _get_object_uid(self, obj):
        return getattr(obj, self.ID_ATTRIBUTE)

    def _add_and_get_uid(self, obj):
        uid = self._get_object_uid(obj)
        self._objects[uid] = obj
        return uid, obj

    def add(self, obj, backref=False):
        if backref:
            assert isinstance(obj, self.Collectable), "Can add backref only to Collectable classes"
            assert obj.collection is None or obj.collection is self, \
                "Collectable object not allowed to be added to multiple collections"
            obj.collection = self
        return self._add_and_get_uid(obj)[1]

    def _remove_and_get_uid(self, obj):
        uid = self._get_object_uid(obj)
        return uid, self._objects.pop(uid)

    def remove(self, obj):
        return self._remove_and_get_uid(obj)[1]

    def clear(self):
        self._objects.clear()

    def remove_by_uid(self, uid):
        obj = self._objects.pop(uid)
        return obj

    def get(self, *preds, **filters):
        if len(preds) == 1 and not filters and not callable(preds[0]):
            key, = preds
            return self.get_by_key(key)
        return super().get(*preds, **filters)

    def get_by_key(self, key):
        if key not in self._objects:
            raise ObjectNotFound(self, (), dict(key=key),)
        return self._objects[key]

    def index(self, obj):
        lookup_uid = self._get_object_uid(obj)
        if lookup_uid not in self._objects:
            raise ValueError("%r not in %r" % (obj, self))
        for i, uid in enumerate(self._objects.keys()):
            if lookup_uid == uid:
                return i

    if SUPPORT_GET_SIBLING:
        def get_next(self, obj):
            uid = self._get_object_uid(obj)
            if uid not in self._objects:
                raise ObjectNotFound(self, (), dict(key=uid),)
            next_uid = self._objects._OrderedDict__map[uid].next
            try:
                key = next_uid.key
            except AttributeError:
                key = next_uid.next.key
            return self._objects[key]

        def get_prev(self, obj):
            uid = self._get_object_uid(obj)
            if uid not in self._objects:
                raise ObjectNotFound(self, (), dict(key=uid),)
            prev_uid = self._objects._OrderedDict__map[uid].prev
            try:
                key = prev_uid.key
            except AttributeError:
                key = prev_uid.prev.key
            return self._objects[key]
    else:
        def get_next(self, obj):
            from warning import warn
            warn("Couldn't import a pure-python OrderedDict so this will be a O(n) operation")

            from itertools import cycle
            l1, l2 = cycle(self._objects.values()), cycle(self._objects.values())
            next(l2)
            for a, b in zip(l1, l2):
                if obj == a:
                    return b

        def get_prev(self, obj):
            from warning import warn
            warn("Couldn't import a pure-python OrderedDict so this will be a O(n) operation")

            from itertools import cycle
            l1, l2 = cycle(self._objects.values()), cycle(self._objects.values())
            next(l2)
            for a, b in zip(l1, l2):
                if obj == b:
                    return a

    def keys(self):
        return self._objects.keys()

    def __iter__(self):
        return iter(self._objects.values())

    def __len__(self):
        return len(self._objects)

    @property
    def M(self):
        from .concurrency import MultiObject
        return MultiObject(self, list(self._objects.keys()))


class FilterCollection(ObjectCollectionBase):
    """
    A dynamic collection that filters another ``base`` collection according to the
    given predicates and filters.
    """

    def __init__(self, base, preds, filters, parent=None, name=None):
        super().__init__(name=name)
        self.base = base
        self.parent = parent if parent is not None else base
        self.preds = preds
        self.filters = filters

    def _new(self, items):
        if hasattr(self.base, 'ID_ATTRIBUTE'):
            return self.base.__class__(items, ID_ATTRIBUTE=self.base.ID_ATTRIBUTE)
        return self.base.__class__(items)

    def __repr__(self):
        if self.name and self.base.name:
            return "<'{self.name}', based on '{self.base.name}'>".format(**locals())
        elif self.name:
            return "<'{self.name}', based on a {self.base.__class__.__name__}>".format(**locals())
        elif not self.name:
            filters_str = _format_filter_string(self.preds, self.filters)
            if self.base.name:
                return "<Filtered collection based on '{self.base.name}' ({filters_str})>".format(**locals())
            else:
                return "<Filtered collection based on an anonymous '{self.base.__class__.__name__}' ({filters_str})>".format(**locals())
        else:
            return ObjectCollectionBase.__repr__(self)

    def __iter__(self):
        return self.iter_filtered()

    def iter_filtered(self, *preds, **filters):
        _shuffle = filters.pop("_shuffle", False)
        combined_preds = list(self.preds)
        combined_preds += preds
        for k, v in self.filters.items():
            if k in filters:
                assert v == filters[k], "Two different values provided for the same filter key!"
        filters.update(self.filters)
        return self.parent.iter_filtered(_shuffle=_shuffle, *combined_preds, **filters)

    def filtered(self, *preds, **filters):
        return FilterCollection(self.base, preds, filters, parent=self)

    def get(self, *preds, **filters):
        if len(preds) == 1 and not filters and not callable(preds[0]):
            key, = preds
            return self.get_by_key(key)
        else:
            return super().get(*preds, **filters)

    def get_by_key(self, key):
        try:
            obj = self.base.get_by_key(key)
        except ObjectNotFound:
            raise ObjectNotFound(self, key=key) from None
        for obj in filtered([obj], self.preds, self.filters):
            return obj
        raise ObjectNotFound(self, key=key)

    def get_next(self, obj):
        while True:
            next_obj = self.base.get_next(obj)
            if any(filtered([next_obj], self.preds, self.filters)):
                return next_obj
            obj = next_obj

    def __getitem__(self, uid):
        if not isinstance(uid, str):
            return super().__getitem__(uid)

        try:
            obj = self.base[uid]
        except ObjectNotFound:
            raise ObjectNotFound(self, key=uid) from None

        for obj in filtered([obj], self.preds, self.filters):
            return obj

        raise ObjectNotFound(self, key=uid)


def TypeFilterCollection(base, type):
    """
    A collection that filters another collection according to the specified type (class)
    """
    return base.filtered(lambda obj: isinstance(obj, type))


class IteratorBasedCollection(ObjectCollectionBase):
    """
    An iterator-based collection. Each time the collection is read, the iterator function is called again,
    to provide a fresh iterator object.
    """

    def __init__(self, iterator_func):
        super().__init__()
        self._iterator_func = iterator_func

    def __iter__(self):
        return self._iterator_func()


class IndexedObjectCollection(SimpleObjectCollection):
    """
    Work-in-progress

    An indexed collection, allowing fast lookup of object by using multiple key indices.
    """

    def __init__(self, objs=(), keys=(), **kwargs):
        self._indices = {key: collections.defaultdict(set) for key in keys}
        super().__init__(objs, **kwargs)

    def _new(self, items):
        return self.__class__(items, keys=self._indices.keys(), ID_ATTRIBUTE=self.ID_ATTRIBUTE)

    def add(self, obj):
        uid, obj = self._add_and_get_uid(obj)
        for key, uid_map in self._indices.items():
            value = getattr(obj, key)
            uid_map[value].add(uid)

    def remove(self, obj):
        uid, obj = self._remove_and_get_uid(obj)
        for uid_map in self._indices.values():
            for uids in uid_map.values():
                uids.discard(uid)
        return obj

    def clear(self):
        self._objects.clear()
        for idx in self._indices.values():
            idx.clear()

    def remove_by_uid(self, uid):
        return self.remove(self._objects[uid])

    def get(self, *preds, **filters):
        if len(preds) == 1 and not filters and not callable(preds[0]):
            key, = preds
            return self.get_by_key(key)
        return super().get(*preds, **filters)

    def iter_filtered(self, *preds, **filters):
        _shuffle = filters.pop("_shuffle", False)
        indexed = set(filters) & set(self._indices)
        unindexed = set(filters) - indexed
        if indexed:
            filtered_keys = set.intersection(*(self._indices.get(key, {}).get(filters[key], set()) for key in indexed))
            objects = (self.get_by_key(key) for key in filtered_keys)
        else:
            objects = self
        if _shuffle:
            objects = list(objects)
            random.shuffle(objects)
        if not (preds or unindexed):
            return objects
        filters = {k: filters[k] for k in unindexed}
        return filtered(objects, preds, filters)


def grouped(sequence, key=None, transform=None):
    """
    Parse the sequence into groups, according to key:

        >>> ret = grouped(range(10), lambda n: n % 3)
        >>> ret[0]
        [0, 3, 6, 9]
        >>> ret[1]
        [1, 4, 7]
        >>> ret[2]
        [2, 5, 8]
    """
    groups = {}
    if not key:
        key = lambda x: x
    if not transform:
        transform = lambda x: x
    for item in sequence:
        groups.setdefault(key(item), []).append(transform(item))
    return groups


def separate(sequence, key=None):
    """
    Partition the sequence into items that match and items that don't:

        >>> above_three, three_or_less = separate(range(10), lambda n: n > 3)

        >>> print(above_three)
        [4, 5, 6, 7, 8, 9]

        >>> print(three_or_less)
        [0, 1, 2, 3]
    """
    if not key:
        key = lambda x: x
    groups = grouped(sequence, key=lambda e: bool(key(e)))
    return groups.get(True, []), groups.get(False, [])


def iterable(obj):
    return isinstance(obj, collections.Iterable) and not isinstance(obj, (str, bytes, dict))


def ilistify(obj):
    """
    Normalize input into an iterable:

        >>> list(ilistify(1))
        [1]
        >>> list(ilistify([1, 2, 3]))
        [1, 2, 3]
    """
    if not iterable(obj):
        yield obj
    else:
        yield from obj


def listify(obj):
    """
    Normalize input into a list:
        >>> listify(1)
        [1]
        >>> listify([1, 2])
        [1, 2]
    """
    return list(ilistify(obj))


def chunkify(sequence, size):
    """
    Chunk a sequence into equal-size chunks (except for the last chunk):

        >>> ["".join(p) for p in chunkify('abcdefghijklmnopqrstuvwxyz', 7)]
        ['abcdefg', 'hijklmn', 'opqrstu', 'vwxyz']
    """
    sequence = iter(sequence)
    while True:
        chunk = [e for _, e in zip(range(size), sequence)]
        if not chunk:
            return
        yield chunk


def partial_dict(d, keys):
    """
    Returns a new dict with a subset of the original items.

    >>> d = partial_dict({'a': 1, 'b': 2, 'c': 3}, ['a', 'b'])
    >>> d == {'a': 1, 'b': 2}
    True
    """
    return {k: d[k] for k in keys}


def intersected_dict(d, keys):
    """
    Returns a new dict with a subset of the original items.
    Items that are specified in keys but not in the original dict are discarded

    >>> d = intersected_dict({'a': 1, 'b': 2, 'c': 3}, ['a', 'b', 'z'])
    >>> d == {'a': 1, 'b': 2}
    True
    """
    return {k: d[k] for k in keys if k in d}


def dicts_to_table(dicts, index_header="idx"):
    """
    Convert a sequence of homogenuous dicts into a table (2D list),
    with the dict keys present in the first row:

    >>> dict_list = [
    ...     dict(a=1, b=2, c=3),
    ...     dict(a=7, b=8, c=9)]
    >>> for row in dicts_to_table(dict_list, index_header='R'):
    ...     print(*row, sep="|")
    R|a|b|c
    0|1|2|3
    1|7|8|9

    >>> dicts_dict = dict(
    ...     X=dict(a=1, b=2, c=3),
    ...     Y=dict(a=7, b=8, c=9))
    >>> for row in dicts_to_table(dicts_dict, index_header='R'):
    ...     print(*row, sep="|")
    R|a|b|c
    X|1|2|3
    Y|7|8|9

    """

    if isinstance(dicts, dict):
        _, sample_row = next(iter(dicts.items()))
        rows = ((key, dicts[key]) for key in sorted(dicts))
    else:
        sample_row = dicts[0]
        rows = enumerate(dicts)
    headers = sorted(sample_row.keys())
    table = [[index_header] + headers]
    for key, row in rows:
        table.append([key] + [row[header] for header in headers])
    return table


def shuffled(items):
    """
    Return a shuffled version of the input sequence
    """
    items = list(items)
    random.shuffle(items)
    return items


class SlidingWindow(list):
    """
    A list that maintains a constant size, popping old items as new items are appended (FIFO)
    """
    def __init__(self, *args, size):
        self.size = size
        super().__init__(*args)

    def append(self, item):
        super().append(item)
        if len(self) > self.size:
            self.pop(0)


@parametrizeable_decorator
def as_list(generator, sort_by=None):
    """
    Forces a generator to output a list.

    For when writing a generator is more elegant::

    >>> @as_list(sort_by=lambda n: -n)
    ... def g():
    ...     yield 2
    ...     yield 1
    ...     yield from range(2)
    >>> g()
    [2, 1, 1, 0]

    """

    def inner(*args, **kwargs):
        l = list(generator(*args, **kwargs))
        if sort_by:
            l.sort(key=sort_by)
        return l
    return inner


def takesome(generator, max=None, min=0):
    """
    Take between ``min`` and ``max`` items from the generator.
    Throws a ValueError if the generator didn't yield ``min`` items.

    >>> list(takesome("abcdef", 3))
    ['a', 'b', 'c']

    >>> list(takesome("abcdef", 10))
    ['a', 'b', 'c', 'd', 'e', 'f']

    >>> list(takesome("abcdef", 0))
    []

    >>> list(takesome("abcdef", min=8))
    Traceback (most recent call last):
        ...
    ValueError: Not enough items in sequence (wanted 8, got 6)
    """

    if max is not None and max < min:
        raise ValueError("'max' must be great than 'min'")

    items = [p for p, _ in zip(generator, range(min))]

    if len(items) < min:
        raise ValueError("Not enough items in sequence (wanted {}, got {})".format(min, len(items)))

    yield from items

    if max is None:
        yield from generator
        return

    remain = max - min
    for p, _ in zip(generator, range(remain)):
        yield p


if __name__ == "__main__":
    import doctest
    doctest.testmod()
