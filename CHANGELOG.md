# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- resilience: support returning exceptions captured by `resilient` decorator

### Changed
- Logging Initialization
    - a run-once `initialize` method takes care of it all
    - removed `set_width`, `set_coloring`, `set_graphics`
    - the env-var `TERM_COLOR_SUPPORT` is no longer checked
    - the env-var `EASYPY_LOG_INDENTATION` is no longer checked
- DataSize - return 'byte' and 'MiB' instead of '1 byte' and '1 MiB'

### Added
- Examples
    - a skeleton for initializing logging with easypy

### Added
- `ensure_same_defaults` decorator for setting one function's defaults as source of truth for other function

## [0.4.0] - 2019-11-14

### Changed
- units: DataSize - return 'byte' and 'MiB' instead of '1 byte' and '1 MiB'
- units: Duration(inf) is now 'Eternity' instead of 'Never'
- timing: The Timer class now renders the duration using the Duration's repr, instead of as a float
- aliasing: Improved `RecursionError` for aliasing-induced infinite recursion

### Added
- CLI: expose 'colorize' and 'ziplog' as clis
- units: Added a `Percentage` class
- timing: Added a `TimeInterval` class, for use for timestamp comparisons. Can be converted to/from the `Timer` class
- colors: Added '@[..]@' as coloring markup delimiter
- concurrency: Add parent's name to new thread's name, to show hierarchy
- concurrency: Add `done()` function to `concurrent` to check if the thread is finished.
- collections: Add `.menu()` to `ObjectCollectionBase`, integrating with `termenu`
- semver: Added a module and class for parsing and comparing semver-styled version strings
- random: Added 'perchance', for a common random use-case
- bunch: Added `.render()` to render the data cleanly
- exceptions: support pickling
- CI: Add python 3.7 and 3.8

### Fixed
- sync: prevent an import cycle
- colors/ziplog: swallow BrokenPipeError when parsing from cmdline
- aliasing: bug fix for 'Cannot create a consistent method resolution', add unittest
- collections: workaround python bug (3.6.8+) importing 'test.support'

## [0.3.1] - 2019-07-30

### Added
- ci: run tests in random order
- collections: added unittest

### Fixed
- sync: Don't release unacquired lock
- caching: Don't leak open db, as it is concurrent access which has undefined behavior
- concurrency
	- refactor to fix stalling of main-thread after Futures.executor breaks on exception
	- Update max number of threads
- logging: set default host in thread logging context
- resilience: match ``retry`` and ``retrying`` function signature


## [0.3.0] - 2019-06-10

### Added
- Support for suppressing and soloing logging to console per thread.
- `TypedStruct`: Support for inheritance.
- `EasyMeta`: the `before_subclass_init` hook.
- `wait` and `iter_wait` support `log_interval` and `log_level` for printing
  the thrown `PredicateNotSatisfied` to the log.
- `takesome`: a new generator that partially yields a sequence
- `repr` and `hash` to typed struct fields.
- `PersistentCache`: allow disabling persistence via env-var (`DISABLE_CACHING_PERSISTENCE`)
- collections: raising user-friendly exceptions for failed object queries (too many, too few)

### Fixed
- `ExponentialBackoff`: return the value **before** the incrementation.
- `concurrent`: capture `KeyboardInterrupt` exceptions like any other.
- doctests in various functions and classes.
- `SynchronizedSingleton` on `contextmanager` deadlock when some (but not all)
  of the CMs throw.
- `resilient` between `timecache`s bug.
- `FilterCollection`: deal with missing attributes on objects (#163)
- `PersistentCache`: don't clear old version when changing cache version
- concurrency: documentation
- `SynchronizedSingleton`: deadlock condition when used with `contextmanager` (#150)
- concurrency: make 'async' available only in python < 3.7, where it became reserved

### Changed
- Reorganization:
  - Moved tokens to a proper module.
  - Moved function from `easypy.concurrency` and `easypy.timing` to new module
    `easypy.sync`
  - Moved `throttled` from `easypy.concurrency` to `easypy.timing`.
- `easypy.signals`: Async handlers are invoked first, then the sequential handlers.
- `async` -> `asynchronous`: to support python 3.7, where this word is reserved
- `concurrency.concurrent`: `.result` property changed into method `.result()`, which also waits on the thread
- `easypy.colors`: clean-up, documentation, and reverse-parsing from ansi to markup

### Removed
- `Bunch`: The rigid `KEYS` feature.
- `synchronized_on_first_call`.
- `ExponentialBackoff`: The unused `iteration` argument.
- `easypy.cartesian`
- `easypy.selective_queue`
- `easypy.timezone`

### Deprecated
- `locking_lru_cache`.

## [0.2.0] - 2018-11-15
### Added
- Add the `easypy.aliasing` module.
- Add the `easypy.bunch` module.
- Add the `easypy.caching` module.
- Add the `easypy.cartesian` module.
- Add the `easypy.collections` module.
- Add the `easypy.colors` module.
- Add the `easypy.concurrency` module.
- Add the `easypy.contexts` module.
- Add the `easypy.decorations` module.
- Add the `easypy.exceptions` module.
- Add the `easypy.fixtures` module.
- Add the `easypy.gevent` module.
- Add the `easypy.humanize` module.
- Add the `easypy.interaction` module.
- Add the `easypy.lockstep` module.
- Add the `easypy.logging` module.
- Add the `easypy.meta` module.
- Add the `easypy.misc` module.
- Add the `easypy.mocking` module.
- Add the `easypy.predicates` module.
- Add the `easypy.properties` module.
- Add the `easypy.randutils` module.
- Add the `easypy.resilience` module.
- Add the `easypy.selective_queue` module.
- Add the `easypy.signals` module.
- Add the `easypy.tables` module.
- Add the `easypy.threadtree` module.
- Add the `easypy.timezone` module.
- Add the `easypy.timing` module.
- Add the `easypy.typed_struct` module.
- Add the `easypy.units` module.
- Add the `easypy.words` module.
- Add the `easypy.ziplog` module.
