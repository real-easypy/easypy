from __future__ import absolute_import
import sys
import os
import json
import socket
import codecs
from contextlib import contextmanager

from .py5 import TimeoutError, PY3

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
try:
    import http.client
    httpclient = http.client
    from http.client import IncompleteRead
except ImportError:
    import httplib
    httpclient = httplib
    from httplib import IncompleteRead

import time
from itertools import count, chain
from functools import wraps
from threading import RLock

from easypy.py5 import raise_with_traceback
from easypy.exceptions import PException, TException
from easypy.bunch import bunchify, Bunch
from easypy.aliasing import super_dir
from easypy.resilience import retrying, raise_if_async_exception
from easypy.timing import Timer, wait, timing
from easypy.units import DataSize


from logging import getLogger
_logger = getLogger(__name__)


class JsonRpcException(TException): pass


class ReadError(JsonRpcException):
    template = "Error while reading from pipe"


class ConnectionError(JsonRpcException):
    template = "Error connecting to JRPC server: {exc}"


class RequestError(ConnectionError):
    template = "Error sending JRPC request to server: {exc}"


class ReplyError(ConnectionError):
    template = "Error getting JRPC reply from server: {exc}"


class ServerTimeout(ReplyError):
    template = "Timeout connecting to server: {url}"


class ResponseError(JsonRpcException):
    template = "Error in JRPC response: {}"


class RemoteException(ResponseError):
    template = 'Exception raised remotely\n{}'

    def __init__(self, remote_message, *args, **params):
        super(RemoteException, self).__init__(remote_message, *args, **params)
        self.remote_message = remote_message


class RemoteMethodNotFound(ResponseError):
    template = 'Remote method {method} not found(maybe the node is not ready yet?)'


class LongOperationTimeout(ResponseError):
    template = "Timeout while waiting for {} to complete"


class ResponseIdMismatch(JsonRpcException):
    template = "Request and Response ids are out of sync"


class HTTPException(JsonRpcException):
    template = "HTTP request failed executing JRPC"


if PY3:
    import builtins
    JRPC_CONNECTION_ERRORS = (
        socket.timeout,
        builtins.ConnectionError,
        ConnectionError,
        TimeoutError,
    )
else:
    JRPC_CONNECTION_ERRORS = (
        socket.error,
        ConnectionError,
        TimeoutError,
    )

def _make_request(message_id, method, params):
    result = dict(
        jsonrpc='2.0',
        method=method,
        params=params or {},
        id=message_id
    )
    return json.dumps(result, allow_nan=False)


def _args_generator(args_specs):
    for arg in args_specs:
        if 'default' in arg:
            default_value = arg.default
            if isinstance(default_value, Bunch):
                default_value = default_value.to_dict()
            yield '%s=%s' % (arg.name, repr(default_value))
        else:
            yield arg.name


def locked(lock):
    def deco(func):
        @wraps(func)
        def inner(*args, **kw):
            with lock:
                return func(*args, **kw)
        return inner
    return deco


def _add_function(self, name, spec, *defaults):
    arg_names = [arg.name for arg in spec.args]
    args = ", ".join(chain(_args_generator(spec.args), defaults))
    params = ", ".join(arg_names + ["{0}={0}".format(n.split("=", 1)[0]) for n in defaults])
    func_def = ("""def {name}({args}): return _self_.rpc('{name}', {params})""").format(name=name, args=args, params=params)
    localns = dict(_self_=self)
    eval(compile(func_def, "<string>", "exec"), localns, localns)
    func = localns[name]
    func.__doc__ = spec.doc or "(No docstring)"
    setattr(self, name, func)
    return func


@contextmanager
def _noop_ctx(*_, **__):
    yield


def _logger_context(*args, **kwargs):  # weka-ui codebase
    return getattr(_logger, "context", _noop_ctx)(*args, **kwargs)


def _logger_contexted(*ctx_args, **ctx_kwargs):  # compatibility with python2
    def deco(func):
        @wraps(func)
        def inner(*args, **kwargs):
            with _logger_context(*ctx_args, **ctx_kwargs):
                return func(*args, **kwargs)
        return inner
    return deco


def _host_context(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        with _logger_context(host=self.name):
            return func(self, *args, **kwargs)
    return inner



class RetrySignal(PException): pass

_id_counter = count(int(time.time()*1000000) & 0xffffffffff)


class JsonRpcClient():
    DEFAULT_CREDENTIALS = "smeagol:bagginses"

    def __init__(self, host, port, path="/api", name=None, timeout=None, ttl=120, init_methods=True, client_type=None,
                 credentials=DEFAULT_CREDENTIALS, rpc_retries=2):
        self._rpc_retries = rpc_retries
        self._conn_params = bunchify(host=host, port=port, timeout=timeout or 120)
        self._path = path
        self._ttl = ttl
        self._conn = None
        self._headers = {}
        if credentials:
            b64 = codecs.encode(credentials.encode("latin1"), "base64").replace(b"\n", b"")
            self._headers["authorization"] = b"Basic " + b64
        if client_type:
            self._headers['Client-Type'] = client_type
        self._spec = None
        self._exception_handlers = {}
        self._lock = RLock()
        self.long_operations = self._LongOperations(self)
        self.name = name
        if init_methods:
            self._populate_methods()

    @property
    def url(self):
        return '%(host)s:%(port)s%(path)s' % dict(self._conn_params, path=self._path)

    def __repr__(self):
        return "%s(<%s>)" % (self.__class__.__name__, self.name)

    def __dir__(self):
        if not self._spec:
            self._populate_methods()
        return super_dir(self)

    @_host_context
    def _populate_methods(self):
        try:
            self.methods = None
            self._spec = self.rpc("getServiceSpec", jrpc_timeout=5)
        except JsonRpcException as e:
            _logger.debug("Could not populate methods (%s)", e)
            return
        for name, spec in self._spec.items():
            _add_function(self, name, spec, "jrpc_timeout=None", "jrpc_quiet=False")
            if spec.get('returnStructure', {}).get('type', {}) == 'LongOperation':
                self.long_operations._register(name, spec)
        self.methods = [str(method) for method in self._spec]

    class _LongOperations:
        def __init__(self, client):
            self.client = client

        def rpc(self, method, *params_list, **params_dict):
            long_operation_timeout_override = params_dict.pop("long_operation_timeout") or 60

            def get_results():
                rpc_result = self.client.rpc(method, *params_list, **params_dict)
                if rpc_result.completed:
                    # rpc_result.result is allowed to be False or None, so we
                    # return the entire rpc_result
                    return rpc_result
                wait(rpc_result.resend_secs)

            try:
                rpc_result = wait(long_operation_timeout_override, get_results, sleep=0)
            except TimeoutError:
                raise LongOperationTimeout(method)

            assert rpc_result.completed
            return rpc_result.result

        def __getattr__(self, attr):
            if attr.startswith("_") or attr == "trait_names":
                raise AttributeError(attr)

            def func(*args, **kwargs):
                return self.rpc(attr, *args, **kwargs)
            func.__name__ = attr
            return func

        def _register(self, name, spec):
            _add_function(self, name, spec, "jrpc_timeout=None", "jrpc_quiet=False", "long_operation_timeout=None")

    def register_exception_handler(self, name, handler):
        assert name not in self._exception_handlers
        self._exception_handlers[name] = handler

    def expire_connection(self):
        if self._conn:
            self._conn.close()
        self._conn = None

    @_host_context
    def get_connection(self, timeout_override=None):
        if timeout_override:
            kw = dict(self._conn_params, timeout=timeout_override)
            _logger.debug("creating an ad-hoc connection (timeout=%s)", timeout_override)
            return httpclient.HTTPConnection(**kw)

        conn = self._conn

        if not conn:
            _logger.debug("creating a new connection (timeout=%s, ttl=%s)", self._conn_params.timeout, self._ttl)
        elif conn.timer.expired:
            _logger.debug("connection expired (%ss ago) - resetting", conn.timer.expired)
            conn = None
        else:
            conn.timer.reset()

        if not conn:
            conn = httpclient.HTTPConnection(**self._conn_params)
            if not self._ttl:
                return conn
            conn.timer = Timer(expiration=self._ttl)
        self._conn = conn
        return conn

    def retry_last_rpc(self, msg="Exception handler signaled to retry last JsonRPC"):
        raise RetrySignal(msg)

    @_host_context
    def rpc(self, method, *params_list, **params_dict):
        timeout_override = params_dict.pop("jrpc_timeout", None)
        quiet = params_dict.pop("jrpc_quiet", False)
        should_bunchify = params_dict.pop("bunchify", True)

        if params_list and params_dict:
            raise ValueError('RPC can not be invoked with both positional and named arguments')
        elif params_list:
            params = params_list
        elif params_dict:
            params = params_dict
        else:
            params = None

        exc_params = dict(url=self.url, method=method, params=params)
        if self.name:
            exc_params['name'] = self.name

        message_id = next(_id_counter)
        exc_params.update(message_id=message_id)

        @_logger_contexted("#%s" % message_id)
        @retrying(times=self._rpc_retries , acceptable=(RetrySignal, ConnectionError), sleep=0.5)
        @locked(self._lock)
        def make_request():
            if not quiet:
                _logger.debug("request  >> #%04d: %s/%s", message_id, self.url, method)
            try:
                try:
                    conn = self.get_connection(timeout_override=timeout_override)
                except:
                    typ, exc, tb = sys.exc_info()
                    raise_if_async_exception(exc)
                    raise_with_traceback(ConnectionError(url=self.url, message_id=message_id, exc=exc), tb)

                json_req = _make_request(message_id, method, params)

                try:
                    conn.request('POST', self._path, json_req, self._headers)
                except socket.timeout:
                    typ, exc, tb = sys.exc_info()
                    raise_if_async_exception(exc)
                    raise_with_traceback(ServerTimeout(**exc_params), tb)
                except:
                    typ, exc, tb = sys.exc_info()
                    raise_if_async_exception(exc)
                    raise_with_traceback(RequestError(exc=exc, **exc_params), tb)

                try:
                    response = conn.getresponse()
                except socket.timeout:
                    typ, exc, tb = sys.exc_info()
                    raise_if_async_exception(exc)
                    raise_with_traceback(ServerTimeout(**exc_params), tb)
                except:
                    typ, exc, tb = sys.exc_info()
                    raise_if_async_exception(exc)
                    # To address leader temporarily being down ("BadStatusLine")
                    time.sleep(1)
                    raise_with_traceback(ReplyError(exc=exc, **exc_params), tb)
            except:
                self._conn = None  # create a new connection next time
                raise

            if response.status == 408 or response.status == 400:
                # AWS issues
                _logger.debug("Got error code %s, on request #%04d, retrying", response.status, message_id)
                self._conn = None  # create a new connection next time
                time.sleep(0.5)
                self.retry_last_rpc('Got error code %s - %s' % (response.status, httpclient.responses[response.status]))

            if response.status == 301:
                # redirect
                new_url = response.getheader("location")
                parsed = urlparse(new_url)
                _logger.debug("redirecting http://%s:%s/%s to %s", self._conn_params.host, self._conn_params.port,
                              self._path, new_url)
                #   print("redirect to", new_url)

                orig = Bunch(**self._conn_params)
                if ":" in parsed.netloc:
                    host, port = parsed.netloc.split(":")
                    self._conn_params.host = host
                    self._conn_params.port = int(port)
                else:
                    self._conn_params.host = parsed.netloc

                self.expire_connection()
                conn = self.get_connection(timeout_override=timeout_override)
                conn.request('POST', self._path, _make_request(message_id, method, params), self._headers)
                response = conn.getresponse()
                self.expire_connection()
                self._conn_params = orig

            if response.status == 503:
                # service not available, retry again later
                time.sleep(0.5)
                return self.retry_last_rpc("Service unavailable, retry later")

            try:
                response_text = response.read().decode('utf-8')
            except socket.timeout:
                self._conn = None  # create a new connection next time
                typ, exc, tb = sys.exc_info()
                raise_with_traceback(ServerTimeout(**exc_params), tb)
            except IncompleteRead:
                self._conn = None  # create a new connection next time
                typ, exc, tb = sys.exc_info()
                raise_with_traceback(ReadError(exc=exc, **exc_params), tb)

            if response.status != 200:
                # some other error
                raise HTTPException(status=response.status, reason=response.reason, text=response_text, **exc_params)

            try:
                response_object = json.loads(response_text)
            except ValueError:
                raise ResponseError("Could not parse json respone", response_text=response_text, **exc_params)

            if not quiet:
                _logger.debug("response << #%04d: %s -> %s", response_object.get('id', -1), method,
                              DataSize(response.getheader("content-length")))

            if message_id != response_object.get('id'):
                raise ResponseIdMismatch(responded=response_object.get('id'), response_text=response_text, **exc_params)

            if 'error' not in response_object:
                result = response_object['result']
                return bunchify(result) if should_bunchify else result

            error_code = response_object['error'].pop('code', None)
            error_data = response_object['error'].pop('data', "(no further information)")
            remote_message = response_object['error'].pop('message', "(no message)")
            server_node_id = response.getheader('server-node-id', None)

            if error_code == -32601:  # Method not Found
                raise RemoteMethodNotFound(method=method)
                # raise AttributeError('%s' % remote_message)

            if error_code == -32602:  # Invalid Params
                raise TypeError('%s(%s)' % (remote_message, error_data))

            if isinstance(error_data, dict):
                if server_node_id:
                    error_data['served_by_node'] = server_node_id

                if 'exceptionClass' in error_data:
                    exception_class = error_data.pop('exceptionClass')
                    if isinstance(exception_class, list):
                        exceptions._register_ancestry(exception_class)
                        exception_class = exception_class[0]
                    exception_name = exception_class.rpartition(".")[-1]
                    handler = self._exception_handlers.get(exception_name) or getattr(exceptions, exception_name)
                    if hasattr(handler, '__bases__') and error_data.get("retry", False) and RetrySignal not in handler.__bases__:
                        handler.__bases__ += (RetrySignal,)
                    exception_text = error_data.pop('exceptionText', remote_message)
                    error_data['jrpc'] = exc_params
                    raise handler(exception_text, **error_data)
                else:
                    exc_params.update(error_data)
                    raise RemoteException(remote_message, **exc_params)
            else:
                if error_data:
                    remote_message += "\n%s" % (error_data,)
                exc_params.update(response_object['error'])

                if server_node_id:
                    exc_params['served_by_node'] = server_node_id

                raise RemoteException(remote_message, **exc_params)

        return make_request()

    def __getattr__(self, attr):
        if attr.startswith("_") or attr == "trait_names":
            raise AttributeError(attr)

        def func(*args, **kwargs):
            return self.rpc(attr, *args, **kwargs)
        func.__name__ = attr
        return func

    def close_connection(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _extract_args(self, rpc, args, kwargs):
        result = Bunch(**kwargs)
        if self._spec is None:
            self._populate_methods()
        spec = self._spec[rpc]
        arg_names = (arg.name for arg in spec.args)
        for arg, arg_name in zip(args, arg_names):
            result[arg_name] = arg
        return result


# ===================================================================================================
# Module hack: ``from jrpc.exceptions import ErrnoException``
# ===================================================================================================
from types import ModuleType


class ExceptionsModule(ModuleType):
    """The module-hack that allows us to use ``from jrpc.exceptions import SomeException``"""
    __all__ = ()  # to make help() happy
    __package__ = __name__

    def _register_ancestry(self, ancestry):
        expected_parent = RemoteException
        for exception_class_name in reversed(ancestry):
            exception_class = getattr(self, exception_class_name)
            # we need to fix existing classes, that have been created by
            # jrpc clients before the server had a chance of creating them
            if exception_class.__base__ is not expected_parent:
                if exception_class.__base__ is not RemoteException:
                    _logger.warning("JRPC exception '%s' changed its parent: %s -> %s",
                                    exception_class_name, exception_class.__base__.__name__, expected_parent.__name__)
                exception_class.__bases__ = (expected_parent,)
            expected_parent = exception_class

    def __getattr__(self, attr):
        if attr.startswith("_") or attr == 'trait_names':
            raise AttributeError(attr)
        exc = type(attr, (RemoteException, ), {})
        exc.__module__ = exceptions.__name__
        setattr(self, attr, exc)
        return exc

    __path__ = []
    __file__ = __file__

exceptions = ExceptionsModule(__name__ + ".exceptions", ExceptionsModule.__doc__)
sys.modules[exceptions.__name__] = exceptions

del ModuleType
del ExceptionsModule


if __name__ == "__main__":
    params = sys.argv[1:]
    if params:
        j = JsonRpcClient(*sys.argv[1:])
        from IPython import embed as _embed
        _embed(header="JSON RPC Shell:\n>> j = %s" % j)
