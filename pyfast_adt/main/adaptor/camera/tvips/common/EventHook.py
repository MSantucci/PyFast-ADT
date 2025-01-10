"""
Stolen from 
https://stackoverflow.com/questions/1092531/event-system-in-python/35957226#35957226
"""
from typing import Callable, Type, List
import inspect

class EventHook:
    '''
    A simple implementation of the Observer-Pattern.
    The user can specify an event signature upon inizializazion,
    defined by kwargs in the form of argumentname=class (e.g. id=int).
    The arguments' types are not checked in this implementation though.
    Callables with a fitting signature can be added with += or removed with -=.
    All listeners can be notified by calling the EventHook class with fitting
    arguments.

    >>> event = EventHook(id=int, data=dict)
    >>> event += lambda id, data: print("%d %s" % (id, data))
    >>> event(id=5, data={"foo": "bar"})
    5 {'foo': 'bar'}

    >>> event = EventHook(id=int)
    >>> event += lambda wrong_name: None
    Traceback (most recent call last):
        ...
    ValueError: Listener must have these arguments: (id=int)

    >>> event = EventHook(id=int)
    >>> event += lambda id: None
    >>> event(wrong_name=0)
    Traceback (most recent call last):
        ...
    ValueError: This EventHook must be called with these arguments: (id=int)
    '''
    def __init__(self, **signature: Type) -> None:
        self._signature = signature 
        self._argnames = set(signature.keys())
        self._handlers = [] # type: List[Callable]

    def _kwargs_str(self) -> str:
        return ", ".join(k+"="+v.__name__ for k, v in self._signature.items())

    def __iadd__(self, handler: Callable) -> EventHook:
        params = inspect.signature(handler).parameters
        valid = True
        argnames = set(n for n in params.keys())
        if argnames != self._argnames:
            valid = False
        for p in params.values():
            if p.kind == p.VAR_KEYWORD:
                valid = True
                break
            if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                valid = False
                break
        if not valid:
            raise ValueError("Listener must have these arguments: (%s)"
                             % self._kwargs_str())
        self._handlers.append(handler)
        return self

    def __isub__(self, handler: Callable) -> EventHook:
        self._handlers.remove(handler)
        return self

    def __call__(self, *args, **kwargs) -> None:
        if args or set(kwargs.keys()) != self._argnames:
            raise ValueError("This EventHook must be called with these " +
                             "keyword arguments: (%s)" % self._kwargs_str())
        for handler in self._handlers[:]:
            handler(**kwargs)

    def __repr__(self) -> str:
        return "EventHook(%s)" % self._kwargs_str()
