Contributing
============

Docstring Format
----------------

The documentation style is [Sphinx' reStructuredText](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). Here is an example demonstrating the most common syntax:

```python
def foo(a, b):
    """
    Short description of the function.

    :param a: description of the parameter
    :type a: the type of a parameter
    :param int b: another syntax, for specifying the parameter's type and description together

    :return: what the function returns
    :rtype: the type the function returns

    :raises TypeError: possible exception and when is it thrown

    Longer description of the function, may contain multiple lines.

    Italic is formatted with a single asterisk - *italic*.

    Bold is formatted with a double asterisks - **bold**.

    Inline code style is formatted with double backticks: ``identifier``.

    Code blocks are formatted with the code-block directive and an indentation:

    code-block::

        assert foo(1, 1) == True

    Or, you can add double colon at the end of a line to include a code block after that line::

        assert foo(1, 2) == False

    Example of interactive Python session that demonstrate the function are formatted with three greater-thans:

    >>> foo(2, 2)
    True

    The interactive Python session formatting ends after one blank line
    """
    if type(a) is not type(b):
        raise TypeError('a and b are not of the same type')
    return a == b
```
