from collections import namedtuple
import re


class SemVerParseException(ValueError):
    pass


class SemVer(namedtuple("SemVer", "major minor patch build tag")):
    """ Semantic Version object

    From https://semver.org:
    Given a version number MAJOR.MINOR.PATCH, increment the:

    MAJOR version when you make incompatible API changes,
    MINOR version when you add functionality in a backwards-compatible manner, and
    PATCH version when you make backwards-compatible bug fixes.
    Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

    We use a fourth part, let's call it BUILD and define it is incremented sporadically.
    """

    @classmethod
    def loads(cls, string, *, separator='.', tag_separator='-', raise_on_failure=True):
        """
        Load a string into a SemVer object.

        :param string: The string representing the semantic version. Must adhere to semver format.
        :type string: str
        :param separator: Use a different version part separator.
        :type separator: str, optional
        :param tag_separator: Use a different tag separator.
        :type tag_separator: str, optional
        :param raise_on_failure: Whether to raise on failure or just return None
        :type raise_on_failure: bool, optional
        """

        string, _, tag = string.partition(tag_separator)
        parts = string.split(separator)
        try:
            return cls(*parts, tag=tag)
        except ValueError as e:
            if raise_on_failure:
                raise SemVerParseException("Error parsing %s: %s" % (string, str(e))) from None
            else:
                return None

    @classmethod
    def loads_fuzzy(cls, string):
        """
        Load a string into a SemVer without enforcing semver format.

        :param string: The string representing the semantic version. Must adhere to semver format.
        :type string: str
        """

        regex = re.compile(r"((?:\d+[.-])+)(.*)")
        string, tag = regex.fullmatch(string).groups()
        parts = re.split("[-.]", string)
        return cls(*filter(None, parts), tag=tag)

    def __new__(cls, major=0, minor=0, patch=None, build=None, *, tag=None):
        return super().__new__(
            cls,
            major=int(major),
            minor=int(minor),
            patch=int(patch) if patch is not None else None,
            build=int(build) if build is not None else None,
            tag="" if tag is None else str(tag),
        )

    def __str__(self):
        return self.dumps()

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self)

    def _to_tuple(self):
        return (self.major, self.minor, self.patch or 0, self.build or 0)

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        return self._to_tuple() == other._to_tuple() and self.tag == other.tag

    def __lt__(self, other):
        assert isinstance(other, self.__class__)
        return (self._to_tuple(), self.tag) < (other._to_tuple(), other.tag)

    def __gt__(self, other):
        assert isinstance(other, self.__class__)
        return (self._to_tuple(), self.tag) > (other._to_tuple(), other.tag)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        return not self.__gt__(other)

    def dumps(self, *, separator='.', tag_separator='-'):
        """
        Dump SemVer into a string representation.

        :param separator: Use a different version part separator.
        :type separator: str, optional
        :param tag_separator: Use a different tag separator.
        :type tag_separator: str, optional
        """

        template = "{self.major}{separator}{self.minor}"
        if self.patch is not None:
            template += "{separator}{self.patch}"
        if self.build:
            template += "{separator}{self.build}"
        if self.tag:
            template += "{tag_separator}{self.tag}"

        return template.format(**locals())

    def copy(self, **kw):
        """
        Copy this SemVer object to a new one with optional changes.

        :param kw: Change some of the object's attributes
            (Any of major, minor, patch, build, tag).
        """

        return self.__class__(**dict(self._asdict(), **kw))

    def bump_build(self, clear_tag=True):
        """
        Return a copy of this object with the build part incremented by one

        :param clear_tag: Whether to clear the SemVer tag part
        :type clear_tag: bool, optional
        """

        return self.copy(
            build=(0 if self.build is None else self.build) + 1,
            tag='' if clear_tag else self.tag)

    def bump_patch(self, clear_tag=True):
        """
        Return a copy of this object with the patch part incremented by one
        Bumping patch resets build part.

        :param clear_tag: Whether to clear the SemVer tag part
        :type clear_tag: bool, optional
        """

        return self.copy(
            build=0, patch=self.patch + 1,
            tag='' if clear_tag else self.tag)

    def bump_minor(self, clear_tag=True):
        """
        Return a copy of this object with the minor part incremented by one
        Bumping minor resets build and patch parts.

        :param clear_tag: Whether to clear the SemVer tag part
        :type clear_tag: bool, optional
        """

        return self.copy(
            build=0, patch=0, minor=self.minor + 1,
            tag='' if clear_tag else self.tag)

    def bump_major(self, clear_tag=True):
        """
        Return a copy of this object with the major part incremented by one
        Bumping major resets build, patch and minor parts.

        :param clear_tag: Whether to clear the SemVer tag part
        :type clear_tag: bool, optional
        """

        return self.copy(
            build=0, patch=0, minor=0, major=self.major + 1,
            tag='' if clear_tag else self.tag)


SMV = SemVer.loads
