from easypy.semver import SemVer, SMV


def test_loads():
    semver = SemVer.loads('3.4.5.6-hello')
    assert semver.build == 6
    assert semver.patch == 5
    assert semver.minor == 4
    assert semver.major == 3
    assert semver.tag == 'hello'

    semver = SemVer.loads('3_4_5:hello', separator='_', tag_separator=':')
    assert semver.build is None
    assert semver.patch == 5
    assert semver.minor == 4
    assert semver.major == 3
    assert semver.tag == 'hello'


def test_dumps():
    version = '3.4.5.6-hello'
    assert SemVer.loads(version).dumps() == version

    string = SemVer.loads(version).dumps(separator='_', tag_separator=':')
    assert string == version.replace('.', '_').replace('-', ':')


def test_copy():
    semver = SemVer.loads('1000.2000.3000.5000-10102010')
    assert semver.copy() == semver

    assert semver.copy(major=1500).major == 1500


def test_comparison():
    semver1 = SemVer.loads('2.1-aaa')
    semver2 = SemVer.loads('2.1-bbb')
    semver3 = SemVer.loads('2.1.2-a')
    semver4 = SemVer.loads('2.2')
    assert semver1 < semver2 < semver3 < semver4

    assert semver1 == semver2.copy(tag='aaa')


def test_bumping():
    semver = SemVer.loads('1.1.1.1-aaa')
    assert semver.bump_build().dumps() == '1.1.1.2'

    # Build part is only printed when it is set
    assert semver.bump_patch().dumps() == '1.1.2'
    assert semver.bump_minor().dumps() == '1.2.0'
    assert semver.bump_major().dumps() == '2.0.0'
    assert semver.bump_major(clear_tag=False).dumps() == '2.0.0-aaa'
