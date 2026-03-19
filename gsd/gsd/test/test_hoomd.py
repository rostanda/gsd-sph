# Copyright (c) 2016-2024 The Regents of the University of Michigan
# Part of GSD, released under the BSD 2-Clause License.

"""Test the gsd.hoomd API."""

import pickle

import numpy
import pytest

import gsd.fl
import gsd.hoomd


def test_create(tmp_path):
    """Test that gsd files can be created."""
    with gsd.hoomd.open(name=tmp_path / 'test_create.gsd', mode='w') as hf:
        assert hf.file.schema == 'hoomd'
        assert hf.file.schema_version >= (1, 0)


def test_append(tmp_path, open_mode):
    """Test that gsd files can be appended to."""
    frame = gsd.hoomd.Frame()
    frame.particles.N = 10

    with gsd.hoomd.open(name=tmp_path / 'test_append.gsd', mode=open_mode.write) as hf:
        for i in range(5):
            frame.configuration.step = i + 1
            hf.append(frame)

    with gsd.hoomd.open(name=tmp_path / 'test_append.gsd', mode=open_mode.read) as hf:
        assert len(hf) == 5


def test_flush(tmp_path, open_mode):
    """Test that HOOMTrajectory objects flush buffered writes."""
    frame = gsd.hoomd.Frame()
    frame.particles.N = 10

    hf = gsd.hoomd.open(name=tmp_path / 'test_append.gsd', mode=open_mode.write)
    for i in range(5):
        frame.configuration.step = i + 1
        hf.append(frame)

    hf.flush()

    with gsd.hoomd.open(name=tmp_path / 'test_append.gsd', mode=open_mode.read) as hf:
        assert len(hf) == 5


def create_frame(i):
    """Helper function to create frame objects."""
    frame = gsd.hoomd.Frame()
    frame.configuration.step = i + 1
    return frame


def test_extend(tmp_path, open_mode):
    """Test that the extend method works."""
    frame = gsd.hoomd.Frame()
    frame.particles.N = 10

    with gsd.hoomd.open(name=tmp_path / 'test_extend.gsd', mode=open_mode.write) as hf:
        hf.extend(create_frame(i) for i in range(5))

    with gsd.hoomd.open(name=tmp_path / 'test_extend.gsd', mode=open_mode.read) as hf:
        assert len(hf) == 5


def test_defaults(tmp_path, open_mode):
    """Test that the property defaults are properly set."""
    frame = gsd.hoomd.Frame()
    frame.particles.N = 2
    frame.bonds.N = 3
    frame.constraints.N = 4
    frame.pairs.N = 7

    with gsd.hoomd.open(
        name=tmp_path / 'test_defaults.gsd', mode=open_mode.write
    ) as hf:
        hf.append(frame)

    with gsd.hoomd.open(name=tmp_path / 'test_defaults.gsd', mode=open_mode.read) as hf:
        s = hf[0]

        assert s.configuration.step == 0
        assert s.configuration.dimensions == 3
        numpy.testing.assert_array_equal(
            s.configuration.box, numpy.array([1, 1, 1, 0, 0, 0], dtype=numpy.float32)
        )
        assert s.particles.N == 2
        assert s.particles.types == ['A']
        assert s.particles.type_shapes == [{}]
        numpy.testing.assert_array_equal(
            s.particles.typeid, numpy.array([0, 0], dtype=numpy.uint32)
        )
        numpy.testing.assert_array_equal(
            s.particles.mass, numpy.array([1, 1], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.particles.body, numpy.array([-1, -1], dtype=numpy.int32)
        )
        numpy.testing.assert_array_equal(
            s.particles.position,
            numpy.array([[0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.velocity,
            numpy.array([[0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.slength, numpy.array([1, 1], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.particles.density, numpy.array([0, 0], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.particles.pressure, numpy.array([0, 0], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.particles.energy, numpy.array([0, 0], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.particles.auxiliary1,
            numpy.array([[0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.auxiliary2,
            numpy.array([[0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.auxiliary3,
            numpy.array([[0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.auxiliary4,
            numpy.array([[0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.auxiliary5,
            numpy.array([[0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.image, numpy.array([[0, 0, 0], [0, 0, 0]], dtype=numpy.int32)
        )

        assert s.bonds.N == 3
        assert s.bonds.types == []
        numpy.testing.assert_array_equal(
            s.bonds.typeid, numpy.array([0, 0, 0], dtype=numpy.uint32)
        )
        numpy.testing.assert_array_equal(
            s.bonds.group, numpy.array([[0, 0], [0, 0], [0, 0]], dtype=numpy.uint32)
        )

        assert s.constraints.N == 4
        numpy.testing.assert_array_equal(
            s.constraints.value, numpy.array([0, 0, 0, 0], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.constraints.group,
            numpy.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=numpy.uint32),
        )

        assert s.pairs.N == 7
        assert s.pairs.types == []
        numpy.testing.assert_array_equal(
            s.pairs.typeid, numpy.array([0] * 7, dtype=numpy.uint32)
        )
        numpy.testing.assert_array_equal(
            s.pairs.group, numpy.array([[0, 0]] * 7, dtype=numpy.uint32)
        )

        assert len(s.state) == 0


def make_nondefault_frame():
    """Make a frame with all non-default values."""
    frame0 = gsd.hoomd.Frame()
    frame0.configuration.step = 10000
    frame0.configuration.dimensions = 2
    frame0.configuration.box = [4, 5, 6, 1.0, 0.5, 0.25]
    frame0.particles.N = 2
    frame0.particles.types = ['A', 'B', 'C']
    frame0.particles.type_shapes = [
        {'type': 'Sphere', 'diameter': 2.0},
        {'type': 'Sphere', 'diameter': 3.0},
        {'type': 'Sphere', 'diameter': 4.0},
    ]
    frame0.particles.typeid = [1, 2]
    frame0.particles.mass = [2, 3]
    frame0.particles.body = [10, 20]
    frame0.particles.position = [[0.1, 0.2, 0.3], [-1.0, -2.0, -3.0]]
    frame0.particles.velocity = [[1.1, 2.2, 3.3], [-3.3, -2.2, -1.1]]
    frame0.particles.slength = [2.0, 3.0]
    frame0.particles.density = [1.5, 2.5]
    frame0.particles.pressure = [0.5, 0.75]
    frame0.particles.energy = [10.0, 20.0]
    frame0.particles.auxiliary1 = [[1, 2, 3], [4, 5, 6]]
    frame0.particles.auxiliary2 = [[1, 0, 0], [0, 1, 0]]
    frame0.particles.auxiliary3 = [[0, 1, 0], [0, 0, 1]]
    frame0.particles.auxiliary4 = [[-1, 0, 0], [0, -1, 0]]
    frame0.particles.auxiliary5 = [[0, 0, 1], [1, 0, 0]]
    frame0.particles.image = [[10, 20, 30], [5, 6, 7]]

    frame0.bonds.N = 1
    frame0.bonds.types = ['bondA', 'bondB']
    frame0.bonds.typeid = [1]
    frame0.bonds.group = [[0, 1]]

    frame0.constraints.N = 1
    frame0.constraints.value = [1.1]
    frame0.constraints.group = [[0, 1]]

    frame0.pairs.N = 1
    frame0.pairs.types = ['pairA', 'pairB']
    frame0.pairs.typeid = [1]
    frame0.pairs.group = [[0, 3]]

    frame0.log['value'] = [1, 2, 4, 10, 12, 18, 22]
    return frame0


def assert_frames_equal(s, frame0, check_position=True, check_step=True):
    """Assert that two frames are equal."""
    if check_step:
        assert s.configuration.step == frame0.configuration.step

    assert s.configuration.dimensions == frame0.configuration.dimensions
    numpy.testing.assert_array_equal(s.configuration.box, frame0.configuration.box)
    assert s.particles.N == frame0.particles.N
    assert s.particles.types == frame0.particles.types
    assert s.particles.type_shapes == frame0.particles.type_shapes
    numpy.testing.assert_array_equal(s.particles.typeid, frame0.particles.typeid)
    numpy.testing.assert_array_equal(s.particles.mass, frame0.particles.mass)
    numpy.testing.assert_array_equal(s.particles.body, frame0.particles.body)
    if check_position:
        numpy.testing.assert_array_equal(
            s.particles.position, frame0.particles.position
        )
    numpy.testing.assert_array_equal(s.particles.velocity, frame0.particles.velocity)
    numpy.testing.assert_array_equal(s.particles.slength, frame0.particles.slength)
    numpy.testing.assert_array_equal(s.particles.density, frame0.particles.density)
    numpy.testing.assert_array_equal(s.particles.pressure, frame0.particles.pressure)
    numpy.testing.assert_array_equal(s.particles.energy, frame0.particles.energy)
    numpy.testing.assert_array_equal(s.particles.auxiliary1, frame0.particles.auxiliary1)
    numpy.testing.assert_array_equal(s.particles.auxiliary2, frame0.particles.auxiliary2)
    numpy.testing.assert_array_equal(s.particles.auxiliary3, frame0.particles.auxiliary3)
    numpy.testing.assert_array_equal(s.particles.auxiliary4, frame0.particles.auxiliary4)
    numpy.testing.assert_array_equal(s.particles.auxiliary5, frame0.particles.auxiliary5)
    numpy.testing.assert_array_equal(s.particles.image, frame0.particles.image)

    assert s.bonds.N == frame0.bonds.N
    assert s.bonds.types == frame0.bonds.types
    numpy.testing.assert_array_equal(s.bonds.typeid, frame0.bonds.typeid)
    numpy.testing.assert_array_equal(s.bonds.group, frame0.bonds.group)

    assert s.constraints.N == frame0.constraints.N
    numpy.testing.assert_array_equal(s.constraints.value, frame0.constraints.value)
    numpy.testing.assert_array_equal(s.constraints.group, frame0.constraints.group)

    assert s.pairs.N == frame0.pairs.N
    assert s.pairs.types == frame0.pairs.types
    numpy.testing.assert_array_equal(s.pairs.typeid, frame0.pairs.typeid)
    numpy.testing.assert_array_equal(s.pairs.group, frame0.pairs.group)


def test_fallback(tmp_path, open_mode):
    """Test that properties fall back to defaults when the N changes."""
    frame0 = make_nondefault_frame()

    frame1 = gsd.hoomd.Frame()
    frame1.particles.N = 2
    frame1.particles.position = [[-2, -1, 0], [1, 3.0, 0.5]]
    frame1.bonds.N = None
    frame1.constraints.N = None
    frame1.pairs.N = None

    frame2 = gsd.hoomd.Frame()
    frame2.particles.N = 3
    frame2.particles.types = ['q', 's']
    frame2.particles.type_shapes = [
        {},
        {'type': 'Ellipsoid', 'a': 7.0, 'b': 5.0, 'c': 3.0},
    ]
    frame2.bonds.N = 3
    frame2.constraints.N = 4
    frame2.pairs.N = 7

    with gsd.hoomd.open(
        name=tmp_path / 'test_fallback.gsd', mode=open_mode.write
    ) as hf:
        hf.extend([frame0, frame1, frame2])

    with gsd.hoomd.open(name=tmp_path / 'test_fallback.gsd', mode=open_mode.read) as hf:
        assert len(hf) == 3
        s = hf[0]

        assert_frames_equal(s, frame0)
        assert 'value' in s.log
        numpy.testing.assert_array_equal(s.log['value'], frame0.log['value'])

        # test that everything but position remained the same in frame 1
        s = hf[1]

        assert_frames_equal(s, frame0, check_position=False)
        assert 'value' in s.log
        numpy.testing.assert_array_equal(s.log['value'], frame0.log['value'])

        # check that the third frame goes back to defaults because it has a
        # different N
        s = hf[2]

        assert s.particles.N == 3
        assert s.particles.types == ['q', 's']
        assert s.particles.type_shapes == frame2.particles.type_shapes
        numpy.testing.assert_array_equal(
            s.particles.typeid, numpy.array([0, 0, 0], dtype=numpy.uint32)
        )
        numpy.testing.assert_array_equal(
            s.particles.mass, numpy.array([1, 1, 1], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.particles.body, numpy.array([-1, -1, -1], dtype=numpy.int32)
        )
        numpy.testing.assert_array_equal(
            s.particles.position,
            numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.velocity,
            numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.slength, numpy.array([1, 1, 1], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.particles.density, numpy.array([0, 0, 0], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.particles.pressure, numpy.array([0, 0, 0], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.particles.energy, numpy.array([0, 0, 0], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.particles.auxiliary1,
            numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.auxiliary2,
            numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.auxiliary3,
            numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.auxiliary4,
            numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.auxiliary5,
            numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            s.particles.image,
            numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=numpy.int32),
        )

        assert s.bonds.N == 3
        assert s.bonds.types == frame0.bonds.types
        numpy.testing.assert_array_equal(
            s.bonds.typeid, numpy.array([0, 0, 0], dtype=numpy.uint32)
        )
        numpy.testing.assert_array_equal(
            s.bonds.group, numpy.array([[0, 0], [0, 0], [0, 0]], dtype=numpy.uint32)
        )

        assert s.constraints.N == 4
        numpy.testing.assert_array_equal(
            s.constraints.value, numpy.array([0, 0, 0, 0], dtype=numpy.float32)
        )
        numpy.testing.assert_array_equal(
            s.constraints.group,
            numpy.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=numpy.uint32),
        )

        assert s.pairs.N == 7
        assert s.pairs.types == frame0.pairs.types
        numpy.testing.assert_array_equal(
            s.pairs.typeid, numpy.array([0] * 7, dtype=numpy.uint32)
        )
        numpy.testing.assert_array_equal(
            s.pairs.group, numpy.array([[0, 0]] * 7, dtype=numpy.uint32)
        )

        assert 'value' in s.log
        numpy.testing.assert_array_equal(s.log['value'], frame0.log['value'])


def test_fallback_to_frame0(tmp_path, open_mode):
    """Test that missing entries fall back to data in frame when N matches."""
    frame0 = make_nondefault_frame()

    frame1 = gsd.hoomd.Frame()
    frame1.configuration.step = 200000
    frame1.particles.N = None
    frame1.bonds.N = None
    frame1.constraints.N = None
    frame1.pairs.N = None

    with gsd.hoomd.open(
        name=tmp_path / 'test_fallback2.gsd', mode=open_mode.write
    ) as hf:
        hf.extend([frame0, frame1])

    with gsd.hoomd.open(
        name=tmp_path / 'test_fallback2.gsd', mode=open_mode.read
    ) as hf:
        assert len(hf) == 2

        s = hf[1]
        assert s.configuration.step == frame1.configuration.step
        assert_frames_equal(s, frame0, check_step=False)
        assert 'value' in s.log
        numpy.testing.assert_array_equal(s.log['value'], frame0.log['value'])


def test_no_fallback(tmp_path, open_mode):
    """Test that writes of default quantities do not fall back to frame 0."""
    frame0 = make_nondefault_frame()

    frame1 = gsd.hoomd.Frame()
    frame1.configuration.step = 200000
    frame1.configuration.dimensions = 3
    frame1.configuration.box = [1, 1, 1, 0, 0, 0]
    frame1.particles.N = frame0.particles.N
    frame1.particles.types = ['A']
    frame1.particles.typeid = [0] * frame0.particles.N
    frame1.particles.type_shapes = [{}]
    frame1.particles.mass = [1.0] * frame0.particles.N
    frame1.particles.body = [-1] * frame0.particles.N
    frame1.particles.position = [[0, 0, 0]] * frame0.particles.N
    frame1.particles.velocity = [[0, 0, 0]] * frame0.particles.N
    frame1.particles.slength = [1.0] * frame0.particles.N
    frame1.particles.density = [0.0] * frame0.particles.N
    frame1.particles.pressure = [0.0] * frame0.particles.N
    frame1.particles.energy = [0.0] * frame0.particles.N
    frame1.particles.auxiliary1 = [[0, 0, 0]] * frame0.particles.N
    frame1.particles.auxiliary2 = [[0, 0, 0]] * frame0.particles.N
    frame1.particles.auxiliary3 = [[0, 0, 0]] * frame0.particles.N
    frame1.particles.auxiliary4 = [[0, 0, 0]] * frame0.particles.N
    frame1.particles.auxiliary5 = [[0, 0, 0]] * frame0.particles.N
    frame1.particles.image = [[0, 0, 0]] * frame0.particles.N

    frame1.bonds.N = frame0.bonds.N
    frame1.bonds.types = ['A']
    frame1.bonds.typeid = [0] * frame0.bonds.N
    frame1.bonds.group = [[0, 0]] * frame0.bonds.N

    frame1.constraints.N = frame0.constraints.N
    frame1.constraints.value = [0] * frame0.constraints.N
    frame1.constraints.group = [[0, 0]] * frame0.constraints.N

    frame1.pairs.N = frame0.pairs.N
    frame1.pairs.types = ['A']
    frame1.pairs.typeid = [0] * frame0.pairs.N
    frame1.pairs.group = [[0, 0]] * frame0.pairs.N

    with gsd.hoomd.open(
        name=tmp_path / 'test_no_fallback.gsd', mode=open_mode.write
    ) as hf:
        hf.extend([frame0, frame1])

    with gsd.hoomd.open(
        name=tmp_path / 'test_no_fallback.gsd', mode=open_mode.read
    ) as hf:
        assert len(hf) == 2

        s = hf[1]
        assert s.configuration.step == frame1.configuration.step
        assert_frames_equal(s, frame1)


def test_iteration(tmp_path, open_mode):
    """Test the iteration protocols for hoomd trajectories."""
    with gsd.hoomd.open(
        name=tmp_path / 'test_iteration.gsd', mode=open_mode.write
    ) as hf:
        hf.extend(create_frame(i) for i in range(20))

    with gsd.hoomd.open(
        name=tmp_path / 'test_iteration.gsd', mode=open_mode.read
    ) as hf:
        step = hf[-1].configuration.step
        assert step == 20

        step = hf[-2].configuration.step
        assert step == 19

        step = hf[-3].configuration.step
        assert step == 18

        step = hf[0].configuration.step
        assert step == 1

        step = hf[-20].configuration.step
        assert step == 1

        with pytest.raises(IndexError):
            step = hf[-21].configuration.step

        with pytest.raises(IndexError):
            step = hf[20]

        frames = hf[5:10]
        steps = [frame.configuration.step for frame in frames]
        assert steps == [6, 7, 8, 9, 10]

        frames = hf[15:50]
        steps = [frame.configuration.step for frame in frames]
        assert steps == [16, 17, 18, 19, 20]

        frames = hf[15:-3]
        steps = [frame.configuration.step for frame in frames]
        assert steps == [16, 17]


def test_slicing_and_iteration(tmp_path, open_mode):
    """Test that hoomd trajectories can be sliced."""
    with gsd.hoomd.open(name=tmp_path / 'test_slicing.gsd', mode=open_mode.write) as hf:
        hf.extend(create_frame(i) for i in range(20))

    with gsd.hoomd.open(name=tmp_path / 'test_slicing.gsd', mode=open_mode.read) as hf:
        # Test len()-function on trajectory and sliced trajectory.
        assert len(hf) == 20
        assert len(hf[:10]) == 10

        # Test len()-function with explicit iterator.
        assert len(iter(hf)) == len(hf)
        assert len(iter(hf[:10])) == len(hf[:10])

        # Test iteration with implicit iterator.
        # All iterations are run twice to check for issues
        # with iterator exhaustion.
        assert len(list(hf)) == len(hf)
        assert len(list(hf)) == len(hf)
        assert len(list(hf[:10])) == len(hf[:10])
        assert len(list(hf[:10])) == len(hf[:10])

        # Test iteration with explicit iterator.
        hf_iter = iter(hf)
        assert len(hf_iter) == len(hf)  # sanity check
        assert len(list(hf_iter)) == len(hf)
        assert len(list(hf_iter)) == len(hf)

        # Test iteration with explicit sliced iterator.
        hf_iter = iter(hf[:10])
        assert len(hf_iter) == 10  # sanity check
        assert len(list(hf_iter)) == 10
        assert len(list(hf_iter)) == 10

        # Test frame selection
        with pytest.raises(IndexError):
            hf[len(hf)]
        assert hf[0].configuration.step == hf[0].configuration.step
        assert hf[len(hf) - 1].configuration.step == hf[-1].configuration.step


def test_view_slicing_and_iteration(tmp_path, open_mode):
    """Test that trajectories can be sliced."""
    with gsd.hoomd.open(name=tmp_path / 'test_slicing.gsd', mode=open_mode.write) as hf:
        hf.extend(create_frame(i) for i in range(40))

    with gsd.hoomd.open(name=tmp_path / 'test_slicing.gsd', mode=open_mode.read) as hf:
        view = hf[::2]

        # Test len()-function on trajectory and sliced view.
        assert len(view) == 20
        assert len(view[:10]) == 10
        assert len(view[::2]) == 10

        # Test len()-function with explicit iterator.
        assert len(iter(view)) == len(view)
        assert len(iter(view[:10])) == len(view[:10])

        # Test iteration with implicit iterator.
        # All iterations are run twice to check for issues
        # with iterator exhaustion.
        assert len(list(view)) == len(view)
        assert len(list(view)) == len(view)
        assert len(list(view[:10])) == len(view[:10])
        assert len(list(view[:10])) == len(view[:10])
        assert len(list(view[::2])) == len(view[::2])
        assert len(list(view[::2])) == len(view[::2])

        # Test iteration with explicit iterator.
        view_iter = iter(view)
        assert len(view_iter) == len(view)  # sanity check
        assert len(list(view_iter)) == len(view)
        assert len(list(view_iter)) == len(view)

        # Test iteration with explicit sliced iterator.
        view_iter = iter(view[:10])
        assert len(view_iter) == 10  # sanity check
        assert len(list(view_iter)) == 10
        assert len(list(view_iter)) == 10

        # Test frame selection
        with pytest.raises(IndexError):
            view[len(view)]
        assert view[0].configuration.step == view[0].configuration.step
        assert view[len(view) - 1].configuration.step == view[-1].configuration.step


def test_truncate(tmp_path):
    """Test the truncate API."""
    with gsd.hoomd.open(name=tmp_path / 'test_iteration.gsd', mode='w') as hf:
        hf.extend(create_frame(i) for i in range(20))

        assert len(hf) == 20
        s = hf[10]  # noqa
        assert hf._initial_frame is not None

        hf.truncate()
        assert len(hf) == 0
        assert hf._initial_frame is None



def test_log(tmp_path, open_mode):
    """Test the log chunks."""
    frame0 = gsd.hoomd.Frame()

    frame0.log['particles/net_force'] = [[1, 2, 3], [4, 5, 6]]
    frame0.log['particles/pair_lj_energy'] = [0, -5, -8, -3]
    frame0.log['value/potential_energy'] = [10]
    frame0.log['value/pressure'] = [-3]
    frame0.log['category'] = 'A'

    frame1 = gsd.hoomd.Frame()

    frame1.log['particles/pair_lj_energy'] = [1, 2, -4, -10]
    frame1.log['value/pressure'] = [5]
    frame1.log['category'] = 'BBB'

    with gsd.hoomd.open(name=tmp_path / 'test_log.gsd', mode=open_mode.write) as hf:
        hf.extend([frame0, frame1])

    with gsd.hoomd.open(name=tmp_path / 'test_log.gsd', mode=open_mode.read) as hf:
        assert len(hf) == 2
        s = hf[0]

        numpy.testing.assert_array_equal(
            s.log['particles/net_force'], frame0.log['particles/net_force']
        )
        numpy.testing.assert_array_equal(
            s.log['particles/pair_lj_energy'], frame0.log['particles/pair_lj_energy']
        )
        numpy.testing.assert_array_equal(
            s.log['value/potential_energy'], frame0.log['value/potential_energy']
        )
        numpy.testing.assert_array_equal(
            s.log['value/pressure'], frame0.log['value/pressure']
        )
        assert s.log['category'] == frame0.log['category']

        s = hf[1]

        # unspecified entries pull from frame 0
        numpy.testing.assert_array_equal(
            s.log['particles/net_force'], frame0.log['particles/net_force']
        )
        numpy.testing.assert_array_equal(
            s.log['value/potential_energy'], frame0.log['value/potential_energy']
        )

        # specified entries are different in frame 1
        numpy.testing.assert_array_equal(
            s.log['particles/pair_lj_energy'], frame1.log['particles/pair_lj_energy']
        )
        numpy.testing.assert_array_equal(
            s.log['value/pressure'], frame1.log['value/pressure']
        )
        assert s.log['category'] == frame1.log['category']


def test_pickle(tmp_path):
    """Test that hoomd trajectory objects can be pickled."""
    with gsd.hoomd.open(name=tmp_path / 'test_pickling.gsd', mode='w') as traj:
        traj.extend(create_frame(i) for i in range(20))
        with pytest.raises(pickle.PickleError):
            pkl = pickle.dumps(traj)
    with gsd.hoomd.open(name=tmp_path / 'test_pickling.gsd', mode='r') as traj:
        pkl = pickle.dumps(traj)
        with pickle.loads(pkl) as hf:
            assert len(hf) == 20


@pytest.mark.parametrize(
    'container', ['particles', 'bonds', 'pairs']
)
def test_no_duplicate_types(tmp_path, container):
    """Test that duplicate types raise an error."""
    with gsd.hoomd.open(name=tmp_path / 'test_create.gsd', mode='w') as hf:
        frame = gsd.hoomd.Frame()

        getattr(frame, container).types = ['A', 'B', 'B', 'C']

        with pytest.raises(ValueError):
            hf.append(frame)


def test_read_log(tmp_path):
    """Test that data logged in gsd files are read correctly."""
    frame0 = gsd.hoomd.Frame()
    frame0.log['particles/pair_lj_energy'] = [0, -5, -8, -3]
    frame0.log['particles/pair_lj_force'] = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
    ]
    frame0.log['value/potential_energy'] = [10]
    frame0.log['value/pressure'] = [-3]
    frame0.log['category'] = 'A'

    frame1 = gsd.hoomd.Frame()
    frame1.configuration.step = 1
    frame1.log['particles/pair_lj_energy'] = [1, 2, -4, -10]
    frame1.log['particles/pair_lj_force'] = [
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
        (4, 4, 4),
    ]
    frame1.log['value/pressure'] = [5]
    frame1.log['category'] = 'BBB'

    with gsd.hoomd.open(name=tmp_path / 'test_log.gsd', mode='w') as hf:
        hf.extend([frame0, frame1])

    # Test scalar_only = False
    logged_data_dict = gsd.hoomd.read_log(
        name=tmp_path / 'test_log.gsd', scalar_only=False
    )

    assert len(logged_data_dict) == 6
    assert list(logged_data_dict.keys()) == [
        'configuration/step',
        'log/particles/pair_lj_energy',
        'log/particles/pair_lj_force',
        'log/value/potential_energy',
        'log/value/pressure',
        'log/category',
    ]

    numpy.testing.assert_array_equal(logged_data_dict['configuration/step'], [0, 1])
    numpy.testing.assert_array_equal(
        logged_data_dict['log/particles/pair_lj_energy'],
        [
            frame0.log['particles/pair_lj_energy'],
            frame1.log['particles/pair_lj_energy'],
        ],
    )
    numpy.testing.assert_array_equal(
        logged_data_dict['log/particles/pair_lj_force'],
        [frame0.log['particles/pair_lj_force'], frame1.log['particles/pair_lj_force']],
    )
    numpy.testing.assert_array_equal(
        logged_data_dict['log/value/potential_energy'],
        [*frame0.log['value/potential_energy'], *frame0.log['value/potential_energy']],
    )
    numpy.testing.assert_array_equal(
        logged_data_dict['log/value/pressure'],
        [*frame0.log['value/pressure'], *frame1.log['value/pressure']],
    )
    numpy.testing.assert_array_equal(
        logged_data_dict['log/category'],
        numpy.array(
            [frame0.log['category'], frame1.log['category']],
            dtype=numpy.dtypes.StringDType,
        ),
    )

    # Test scalar_only = True
    logged_data_dict = gsd.hoomd.read_log(
        name=tmp_path / 'test_log.gsd', scalar_only=True
    )
    assert len(logged_data_dict) == 4
    assert list(logged_data_dict.keys()) == [
        'configuration/step',
        'log/value/potential_energy',
        'log/value/pressure',
        'log/category',
    ]
    numpy.testing.assert_array_equal(logged_data_dict['configuration/step'], [0, 1])
    numpy.testing.assert_array_equal(
        logged_data_dict['log/value/potential_energy'],
        [*frame0.log['value/potential_energy'], *frame0.log['value/potential_energy']],
    )
    numpy.testing.assert_array_equal(
        logged_data_dict['log/value/pressure'],
        [*frame0.log['value/pressure'], *frame1.log['value/pressure']],
    )
    numpy.testing.assert_array_equal(
        logged_data_dict['log/category'],
        numpy.array(
            [frame0.log['category'], frame1.log['category']],
            dtype=numpy.dtypes.StringDType,
        ),
    )


def test_read_log_warning(tmp_path):
    """Test that read_log issues a warning."""
    frame = gsd.hoomd.Frame()

    with gsd.hoomd.open(name=tmp_path / 'test_log.gsd', mode='w') as hf:
        hf.extend([frame])

    with pytest.warns(RuntimeWarning):
        log = gsd.hoomd.read_log(tmp_path / 'test_log.gsd')

    assert list(log.keys()) == ['configuration/step']


def test_initial_frame_copy(tmp_path, open_mode):
    """Ensure that the user does not unintentionally modify _initial_frame."""
    with gsd.hoomd.open(
        name=tmp_path / 'test_initial_frame_copy.gsd', mode=open_mode.write
    ) as hf:
        frame = make_nondefault_frame()

        hf.append(frame)

        frame.configuration.step *= 2
        del frame.log['value']
        hf.append(frame)

    with gsd.hoomd.open(
        name=tmp_path / 'test_initial_frame_copy.gsd', mode=open_mode.read
    ) as hf:
        assert len(hf) == 2

        # Verify that the user does not get a reference to the initial frame cache.
        frame_0 = hf[0]
        initial = hf._initial_frame
        assert frame_0 is not initial

        # Verify that no mutable objects from the initial frame cache are presented to
        # the user.
        frame_1 = hf[1]
        assert frame_1.configuration.box is not initial.configuration.box
        assert frame_1.particles.types is not initial.particles.types
        assert frame_1.particles.type_shapes is not initial.particles.type_shapes
        assert frame_1.particles.position is initial.particles.position
        assert not frame_1.particles.position.flags.writeable
        assert frame_1.particles.typeid is initial.particles.typeid
        assert not frame_1.particles.typeid.flags.writeable
        assert frame_1.particles.mass is initial.particles.mass
        assert not frame_1.particles.mass.flags.writeable
        assert frame_1.particles.body is initial.particles.body
        assert not frame_1.particles.body.flags.writeable
        assert frame_1.particles.velocity is initial.particles.velocity
        assert not frame_1.particles.velocity.flags.writeable
        assert frame_1.particles.slength is initial.particles.slength
        assert not frame_1.particles.slength.flags.writeable
        assert frame_1.particles.density is initial.particles.density
        assert not frame_1.particles.density.flags.writeable
        assert frame_1.particles.pressure is initial.particles.pressure
        assert not frame_1.particles.pressure.flags.writeable
        assert frame_1.particles.energy is initial.particles.energy
        assert not frame_1.particles.energy.flags.writeable
        assert frame_1.particles.auxiliary1 is initial.particles.auxiliary1
        assert not frame_1.particles.auxiliary1.flags.writeable
        assert frame_1.particles.auxiliary2 is initial.particles.auxiliary2
        assert not frame_1.particles.auxiliary2.flags.writeable
        assert frame_1.particles.auxiliary3 is initial.particles.auxiliary3
        assert not frame_1.particles.auxiliary3.flags.writeable
        assert frame_1.particles.auxiliary4 is initial.particles.auxiliary4
        assert not frame_1.particles.auxiliary4.flags.writeable
        assert frame_1.particles.auxiliary5 is initial.particles.auxiliary5
        assert not frame_1.particles.auxiliary5.flags.writeable
        assert frame_1.particles.image is initial.particles.image
        assert not frame_1.particles.image.flags.writeable

        assert frame_1.bonds.types is not initial.bonds.types
        assert frame_1.bonds.typeid is initial.bonds.typeid
        assert frame_1.bonds.group is initial.bonds.group
        assert not frame_1.bonds.typeid.flags.writeable
        assert not frame_1.bonds.group.flags.writeable

        assert frame_1.constraints.value is initial.constraints.value
        assert frame_1.constraints.group is initial.constraints.group
        assert not frame_1.constraints.value.flags.writeable
        assert not frame_1.constraints.group.flags.writeable

        assert frame_1.pairs.types is not initial.pairs.types
        assert frame_1.pairs.typeid is initial.pairs.typeid
        assert frame_1.pairs.group is initial.pairs.group
        assert not frame_1.pairs.typeid.flags.writeable
        assert not frame_1.pairs.group.flags.writeable

        assert frame_1.log is not initial.log
        for key in frame_1.log.keys():
            assert frame_1.log[key] is initial.log[key]
            assert not frame_1.log[key].flags.writeable
