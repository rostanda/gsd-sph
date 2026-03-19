# Copyright (c) 2016-2024 The Regents of the University of Michigan
# Part of GSD, released under the BSD 2-Clause License.

"""Read and write HOOMD schema GSD files extended for SPH simulations.

:py:mod:`gsd.hoomd` reads and writes GSD files with the ``hoomd`` schema.
This version is extended for use with the SPH solver in ``hoomd-sph3``:
additional particle fields (``slength``, ``density``, ``pressure``, ``energy``,
``auxiliary1``–``auxiliary5``) are added to `ParticleData`, and fields unused
by SPH (``orientation``, ``angmom``, ``charge``, ``diameter``,
``moment_inertia``) are removed from the schema defaults.

Classes:

* `HOOMDTrajectory` - Read and write hoomd schema GSD files.
* `Frame` - Store the state of a single frame.

  * `ConfigurationData` - Store configuration data in a frame.
  * `ParticleData` - Store particle data in a frame (SPH-extended).
  * `BondData` - Store bond topology data in a frame.
  * `ConstraintData` - Store distance constraint data in a frame.

Functions:

* `open` - Open a hoomd schema GSD file.
* `read_log` - Read logged quantities into a dict of time-series arrays.

See Also:
    See :ref:`hoomd-examples` for full examples.
"""

import copy
import json
import logging
import warnings
from collections import OrderedDict

import numpy

try:
    import gsd
except ImportError:
    gsd = None

fl_imported = True
try:
    import gsd.fl
except ImportError:
    fl_imported = False

logger = logging.getLogger('gsd.hoomd')


class ConfigurationData:
    """Store configuration data.

    Access via the `Frame.configuration` attribute.

    Attributes:
        step (int): Time step of this frame (:chunk:`configuration/step`).

        dimensions (int): Number of spatial dimensions
            (:chunk:`configuration/dimensions`). When not set explicitly,
            the value is inferred from `box`: if :math:`L_z = 0` the default
            is 2, otherwise 3. User-set values always take precedence.

        box ((6,) `numpy.ndarray` of ``numpy.float32``): Simulation box
            parameters ``[lx, ly, lz, xy, xz, yz]``
            (:chunk:`configuration/box`).
    """

    _default_value = OrderedDict()
    _default_value['step'] = numpy.uint64(0)
    _default_value['dimensions'] = numpy.uint8(3)
    _default_value['box'] = numpy.array([1, 1, 1, 0, 0, 0], dtype=numpy.float32)

    def __init__(self):
        self.step = None
        self.dimensions = None
        self._box = None

    @property
    def box(self):
        """((6, 1) `numpy.ndarray` of ``numpy.float32``): Box dimensions.

        [lx, ly, lz, xy, xz, yz]. See :chunk:`configuration/box`.
        """
        return self._box

    @box.setter
    def box(self, box):
        self._box = box
        try:
            Lz = box[2]
        except TypeError:
            return
        else:
            if self.dimensions is None:
                self.dimensions = 2 if Lz == 0 else 3

    def validate(self):
        """Validate all attributes.

        Convert every array attribute to a `numpy.ndarray` of the proper
        type and check that all attributes have the correct dimensions.

        Ignore any attributes that are ``None``.

        Warning:
            Array attributes that are not contiguous numpy arrays will be
            replaced with contiguous numpy arrays of the appropriate type.
        """
        logger.debug('Validating ConfigurationData')

        if self.box is not None:
            self.box = numpy.ascontiguousarray(self.box, dtype=numpy.float32)
            self.box = self.box.reshape([6])


class ParticleData:
    """Store particle data chunks (SPH-extended).

    Access via the `Frame.particles` attribute.

    Instances resulting from file read operations always store array quantities
    as `numpy.ndarray` objects of the types listed below. User-created frames
    may supply any array-like that can be converted to the appropriate
    `numpy.ndarray`.

    Attributes:
        N (int): Number of particles in the frame (:chunk:`particles/N`).

        types (tuple[str]):
            Names of the particle types (:chunk:`particles/types`).

        position ((*N*, 3) `numpy.ndarray` of ``numpy.float32``):
            Particle position (:chunk:`particles/position`).

        typeid ((*N*, ) `numpy.ndarray` of ``numpy.uint32``):
            Particle type id (:chunk:`particles/typeid`).

        mass ((*N*, ) `numpy.ndarray` of ``numpy.float32``):
            Particle mass (:chunk:`particles/mass`).

        body ((*N*, ) `numpy.ndarray` of ``numpy.int32``):
            Particle body (:chunk:`particles/body`).

        velocity ((*N*, 3) `numpy.ndarray` of ``numpy.float32``):
            Particle velocity (:chunk:`particles/velocity`).

        slength ((*N*, ) `numpy.ndarray` of ``numpy.float32``):
            Particle smoothing length (:chunk:`particles/slength`).

        density ((*N*, ) `numpy.ndarray` of ``numpy.float32``):
            Particle density (:chunk:`particles/density`).

        pressure ((*N*, ) `numpy.ndarray` of ``numpy.float32``):
            Particle pressure (:chunk:`particles/pressure`).

        energy ((*N*, ) `numpy.ndarray` of ``numpy.float32``):
            Particle energy (:chunk:`particles/energy`).

        auxiliary1 ((*N*, 3) `numpy.ndarray` of ``numpy.float32``):
            Auxiliary vector field 1 (:chunk:`particles/auxiliary1`).

        auxiliary2 ((*N*, 3) `numpy.ndarray` of ``numpy.float32``):
            Auxiliary vector field 2 (:chunk:`particles/auxiliary2`).

        auxiliary3 ((*N*, 3) `numpy.ndarray` of ``numpy.float32``):
            Auxiliary vector field 3 (:chunk:`particles/auxiliary3`).

        auxiliary4 ((*N*, 3) `numpy.ndarray` of ``numpy.float32``):
            Auxiliary vector field 4 (:chunk:`particles/auxiliary4`).

        auxiliary5 ((*N*, 3) `numpy.ndarray` of ``numpy.float32``):
            Auxiliary vector field 5 (:chunk:`particles/auxiliary5`).

        image ((*N*, 3) `numpy.ndarray` of ``numpy.int32``):
            Particle image (:chunk:`particles/image`).

        type_shapes (tuple[dict]): Shape specifications for
            visualizing particle types (:chunk:`particles/type_shapes`).
    """

    _default_value = OrderedDict()
    _default_value['N'] = numpy.uint32(0)
    _default_value['types'] = ['A']
    _default_value['typeid'] = numpy.uint32(0)
    _default_value['mass'] = numpy.float32(1.0)
    _default_value['body'] = numpy.int32(-1)
    _default_value['position'] = numpy.array([0, 0, 0], dtype=numpy.float32)
    _default_value['velocity'] = numpy.array([0, 0, 0], dtype=numpy.float32)
    _default_value['slength'] = numpy.float32(1.0)
    _default_value['density'] = numpy.float32(0.0)
    _default_value['pressure'] = numpy.float32(0.0)
    _default_value['energy'] = numpy.float32(0.0)
    _default_value['auxiliary1'] = numpy.array([0, 0, 0], dtype=numpy.float32)
    _default_value['auxiliary2'] = numpy.array([0, 0, 0], dtype=numpy.float32)
    _default_value['auxiliary3'] = numpy.array([0, 0, 0], dtype=numpy.float32)
    _default_value['auxiliary4'] = numpy.array([0, 0, 0], dtype=numpy.float32)
    _default_value['auxiliary5'] = numpy.array([0, 0, 0], dtype=numpy.float32)
    _default_value['image'] = numpy.array([0, 0, 0], dtype=numpy.int32)
    _default_value['type_shapes'] = [{}]

    def __init__(self):
        self.N = 0
        self.position = None
        self.types = None
        self.typeid = None
        self.mass = None
        self.body = None
        self.velocity = None
        self.slength = None
        self.density = None
        self.pressure = None
        self.energy = None
        self.auxiliary1 = None
        self.auxiliary2 = None
        self.auxiliary3 = None
        self.auxiliary4 = None
        self.auxiliary5 = None
        self.image = None
        self.type_shapes = None

    def validate(self):
        """Validate all attributes.

        Convert every array attribute to a `numpy.ndarray` of the proper
        type and check that all attributes have the correct dimensions.

        Ignore any attributes that are ``None``.

        Warning:
            Array attributes that are not contiguous numpy arrays will be
            replaced with contiguous numpy arrays of the appropriate type.
        """
        logger.debug('Validating ParticleData')

        if self.position is not None:
            self.position = numpy.ascontiguousarray(self.position,
                                                    dtype=numpy.float32)
            self.position = self.position.reshape([self.N, 3])
        if self.typeid is not None:
            self.typeid = numpy.ascontiguousarray(self.typeid,
                                                  dtype=numpy.uint32)
            self.typeid = self.typeid.reshape([self.N])
        if self.mass is not None:
            self.mass = numpy.ascontiguousarray(self.mass, dtype=numpy.float32)
            self.mass = self.mass.reshape([self.N])
        if self.body is not None:
            self.body = numpy.ascontiguousarray(self.body, dtype=numpy.int32)
            self.body = self.body.reshape([self.N])
        if self.velocity is not None:
            self.velocity = numpy.ascontiguousarray(self.velocity,
                                                    dtype=numpy.float32)
            self.velocity = self.velocity.reshape([self.N, 3])
        
        if self.slength is not None:
            self.slength = numpy.ascontiguousarray(self.slength, dtype=numpy.float32)
            self.slength = self.slength.reshape([self.N])
        if self.density is not None:
            self.density = numpy.ascontiguousarray(self.density, dtype=numpy.float32)
            self.density = self.density.reshape([self.N])
        if self.pressure is not None:
            self.pressure = numpy.ascontiguousarray(self.pressure, dtype=numpy.float32)
            self.pressure = self.pressure.reshape([self.N])
        if self.energy is not None:
            self.energy = numpy.ascontiguousarray(self.energy, dtype=numpy.float32)
            self.energy = self.energy.reshape([self.N])
        if self.auxiliary1 is not None:
            self.auxiliary1 = numpy.ascontiguousarray(self.auxiliary1, dtype=numpy.float32)
            self.auxiliary1 = self.auxiliary1.reshape([self.N, 3])
        if self.auxiliary2 is not None:
            self.auxiliary2 = numpy.ascontiguousarray(self.auxiliary2, dtype=numpy.float32)
            self.auxiliary2 = self.auxiliary2.reshape([self.N, 3])
        if self.auxiliary3 is not None:
            self.auxiliary3 = numpy.ascontiguousarray(self.auxiliary3, dtype=numpy.float32)
            self.auxiliary3 = self.auxiliary3.reshape([self.N, 3])
        if self.auxiliary4 is not None:
            self.auxiliary4 = numpy.ascontiguousarray(self.auxiliary4, dtype=numpy.float32)
            self.auxiliary4 = self.auxiliary4.reshape([self.N, 3])
        if self.auxiliary5 is not None:
            self.auxiliary5 = numpy.ascontiguousarray(self.auxiliary5, dtype=numpy.float32)
            self.auxiliary5 = self.auxiliary5.reshape([self.N, 3])

        if self.image is not None:
            self.image = numpy.ascontiguousarray(self.image, dtype=numpy.int32)
            self.image = self.image.reshape([self.N, 3])

        if (self.types is not None
                and (not len(set(self.types)) == len(self.types))):
            raise ValueError("Type names must be unique.")


class BondData:
    """Store bond data chunks.

    Use the `Frame.bonds`, `Frame.angles`, `Frame.dihedrals`,
    `Frame.impropers`, and `Frame.pairs` attributes to access the bond
    topology.

    Instances resulting from file read operations will always store array
    quantities in `numpy.ndarray` objects of the defined types. User created
    frames may provide input data that can be converted to a `numpy.ndarray`.

    See Also:
        `hoomd.State` for a full description of how HOOMD interprets this
        data.

    Note:
        *M* varies depending on the type of bond. `BondData` represents all
        types of topology connections.

        ======== ===
        Type     *M*
        ======== ===
        Bond      2
        Angle     3
        Dihedral  4
        Improper  4
        Pair      2
        ======== ===

    Attributes:
        N (int): Number of bonds/angles/dihedrals/impropers/pairs in the
          frame
          (:chunk:`bonds/N`, :chunk:`angles/N`, :chunk:`dihedrals/N`,
          :chunk:`impropers/N`, :chunk:`pairs/N`).

        types (list[str]): Names of the particle types
          (:chunk:`bonds/types`, :chunk:`angles/types`,
          :chunk:`dihedrals/types`, :chunk:`impropers/types`,
          :chunk:`pairs/types`).

        typeid ((*N*,) `numpy.ndarray` of ``numpy.uint32``):
          Bond type id (:chunk:`bonds/typeid`,
          :chunk:`angles/typeid`, :chunk:`dihedrals/typeid`,
          :chunk:`impropers/typeid`, :chunk:`pairs/types`).

        group ((*N*, *M*) `numpy.ndarray` of ``numpy.uint32``):
          Tags of the particles in the bond (:chunk:`bonds/group`,
          :chunk:`angles/group`, :chunk:`dihedrals/group`,
          :chunk:`impropers/group`, :chunk:`pairs/group`).
    """

    def __init__(self, M):
        self.M = M
        self.N = 0
        self.types = None
        self.typeid = None
        self.group = None

        self._default_value = OrderedDict()
        self._default_value['N'] = numpy.uint32(0)
        self._default_value['types'] = []
        self._default_value['typeid'] = numpy.uint32(0)
        self._default_value['group'] = numpy.array([0] * M, dtype=numpy.int32)

    def validate(self):
        """Validate all attributes.

        Convert every array attribute to a `numpy.ndarray` of the proper
        type and check that all attributes have the correct dimensions.

        Ignore any attributes that are ``None``.

        Warning:
            Array attributes that are not contiguous numpy arrays will be
            replaced with contiguous numpy arrays of the appropriate type.
        """
        logger.debug('Validating BondData')

        if self.typeid is not None:
            self.typeid = numpy.ascontiguousarray(self.typeid, dtype=numpy.uint32)
            self.typeid = self.typeid.reshape([self.N])
        if self.group is not None:
            self.group = numpy.ascontiguousarray(self.group, dtype=numpy.int32)
            self.group = self.group.reshape([self.N, self.M])

        if self.types is not None and (not len(set(self.types)) == len(self.types)):
            msg = 'Type names must be unique.'
            raise ValueError(msg)


class ConstraintData:
    """Store constraint data.

    Use the `Frame.constraints` attribute to access the constraints.

    Instances resulting from file read operations will always store array
    quantities in `numpy.ndarray` objects of the defined types. User created
    frames may provide input data that can be converted to a `numpy.ndarray`.

    See Also:
        `hoomd.State` for a full description of how HOOMD interprets this
        data.

    Attributes:
        N (int): Number of constraints in the frame (:chunk:`constraints/N`).

        value ((*N*, ) `numpy.ndarray` of ``numpy.float32``):
            Constraint length (:chunk:`constraints/value`).

        group ((*N*, *2*) `numpy.ndarray` of ``numpy.uint32``):
            Tags of the particles in the constraint
            (:chunk:`constraints/group`).
    """

    def __init__(self):
        self.M = 2
        self.N = 0
        self.value = None
        self.group = None

        self._default_value = OrderedDict()
        self._default_value['N'] = numpy.uint32(0)
        self._default_value['value'] = numpy.float32(0)
        self._default_value['group'] = numpy.array([0] * self.M, dtype=numpy.int32)

    def validate(self):
        """Validate all attributes.

        Convert every array attribute to a `numpy.ndarray` of the proper
        type and check that all attributes have the correct dimensions.

        Ignore any attributes that are ``None``.

        Warning:
            Array attributes that are not contiguous numpy arrays will be
            replaced with contiguous numpy arrays of the appropriate type.
        """
        logger.debug('Validating ConstraintData')

        if self.value is not None:
            self.value = numpy.ascontiguousarray(self.value, dtype=numpy.float32)
            self.value = self.value.reshape([self.N])
        if self.group is not None:
            self.group = numpy.ascontiguousarray(self.group, dtype=numpy.int32)
            self.group = self.group.reshape([self.N, self.M])


class Frame:
    """System state at one point in time.

    Attributes:
        configuration (`ConfigurationData`): Simulation box and time step.

        particles (`ParticleData`): Per-particle fields (position, velocity,
            SPH fields, etc.).

        bonds (`BondData`): Bond topology (M=2).

        pairs (`BondData`): Special pair topology (M=2).

        constraints (`ConstraintData`): Distance constraints.

        state (dict): In-memory scratch space for caller use. **Not written
            to disk.**

        log (dict): Logged scalar or array quantities written to disk under
            the ``log/`` namespace. Values must be `numpy.ndarray` or
            array-like.
    """

    def __init__(self):
        self.configuration = ConfigurationData()
        self.particles = ParticleData()
        self.bonds = BondData(2)
        self.constraints = ConstraintData()
        self.pairs = BondData(2)
        self.state = {}
        self.log = {}

    def validate(self):
        """Validate all contained frame data."""
        logger.debug('Validating Frame')

        self.configuration.validate()
        self.particles.validate()
        self.bonds.validate()
        self.constraints.validate()
        self.pairs.validate()


class _HOOMDTrajectoryIterable:
    """Iterable over a HOOMDTrajectory object."""

    def __init__(self, trajectory, indices):
        self._trajectory = trajectory
        self._indices = indices
        self._indices_iterator = iter(indices)

    def __next__(self):
        return self._trajectory[next(self._indices_iterator)]

    def __iter__(self):
        return type(self)(self._trajectory, self._indices)

    def __len__(self):
        return len(self._indices)


class _HOOMDTrajectoryView:
    """A view of a HOOMDTrajectory object.

    Enables the slicing and iteration over a subset of a trajectory
    instance.
    """

    def __init__(self, trajectory, indices):
        self._trajectory = trajectory
        self._indices = indices

    def __iter__(self):
        return _HOOMDTrajectoryIterable(self._trajectory, self._indices)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self)(self._trajectory, self._indices[key])

        return self._trajectory[self._indices[key]]


class HOOMDTrajectory:
    """Read and write hoomd gsd files.

    Args:
        file (`gsd.fl.GSDFile`): File to access.

    Open hoomd GSD files with `open`.
    """

    def __init__(self, file):
        if file.mode == 'ab':
            msg = 'Append mode not yet supported'
            raise ValueError(msg)

        self._file = file
        self._initial_frame = None

        # Used to cache positive results when chunks exist in frame 0.
        self._chunk_exists_frame_0 = {}

        logger.info('opening HOOMDTrajectory: ' + str(self.file))

        if self.file.schema != 'hoomd':
            raise RuntimeError('GSD file is not a hoomd schema file: ' + str(self.file))
        valid = False
        version = self.file.schema_version
        if version < (2, 0) and version >= (1, 0):
            valid = True
        if not valid:
            raise RuntimeError(
                'Incompatible hoomd schema version '
                + str(version)
                + ' in: '
                + str(self.file)
            )

        logger.info('found ' + str(len(self)) + ' frames')

    @property
    def file(self):
        """:class:`gsd.fl.GSDFile`: The file handle."""
        return self._file

    def __len__(self):
        """The number of frames in the trajectory."""
        return self.file.nframes

    def append(self, frame):
        """Append a frame to a hoomd gsd file.

        Args:
            frame (:py:class:`Frame`): Frame to append.

        Write the given frame to the file at the current frame and increase
        the frame counter. Do not write any fields that are ``None``. For all
        non-``None`` fields, scan them and see if they match the initial frame
        or the default value. If the given data differs, write it out to the
        frame. If it is the same, do not write it out as it can be instantiated
        either from the value at the initial frame or the default value.
        """
        logger.debug('Appending frame to hoomd trajectory: ' + str(self.file))

        frame.validate()

        # want the initial frame specified as a reference to detect if chunks
        # need to be written
        if self._initial_frame is None and len(self) > 0:
            self._read_frame(0)

        for path in [
                'configuration',
                'particles',
                'bonds',
                'constraints',
                'pairs',
        ]:
            container = getattr(frame, path)
            for name in container._default_value:
                if self._should_write(path, name, frame):
                    logger.debug('writing data chunk: ' + path + '/' + name)
                    data = getattr(container, name)

                    if name == 'N':
                        data = numpy.array([data], dtype=numpy.uint32)
                    if name == 'step':
                        data = numpy.array([data], dtype=numpy.uint64)
                    if name == 'dimensions':
                        data = numpy.array([data], dtype=numpy.uint8)
                    if name in ('types', 'type_shapes'):
                        if name == 'type_shapes':
                            data = [json.dumps(shape_dict) for shape_dict in data]
                        wid = max(len(w) for w in data) + 1
                        b = numpy.array(data, dtype=numpy.dtype((bytes, wid)))
                        data = b.view(dtype=numpy.int8).reshape(len(b), wid)

                    self.file.write_chunk(path + '/' + name, data)

        # # write state data
        # for state, data in frame.state.items():
        #     self.file.write_chunk('state/' + state, data)

        # write log data
        for log, data in frame.log.items():
            self.file.write_chunk('log/' + log, data)

        self.file.end_frame()

    def truncate(self):
        """Remove all frames from the file."""
        self.file.truncate()
        self._initial_frame = None

    def close(self):
        """Close the file."""
        self.file.close()
        del self._initial_frame

    def _should_write(self, path, name, frame):
        """Test if we should write a given data chunk.

        Args:
            path (str): Path part of the data chunk.
            name (str): Name part of the data chunk.
            frame (:py:class:`Frame`): Frame data is from.

        Returns:
            False if the data matches that in the initial frame. False
            if the data matches all default values. True otherwise.
        """
        container = getattr(frame, path)
        data = getattr(container, name)

        if data is None:
            return False

        if self._initial_frame is not None:
            initial_container = getattr(self._initial_frame, path)
            initial_data = getattr(initial_container, name)
            if numpy.array_equal(initial_data, data):
                logger.debug(
                    'skipping data chunk, matches frame 0: ' + path + '/' + name
                )
                return False

        matches_default_value = False
        if name == 'types':
            matches_default_value = data == container._default_value[name]
        else:
            matches_default_value = numpy.array_equiv(
                data, container._default_value[name]
            )

        if matches_default_value and not self._chunk_exists_frame_0.get(
            path + '/' + name, False
        ):
            logger.debug('skipping data chunk, default value: ' + path + '/' + name)
            return False

        return True

    def extend(self, iterable):
        """Append each item of the iterable to the file.

        Args:
            iterable: An iterable object the provides :py:class:`Frame`
                instances. This could be another HOOMDTrajectory, a generator
                that modifies frames, or a list of frames.
        """
        for item in iterable:
            self.append(item)

    def _read_frame(self, idx):
        """Read the frame at the given index from the file.

        Args:
            idx (int): Frame index to read.

        Returns:
            `Frame` with the frame data

        Replace any data chunks not present in the given frame with either data
        from frame 0, or initialize from default values if not in frame 0. Cache
        frame 0 data to avoid file read overhead. Return any default data as
        non-writeable numpy arrays.
        """
        if idx >= len(self):
            raise IndexError

        logger.debug('reading frame ' + str(idx) + ' from: ' + str(self.file))

        if self._initial_frame is None and idx != 0:
            self._read_frame(0)

        frame = Frame()
        # read configuration first
        if self.file.chunk_exists(frame=idx, name='configuration/step'):
            step_arr = self.file.read_chunk(frame=idx, name='configuration/step')
            frame.configuration.step = step_arr[0]

            if idx == 0:
                self._chunk_exists_frame_0['configuration/step'] = True
        elif self._initial_frame is not None:
            frame.configuration.step = self._initial_frame.configuration.step
        else:
            frame.configuration.step = frame.configuration._default_value['step']

        if self.file.chunk_exists(frame=idx, name='configuration/dimensions'):
            dimensions_arr = self.file.read_chunk(
                frame=idx, name='configuration/dimensions'
            )
            frame.configuration.dimensions = dimensions_arr[0]

            if idx == 0:
                self._chunk_exists_frame_0['configuration/dimensions'] = True
        elif self._initial_frame is not None:
            frame.configuration.dimensions = (
                self._initial_frame.configuration.dimensions
            )
        else:
            frame.configuration.dimensions = frame.configuration._default_value[
                'dimensions'
            ]

        if self.file.chunk_exists(frame=idx, name='configuration/box'):
            frame.configuration.box = self.file.read_chunk(
                frame=idx, name='configuration/box'
            )

            if idx == 0:
                self._chunk_exists_frame_0['configuration/box'] = True
        elif self._initial_frame is not None:
            frame.configuration.box = copy.copy(self._initial_frame.configuration.box)
        else:
            frame.configuration.box = copy.copy(
                frame.configuration._default_value['box']
            )

        # then read all groups that have N, types, etc...
        for path in [
                'particles',
                'bonds',
                'constraints',
                'pairs',
        ]:
            container = getattr(frame, path)
            if self._initial_frame is not None:
                initial_frame_container = getattr(self._initial_frame, path)

            container.N = 0
            if self.file.chunk_exists(frame=idx, name=path + '/N'):
                N_arr = self.file.read_chunk(frame=idx, name=path + '/N')
                container.N = N_arr[0]

                if idx == 0:
                    self._chunk_exists_frame_0[path + '/N'] = True
            elif self._initial_frame is not None:
                container.N = initial_frame_container.N

            # type names
            if 'types' in container._default_value:
                if self.file.chunk_exists(frame=idx, name=path + '/types'):
                    tmp = self.file.read_chunk(frame=idx, name=path + '/types')
                    tmp = tmp.view(dtype=numpy.dtype((bytes, tmp.shape[1])))
                    tmp = tmp.reshape([tmp.shape[0]])
                    container.types = list(a.decode('UTF-8') for a in tmp)

                    if idx == 0:
                        self._chunk_exists_frame_0[path + '/types'] = True
                elif self._initial_frame is not None:
                    container.types = copy.copy(initial_frame_container.types)
                else:
                    container.types = copy.copy(container._default_value['types'])

            # type shapes
            if 'type_shapes' in container._default_value and path == 'particles':
                if self.file.chunk_exists(frame=idx, name=path + '/type_shapes'):
                    tmp = self.file.read_chunk(frame=idx, name=path + '/type_shapes')
                    tmp = tmp.view(dtype=numpy.dtype((bytes, tmp.shape[1])))
                    tmp = tmp.reshape([tmp.shape[0]])
                    container.type_shapes = list(
                        json.loads(json_string.decode('UTF-8')) for json_string in tmp
                    )

                    if idx == 0:
                        self._chunk_exists_frame_0[path + '/type_shapes'] = True
                elif self._initial_frame is not None:
                    container.type_shapes = copy.copy(
                        initial_frame_container.type_shapes
                    )
                else:
                    container.type_shapes = copy.copy(
                        container._default_value['type_shapes']
                    )

            for name in container._default_value:
                if name in ('N', 'types', 'type_shapes'):
                    continue

                # per particle/bond quantities
                if self.file.chunk_exists(frame=idx, name=path + '/' + name):
                    container.__dict__[name] = self.file.read_chunk(
                        frame=idx, name=path + '/' + name
                    )

                    if idx == 0:
                        self._chunk_exists_frame_0[path + '/' + name] = True
                else:
                    if (
                        self._initial_frame is not None
                        and initial_frame_container.N == container.N
                    ):
                        # read default from initial frame
                        container.__dict__[name] = initial_frame_container.__dict__[
                            name
                        ]
                    else:
                        # initialize from default value
                        tmp = numpy.array([container._default_value[name]])
                        s = list(tmp.shape)
                        s[0] = container.N
                        container.__dict__[name] = numpy.empty(shape=s, dtype=tmp.dtype)
                        container.__dict__[name][:] = tmp

                    container.__dict__[name].flags.writeable = False


        # read log data
        logged_data_names = self.file.find_matching_chunk_names('log/')
        for log in logged_data_names:
            if self.file.chunk_exists(frame=idx, name=log):
                frame.log[log[4:]] = self.file.read_chunk(frame=idx, name=log)

                if idx == 0:
                    self._chunk_exists_frame_0[log] = True
            elif self._initial_frame is not None:
                frame.log[log[4:]] = self._initial_frame.log[log[4:]]
                frame.log[log[4:]].flags.writeable = False

        # store initial frame
        if self._initial_frame is None and idx == 0:
            self._initial_frame = copy.deepcopy(frame)

        return frame

    def __getitem__(self, key):
        """Index trajectory frames.

        The index can be a positive integer, negative integer, or slice and is
        interpreted the same as `list` indexing.

        Warning:
            As you loop over frames, each frame is read from the file when it is
            reached in the iteration. Multiple passes may lead to multiple disk
            reads if the file does not fit in cache.
        """
        if isinstance(key, slice):
            return _HOOMDTrajectoryView(self, range(*key.indices(len(self))))

        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key >= len(self) or key < 0:
                raise IndexError()
            return self._read_frame(key)

        raise TypeError

    def __iter__(self):
        """Iterate over frames in the trajectory."""
        return _HOOMDTrajectoryIterable(self, range(len(self)))

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the file when the context manager exits."""
        self.file.close()

    def flush(self):
        """Flush all buffered frames to the file."""
        self._file.flush()


def open(name, mode='r'):  # noqa: A001 - allow shadowing builtin open
    """Open a hoomd schema GSD file.

    The return value of `open` can be used as a context manager.

    Args:
        name (str): File name to open.
        mode (str): File open mode.

    Returns:
        `HOOMDTrajectory` instance that accesses the file **name** with the
        given **mode**.

    Valid values for ``mode``:

    +------------------+---------------------------------------------+
    | mode             | description                                 |
    +==================+=============================================+
    | ``'r'``          | Open an existing file for reading.          |
    +------------------+---------------------------------------------+
    | ``'r+'``         | Open an existing file for reading and       |
    |                  | writing.                                    |
    +------------------+---------------------------------------------+
    | ``'w'``          | Open a file for reading and writing.        |
    |                  | Creates the file if needed, or overwrites   |
    |                  | an existing file.                           |
    +------------------+---------------------------------------------+
    | ``'x'``          | Create a gsd file exclusively and opens it  |
    |                  | for reading and writing.                    |
    |                  | Raise :py:exc:`FileExistsError`             |
    |                  | if it already exists.                       |
    +------------------+---------------------------------------------+
    | ``'a'``          | Open a file for reading and writing.        |
    |                  | Creates the file if it doesn't exist.       |
    +------------------+---------------------------------------------+

    """
    if not fl_imported:
        msg = 'file layer module is not available'
        raise RuntimeError(msg)
    if gsd is None:
        msg = 'gsd module is not available'
        raise RuntimeError(msg)

    gsdfileobj = gsd.fl.open(
        name=str(name),
        mode=mode,
        application='gsd.hoomd ' + gsd.version.version,
        schema='hoomd',
        schema_version=[1, 4],
    )

    return HOOMDTrajectory(gsdfileobj)


def read_log(name, scalar_only=False):
    """Read log from a hoomd schema GSD file into a dict of time-series arrays.

    Args:
        name (str): File name to open.
        scalar_only (bool): Set to `True` to include only scalar log values.

    The log data includes :chunk:`configuration/step` and all matching
    :chunk:`log/user_defined`, :chunk:`log/bonds/user_defined`, and
    :chunk:`log/particles/user_defined` quantities in the file.

    Returns:
        `dict`

    Note:
        `read_log` issues a `RuntimeWarning` when there are no matching
        ``log/`` quantities in the file.

    Caution:
        `read_log` requires that a logged quantity has the same shape in all
        frames. Use `open` and `Frame.log` to read files where the shape
        changes from frame to frame.

    To create a *pandas* ``DataFrame`` with the logged data:

    .. ipython:: python

        import pandas

        df = pandas.DataFrame(gsd.hoomd.read_log('log-example.gsd',
                                                  scalar_only=True))
        df
    """
    if not fl_imported:
        msg = 'file layer module is not available'
        raise RuntimeError(msg)
    if gsd is None:
        msg = 'gsd module is not available'
        raise RuntimeError(msg)

    min_supported_numpy = 2
    if int(numpy.version.version.split('.')[0]) < min_supported_numpy:
        msg = 'read_log requires numpy >= 2.0'
        raise RuntimeError(msg)

    with gsd.fl.open(
        name=str(name),
        mode='r',
        application='gsd.hoomd ' + gsd.version.version,
        schema='hoomd',
        schema_version=[1, 4],
    ) as gsdfileobj:
        logged_data_names = gsdfileobj.find_matching_chunk_names('log/')
        # Always log timestep associated with each log entry
        logged_data_names.insert(0, 'configuration/step')
        if len(logged_data_names) == 1:
            warnings.warn(
                'No logged data in file: ' + str(name), RuntimeWarning, stacklevel=2
            )

        logged_data_dict = dict()
        for log in logged_data_names:
            log_exists_frame_0 = gsdfileobj.chunk_exists(frame=0, name=log)
            is_configuration_step = log == 'configuration/step'

            if log_exists_frame_0 or is_configuration_step:
                if is_configuration_step and not log_exists_frame_0:
                    # handle default configuration step on frame 0
                    tmp = numpy.array([0], dtype=numpy.uint64)
                else:
                    tmp = gsdfileobj.read_chunk(frame=0, name=log)
                    # if chunk contains string, put it in the numpy array
                    if isinstance(tmp, str):
                        tmp = numpy.array([tmp], dtype=numpy.dtypes.StringDType)

                if scalar_only and not tmp.shape[0] == 1:
                    continue
                if tmp.shape[0] == 1:
                    logged_data_dict[log] = numpy.full(
                        fill_value=tmp[0],
                        shape=(gsdfileobj.nframes,),
                        dtype=tmp.dtype,
                    )
                else:
                    logged_data_dict[log] = numpy.tile(
                        tmp, (gsdfileobj.nframes, *tuple(1 for _ in tmp.shape))
                    )

            for idx in range(1, gsdfileobj.nframes):
                for key in logged_data_dict.keys():
                    if not gsdfileobj.chunk_exists(frame=idx, name=key):
                        continue
                    data = gsdfileobj.read_chunk(frame=idx, name=key)
                    if (
                        not isinstance(
                            logged_data_dict[key].dtype, numpy.dtypes.StringDType
                        )
                        and len(logged_data_dict[key][idx].shape) == 0
                    ):
                        logged_data_dict[key][idx] = data[0]
                    else:
                        logged_data_dict[key][idx] = data

    return logged_data_dict
