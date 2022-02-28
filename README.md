convDirect
==========

The convDirect library provides different implementations of the direct convolution operation.


Compilation and installation
----------------------------

To compile and install all the convDirect implementations in the convDirect library, the next software should be
available:
* the BLIS library <https://github.com/flame/blis>, and
* the Apache TVM compiler <https://tvm.apache.org/>.

As convDirect also provides a direct call to convGemm, after cloning this project, the corresponding git submodule
should be initialized with:

```shell
git submodule update --init
```

In case you want to update the convGemm submodule to the most recent version, the next commando should be executed: 

```shell
git submodule update --remote --merge
```

Please note that CMake is used to build convDirect. In case you require a more recent version of CMake, you can download it
from https://cmake.org/download/ To install CMake from a compressed CMake archive, just uncompress it and add the
'``cmake-VERSION-ARCH/bin``' directory to the system PATH.

To compile and install the convDirect library, please execute the next commands:

```shell
cd build
cmake [-D CMAKE_PREFIX_PATH=BLIS_INSTALL_PREFIX;TVM_INSTALL_PREFIX] [-D CMAKE_INSTALL_PREFIX=INSTALL_PREFIX] ..
make                 # Alternatively:  cmake --build . --clean-first
make install         # Alternatively:  cmake --install .
```

where ``BLIS_INSTALL_PREFIX`` is the prefix PATH where BLIS is installed, ``TVM_INSTALL_PREFIX`` is the prefix PATH
where TVM is installed, if it is installed on a different path than BLIS, and ``INSTALL_PREFIX`` is the prefix PATH
where ``lib/libconvDirect.so`` and ``include/convdirect.h`` will be installed.

The ``-D CMAKE_PREFIX_PATH=BLIS_INSTALL_PREFIX`` option on the first ``cmake`` command only is necessary if:

1. BLIS is not installed in a system PATH,
2. the environment variable ``LD_LIBRARY_PATH`` is not defined or does not include the ``BLIS_INSTALL_PREFIX`` PATH, and
3. ``BLIS_INSTALL_PREFIX`` is different of ``INSTALL_PREFIX``.

The same applies to ``TVM_INSTALL_PREFIX``, but for Apache TVM.

As for the ``-D CMAKE_PREFIX_PATH=INSTALL_PREFIX`` option, it is only required if the convDirect library should be
installed on a prefix PATH different of ``/usr/local``.

For example, if BLIS is installed under ``~/opt/hpca_pydtnn`` and the convDirect library should be installed also under
that directory, the next commands are sufficient:

```shell
cd build
cmake -D CMAKE_INSTALL_PREFIX=~/opt/hpca_pydtnn ..
make                 # Alternatively:  cmake --build . --clean-first
make install         # Alternatively:  cmake --install . (this does not work with cmake older versions)
```

Performance evaluation of the different implementations
-------------------------------------------------------

To evaluate all the provided implementations on convDirect, recompile convDirect adding the ``-D COMPILE_TESTS=ON``
option and execute the ``evaluation`` target. For example as in:

```shell
cd build
cmake -D CMAKE_INSTALL_PREFIX=~/opt/hpca_pydtnn -D COMPILE_TESTS=ON ..
make evaluation      # Alternatively:  cmake --build . --target=evaluation
```

A Python script is also provided to automatically process the evaluation results and generate the corresponding plots.
Once the evaluation is completed, the instructions to run it locally or on another machine will be shown on screen. This
script requires the next Python modules: ``numpy``, ``pandas``, ``tabulate``, ``matplotlib``.

Testing the different implementations
-------------------------------------

To test the different implementations against a reference convolution, recompile convDirect adding the ``-D COMPILE_TESTS=ON``
option and execute the ``all_close_test`` target. For example as in:

```shell
cd build
cmake -D CMAKE_INSTALL_PREFIX=~/opt/hpca_pydtnn -D COMPILE_TESTS=ON ..
make all_close_test  # Alternatively:  cmake --build . --all_close_test
```

Please note that the performance results shown while performing this test can not be accurate as only one iteration of
each evaluated convolution is performed.
