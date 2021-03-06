if [ "$PYTHON_VERSION" = "pypy" ]; then
    # Upgrade to pypy 4.0.1 -- original recipe taken from google/oauth2client
    git clone https://github.com/yyuu/pyenv.git ${HOME}/.pyenv
    export PYENV_ROOT="${HOME}/.pyenv"
    export PATH="${PYENV_ROOT}/bin:${PATH}"
    eval "$(pyenv init -)"
    pyenv install pypy-4.0.1
    pyenv global pypy-4.0.1

    pypy -m pip install nose pyflakes==1.0.0 nose-timer
    which pyflakes
    pypy -m pip install --user git+https://bitbucket.org/pypy/numpy.git@pypy-4.0.1
else # Otherwise, CPython... go through conda
    if [ "$TRAVIS_OS_NAME" = "osx" ]; then
        wget http://repo.continuum.io/miniconda/Miniconda-3.7.0-MacOSX-x86_64.sh -O miniconda.sh;
    else
        wget http://repo.continuum.io/miniconda/Miniconda-3.7.0-Linux-x86_64.sh -O miniconda.sh;
    fi

    bash miniconda.sh -b

    export PATH=$HOME/miniconda/bin:$PATH
    conda update conda -y
    conda install --yes conda-build jinja2 binstar pip
    conda config --add channels omnia
    conda config --add channels ambermd

    if [ -z "$MINIMAL_PACKAGES" ]; then
        conda create -y -n myenv python=$PYTHON_VERSION \
            numpy scipy pandas nose openmm coverage nose-timer \
            python-coveralls ambermini=16.16 netCDF4
        conda update -y -n myenv --all
        conda install -y -n myenv pyflakes=1.0.0
        conda install -y -n myenv rdkit==2015.09.1 -c omnia
        conda install -y -n myenv boost==1.59.0 -c omnia
        conda install -y -n myenv nglview -c bioconda
        # Do not build dependencies for pysander, since that will pull in an old
        # version of ParmEd.
        conda install -y --no-deps pysander
    else
        # Do not install the full numpy/scipy stack
        conda create -y -n myenv python=$PYTHON_VERSION numpy nose pyflakes=1.0.0 \
            coverage nose-timer python-coveralls
    fi

    source activate myenv
fi # CPython
