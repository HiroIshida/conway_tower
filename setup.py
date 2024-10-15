import sys

try:
    from skbuild import setup
except ImportError:
    raise Exception

setup(
    name="conway_tower",
    version="0.0.0",
    description="conway tower",
    author='Hirokazu Ishida',
    license="MIT",
    packages=["conway_tower"],
    package_dir={'': 'python'},
    cmake_install_dir='python/conway_tower/',
    package_data={"conway_tower": ["_conway_tower.pyi"]}
    )
