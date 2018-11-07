from setuptools import setup, find_packages


setup(
    name='kedavra',
    version='0.1',
    author='gmoben',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'kedavra=kedavra.cli:main'
        ]
    },
)
