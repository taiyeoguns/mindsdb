import setuptools
import os

about = {}
with open("lightwood_handler/__about__.py") as fp:
    exec(fp.read(), about)

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

# TODO: relative import okay?
with open('../../../../requirements_test.txt', 'r') as req_test_file:
    requirements.extend(iter(req_test_file.read().splitlines()))
setuptools.setup(
    name=about['__title__'],
    version=about['__version__'],
    url=about['__github__'],
    download_url=about['__pypi__'],
    license=about['__license__'],
    author=about['__author__'],
    description=about['__description__'],
    packages=setuptools.find_packages(),
    package_data={'project': ['requirements.txt']},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7"
)