from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="geniusrise_healthcare",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[],
    python_requires=">=3.10",
    author="ixaxaar",
    author_email="ixaxaar@geniusrise.ai",
    description="example of NER using vector knowledge graph search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ixaxaar/geniusrise_healthcare",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
