import os
from typing import List
from setuptools import setup, find_packages


ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> List[str]:
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setup(
    name="matcha",
    version="1.0.0",
    author="rayuru",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=get_requirements(),
)
