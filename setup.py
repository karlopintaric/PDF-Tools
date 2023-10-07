from setuptools import setup, find_packages

setup(
    name="pdf-tools",
    version="0.1.0",
    author="Karlo Pintaric",
    packages=find_packages(include=["src"]),
    python_requires=">=3.9",
)
