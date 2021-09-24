import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyPeriod",  # Replace with your own username
    version="0.2.4",
    author="Jacob Sundstrom",
    author_email="jacob.sundstrom@gmail.com",
    description=
    "Periodicity Transforms in Python (Sethares and Staley, Ramanujan, my own)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/woolgathering/pyPeriod",
    install_requires=["numpy", "scipy"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
