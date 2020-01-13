import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

fid = open('statispy/version.py')
vers = fid.readlines()[-1].split()[-1].strip("\"'")
fid.close()

setuptools.setup(
    name="statispy",
    version=vers,
    author="Lukas Adamowicz",
    author_email="lukas.adamowicz@pfizer.com",
    description="Additional Python statistics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/PfizerRD/pysit2stand",
    # download_url="https://pypi.org/project/pysit2stand/",
    # project_urls={
    #     "Documentation": "https://pysit2stand.readthedocs.io/en/latest/"
    # },
    # include_pacakge_data=True,
    # package_data={'pysit2stand': ['data/*.csv']},
    packages=setuptools.find_packages(),
    license='MIT',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.7"
    ],
)
