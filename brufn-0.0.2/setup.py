import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt', "r") as req:
    required_pk = [x[:-1] for x in req.readlines()]

setuptools.setup(
    name="brufn",
    version="0.0.2",
    author="Fernando Raverta",
    author_email="fdraverta@gmail.com",
    description="Best Routing Under Uncertainties",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    data_files=[('', ['requirements.txt'])],
    install_requires=required_pk,
)
