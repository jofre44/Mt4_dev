import setuptools

setuptools.setup(
    name="tools", # Replace with your own username
    version="0.0.0",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description="long_description",
    #long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=['tools'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)