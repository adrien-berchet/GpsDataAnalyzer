from setuptools import find_packages, setup


classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: GIS",
    "Topic :: Scientific/Engineering",
    "Topic :: Utilities",
]


with open("requirements-install.pip") as f:
    req_base = f.read().splitlines()

with open("requirements-test.pip") as f:
    req_tests = f.read().splitlines()

with open("requirements-dev.pip") as f:
    req_dev = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="GpsDataAnalyzer",
    use_scm_version=True,
    description=(
        "A simple Python toolkit to analyze GPS data"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adrien Berchet",
    author_email="adrien.berchet@gmail.com",
    url="https://github.com/adrien-berchet/GpsDataAnalyzer",
    license="MIT",
    classifiers=classifiers,
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=False,
    install_requires=req_base,
    extras_require={
        "dev": req_dev + req_tests,
        "tests": req_tests
    },
    setup_requires=["setuptools_scm"],
    python_requires=">=3.6",
    zip_safe=False,
)
