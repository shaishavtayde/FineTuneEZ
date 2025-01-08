#!/usr/bin/env python
from setuptools import setup, find_packages

requirements = []

setup_requirements = []

test_requirements = ['pytest>=3']


setup(
    author='Shaishav Tayde',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: Phase1',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="FineTune EZ",
    entry_points={
        'console_scripts': [
            'train-hf=finetune_ez.training_hf:_run_train_model_with_hf',
        ],
    },
    install_requires=requirements,
    license="LICENSE",
    keywords='finetune_ez',
    name='finetune_ez',
    packages=find_packages(where="./src"),
    package_dir={"": "src"},
    package_data={'finetune_ez': ['data/*']},
    include_package_data=True,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require = {
    "dev" : [ 
            "pip",
            "wheel",
            "pylint",
            "coverage",
            "Sphinx",
            "twine",
            "pytest",
            "coverage",
            "jupyter"
            ],
    },
    zip_safe=False,
)
