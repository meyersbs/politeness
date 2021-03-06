from setuptools import setup

setup(
        name='politeness',
        version='0.1.2',
        packages=[
            'politeness', 'politeness.data'
        ],
        package_data={
            'politeness': ['models/*.p', 'features/*', 'corenlp-url.txt'],
            'politeness.data': ['*.json']
        },
        install_requires=[
            'numpy==1.12.1',
            'scipy==0.19.0',
            'scikit-learn==0.18.1',
            'nltk==3.2.2',
        ],
        license='The MIT License (MIT) Copyright (c) 2017 Benjamin S. Meyers',
        description='A port of the Stanford Politeness API to Python3.'
                    'Reorganized to function as a library and on the command'
                    'line.',
        author='Benjamin S. Meyers',
        author_email='bsm9339@rit.edu',
        url='https://github.com/meyersbs/politeness',
        test_suite='politeness.tests',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Natural Language :: English',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Text Processing :: Linguistic'
        ],
        keywords=[
            'nlp', 'natural language', 'natural language processing',
            'politeness', 'linguistic politeness', 'classification'
        ]
    )
