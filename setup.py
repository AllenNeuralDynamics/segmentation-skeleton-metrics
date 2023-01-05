from setuptools import setup

if __name__ == "__main__":

    setup(
        name='seg_eval',
        version='0.0.1',    
        description='Python package for evaluating neuron segmentations in terms of the number of splits and merges',
        author='Anna Grim',
        author_email='anna.grim@alleninstitute.org',
        license='MIT',
        packages=['seg_eval'],
    )
