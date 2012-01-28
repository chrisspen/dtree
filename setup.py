from distutils.core import setup
import dtree
setup(name='dtree',
    version=dtree.__version__,
    description='A simple pure-Python batch decision tree contruction algorithm.',
    author='Chris Spencer',
    author_email='chrisspen@gmail.com',
    url='https://github.com/chrisspen/dtree',
    license='LGPL',
    py_modules=['dtree'],
    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    platforms=['OS Independent'],)