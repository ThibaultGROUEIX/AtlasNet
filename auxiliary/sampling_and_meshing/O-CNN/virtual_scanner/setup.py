from skbuild import setup

setup(
    name="ocnn.virtualscanner",
    version="18.09.05",
    description="Virtual scanner utilities",
    author='Microsoft',
    author_email="dapisani@microsoft.com",
    packages=['ocnn', 'ocnn.virtualscanner'],
    zip_safe=False,
    install_requires=['Cython', 'pyyaml'],
    package_dir={'': 'python'},
    package_data={'ocnn.virtualscanner': ['*.pxd']}
)
