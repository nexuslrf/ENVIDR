cd freqencoder/; rm -rf build; python setup.py build_ext --inplace; mkdir _ext; mv *.so _ext; cd ..;
cd gridencoder/; rm -rf build; python setup.py build_ext --inplace; mkdir _ext; mv *.so _ext; cd ..;
cd hashencoder/; rm -rf build; python setup.py build_ext --inplace; mkdir _ext; mv *.so _ext; cd ..;
cd shencoder/; rm -rf build; python setup.py build_ext --inplace; mkdir _ext; mv *.so _ext; cd ..;
cd raymarching/; rm -rf build; python setup.py build_ext --inplace; mkdir _ext; mv *.so _ext; cd ..;