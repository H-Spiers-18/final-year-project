# Data

The directory gathers measurements made on the different software systems tested in the experimental protocol of our submission.

Each sub-directory is named after a software system :
- the *nodejs* directory contains the data measured on *nodeJS*
- the *poppler* directory contains the data measured onn *poppler*
- the *x264* directory contains the data measured on *x264*
- the *xz* directory contains the data measured on *xz*

Once in a sub-directory, it is organized as follows:
- Each comple-time configuration has its own directory e.g. for nodejs the directory *1* is related to the first compile-time configuration of nodejs,  the directory *30* is related to the 30th compile-time configuration of nodejs. The list of compile-time configurations can be consulted in the *ctime_options.csv* file
- The default directory gathers the measurements related to the default compile-time configuration i.e. we do not add any argument after the `./configure` command
- These directories (numbers & default) contain the same list of *.csv* files, named after the different inputs fed to the software system e.g. for nodejs, buffer1.csv is related to the script buffer1.js processed by nodejs. These *.csv* files are tables of raw data; each line is a run-time configuration, coming with the performance properties measured on the system e.g. for nodejs, `jitless`, `experimental-wasm-modules` and `experimental-vm-modules` are three different configuration options , while `ops` is a performance property. The original inputs (like buffer1.js in the first example) can be found directly in the docker used to measure the performances of software systems. 
