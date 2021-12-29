# Dataset

Each subdirectory is named after a software system :
- the *nodejs* directory contains the data measured on *nodeJS*
- the *poppler* directory contains the data measured onn *poppler*
- the *x264* directory contains the data measured on *x264*
- the *xz* directory contains the data measured on *xz*

Once in the subdirectory of a software system, it is organized as follows:
- Each compile-time configuration has its own directory (E.G. for nodejs the directory *1* is related to the first 
compile-time configuration of nodejs,  the directory *30* is related to the 30th compile-time configuration of nodejs). 
The list of compile-time configurations can be found in the *ctime_options.csv* file
- The default directory gathers the measurements using the default compile-time configuration
- These directories (numbers & default) contain the same list of *.csv* files, named after the different inputs fed to 
the software system (E.G. for nodejs, buffer1.csv is related to the script buffer1.js processed by nodeJS). These *.csv*
files are tables of raw data; each line is a run-time configuration, coming with the performance properties measured on 
the system e.g. for nodejs, `jitless`, `experimental-wasm-modules` and `experimental-vm-modules` are three different 
configuration options, while `ops` is a performance property

### Credit

Credit for this dataset goes to https://github.com/llesoil/ctime_opt