These code are extracted from the original MeshworkAnalyzer repository.
https://github.com/KanchanawongLab/MeshworkAnalyzer

#### Dependences
- Matlab
- IDL (>= 8.2)

#### How to run
To run the code, IDL 8.2 or later shoul be installed.
I download IDL 8.5 from https://downloadlynet.ir/2020/18/6812/03/idl-envi/01/?#/6812-exelis-e-102551110318.html

The `meshworkanalyzer.pro` is the main entry point.
I do not use the `meshworkanalyzer.sav`.
I modified some code to make it work with my installed Matlab.

If you want to use the code, you should first repalce all the `Matlab_Application_23.2` in the `meshworkanalyzer.pro` with your Matlab version. You can find the Matlab version in the following way:


There may be some bugs in the original code, I fixed them as follows:
1.  All the 
    ```
    otsulevel = cgotsu_threshold(oftimage[nonzeroindex],nbins = 100)
    ```
    was replaced with 
    ```
    otsulevel = graythresh(oftimage[nonzeroindex])
    ```
2.  I add
    ```
    ; add current path ot matlab path @qiqilu
    thisFile = ROUTINE_FILEPATH()
    thisPath = FILE_DIRNAME(thisFile)
    cmd = "addpath('"+thisPath+"');"
    oMatlab.Execute, cmd
    ```
    to add the current path to matlab path.