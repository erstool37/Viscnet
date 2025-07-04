@echo off

REM ====== Setup ======
pip install pandas pysplishsplash meshio pysplashsurf

REM Download Blender
REM curl -L -o blender.zip https://download.blender.org/release/Blender4.3/blender-4.3.2-windows-x64.zip
REM tar -xf blender.zip

REM Install Python packages using Blender’s Python
cd blender-4.3.2-windows-x64\4.3\python\bin
python.exe -m ensurepip --upgrade
python.exe -m pip install meshio numpy fileseq
cd ..\..\..\..

REM DOWNLOAD ADDONS
REM curl -L -o sequence_loader.zip https://github.com/InteractiveComputerGraphics/blender-sequence-loader/archive/refs/tags/v0.3.4.zip
REM tar -xf sequence_loader.zip
REM move blender-sequence-loader-0.3.4 blender-4.3.2-windows-x64\4.3\scripts\addons_core\sequence_loader

REM Cleanup
REM del blender.zip
REM del sequence_loader.zip

REM ====== VTK Builder (80 sec per file) ======
python jeeljil.py --start 0 --interval 10
REM python jeeljil.py --start 10 --interval 10
REM python jeeljil.py --start 20 --interval 10
REM python jeeljil.py --start 30 --interval 10
REM python jeeljil.py --start 40 --interval 10

REM ====== Mesh Builder ======
python surface.py

REM ====== Headless Render ======
REM conda create -y -n py37_env python=3.7
REM call conda activate py37_env
REM python --version
REM pip install bpy 

REM Create output folder if it doesn't exist
if not exist "videos" (
    mkdir "videos"
)

REM Loop through all folders in obj
for /D %%F in ("final_mesh\*") do (
    echo [JOB] %%F -> videos\%%~nxF.mp4
    call blender-4.3.2-windows-x64\blender.exe -b --python render.py -- "%%F" "videos\%%~nxF.mp4"
)

REM upload to vessl storage, or maybe retrieve them manually
REM pip install vessl
REM cd ../../videos
REM vessl copy-file . vessl-storage/{sph_com1}

navigate to video foldrer C:\\videos
