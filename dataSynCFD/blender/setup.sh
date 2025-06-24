# Install blender in linux server
wget https://mirror.clarkson.edu/blender/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz
tar -xf blender-3.6.0-linux-x64.tar.xz

# Install dependencies
sudo apt update
sudo apt install -y libxrender1 libx11-6 libxi6 libxcursor1 libxrandr2 libxfixes3 libxext6 libxkbcommon0 libsm6

# Install python blender api
# pip install bpy