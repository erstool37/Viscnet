cd slurry_viscnet
git pull origin main
pip install -r requirements.txt
apt update
apt install -y tmux
# vessl storage copy-file volume://vessl-storage/decay-visconly . 
# tmux new-session -d -s slave1
# tmux attach -t slave1