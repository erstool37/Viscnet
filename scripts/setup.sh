git pull origin main
pip install -r requirements.txt
apt update
apt install -y tmux
# vessl storage copy-file volume://vessl-storage/sph-20rpm-increment .
# tmux new-session -d -s slave1
# tmux attach -t slave1
# vessl storage copy-file volume://vessl-storage/sph-final .
# huggingface-cli login