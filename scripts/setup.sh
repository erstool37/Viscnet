git pull origin main
pip install -r requirements.txt
apt update
apt install -y tmux
# vessl storage copy-file volume://vessl-storage/sph-realvisc-10rpm-diffback2 .
# vessl storage copy-file volume://vessl-storage/real-impeller-1000 .
# tmux new-session -d -s slave1
# tmux attach -t slave1
# huggingface-cli login