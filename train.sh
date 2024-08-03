export PYTHONPATH="/home/yi/Documents/DELIVER"
python -m torch.distributed.launch --nproc_per_node=1 --use_env tools/train_mm.py --cfg configs/deliver_rgbdel.yaml