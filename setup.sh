git clone https://github.com/karpathy/nanoGPT
git clone https://github.com/karpathy/nanoGPT nanoGPTSubmission
/usr/bin/python3 -m pip install --upgrade pip
/usr/bin/python3 -m pip install -r requirements.txt
cd nanoGPT
/usr/bin/python3 data/shakespeare_char/prepare.py
/usr/bin/python3 train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=1 --lr_decay_iters=2000 --dropout=0.0
cd ..
cd nanoGPTSubmission
/usr/bin/python3 data/shakespeare_char/prepare.py
/usr/bin/python3 train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=1 --lr_decay_iters=2000 --dropout=0.0
