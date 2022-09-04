# Feedback Prize 2 Solution

1. Create a container from docker image: pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
2. Run sh install.sh
3. Download competition data and extract it in the data folder
4. Pretrain deberta-large using our Feedback Prize 1 code (mtask_v2)
    https://github.com/neroksi/fprize_final_cleanup
    or just take the deberta-large weights from https://www.kaggle.com/code/kneroma/gdrive-db1l-1024-v2-v11-no-pe-weights like we did
    Weights need to be saved to pt_large and named foldX.pth (e.g. fold2.pth)
5. Run sh run_me.sh
