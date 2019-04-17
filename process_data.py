# This code aims at putting the dataset into the sharedMemory, 
# so that it won't take a long time to load the data from memory ( ~30s)
import numpy as np
import SharedArray as sa
import time

filename = "./data/train_x_lpd_5_phr.npz"

# main
def main():
    # try to load the data into the 
    try:
        a = sa.attach("train_x_lpd_5_phr")
        print("[info ] dataset already loaded")
    except:
        print("[info ] dataset not loaded")
        t0 = time.time()
        with np.load(filename) as f:
            data = np.zeros(f['shape'], np.uint8)
            data[[x for x in f['nonzero']]] = 1
            data = data.transpose((0, 4, 1, 2, 3))
            #np.random.shuffle(data)
            sa_data = sa.create("train_x_lpd_5_phr", data.shape, data.dtype)
            np.copyto(sa_data, data)
        t1 = time.time()
        print("[info ] The time costed:", t1 - t0, "s")

if __name__ == '__main__':
    print("[state] Program begin")
    main()
    print("[state] Program end")