#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyzeDf(name, df):
    print(f"{name} analysis ================")
    times = df['time'].unique()
    kp_updates = df['last_frame_update'].unique()
    execution_time = times[1] - times[0]
    print(f"execution time took: {int(execution_time)}sec")
    intervals = [(v if i == 0 else v - kp_updates[i - 1]) for i, v in enumerate(kp_updates)]
    print(f"there were {len(kp_updates)} keypoints updates, on average every {round(np.mean(intervals), 2)} frames :\n{kp_updates}")
    plt.plot(df['corners'])
    plt.plot(df['clusters'])


# %%
orb_df = pd.read_csv('output/ORB.csv')
analyzeDf("ORB", orb_df.tail(orb_df.shape[0] -1))
# %%
sift_df = pd.read_csv('output/SIFT.csv')
analyzeDf("SIFT", sift_df.tail(sift_df.shape[0] -1))
# %%
fast_df = pd.read_csv('output/FAST.csv')
analyzeDf("FAST", fast_df.tail(fast_df.shape[0] -1))
