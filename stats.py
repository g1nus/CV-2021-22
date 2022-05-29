#%%
import pandas as pd
import numpy as np
import matplotlib

orb_df = pd.read_csv('output/ORB.csv')
sift_df = pd.read_csv('output/SIFT.csv')

def analyzeDf(name, df):
    print(f"{name} analysis ================")
    times = df['time'].unique()
    kp_updates = df['last_frame_update'].unique()
    execution_time = times[1] - times[0]
    print(f"execution time took: {int(execution_time)}sec")
    intervals = [(v if i == 0 else v - kp_updates[i - 1]) for i, v in enumerate(kp_updates)]
    print(f"there were {len(kp_updates)} keypoints updates, on average every {round(np.mean(intervals), 2)} frames :\n{kp_updates}")
    df.plot(x="frame", y="corners")



analyzeDf("ORB", orb_df)
# %%
analyzeDf("SIFT", sift_df)
# %%
