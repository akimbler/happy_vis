# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

plt.style.use("dark_background")
with open(
    "data/sub-4001/ses-S1/func/sub-4001_ses-S1_task-rest_run-1_bold.json", "r"
) as rf:
    metadata = json.load(rf)
print(183 * metadata["RepetitionTime"])
data = {
    "cardiac": np.genfromtxt("data\happy_out\_cardfromfmri_25.0Hz.txt"),
    "normcardiac": np.genfromtxt("data\happy_out\_normcardfromfmri_25.0Hz.txt"),
    "normcardiacfiltered": np.genfromtxt(
        "data\happy_out\_normcardfromfmri_dlfiltered_25.0Hz.txt"
    ),
}
data = pd.DataFrame(data)
# bad_pts = pd.read_csv(r"D:\Documents\happy_test\_cardfromfmri_sliceres_badpts.txt", names=['bad'])
L = 3
x = np.linspace(0, L)
ncolors = len(plt.rcParams["axes.prop_cycle"])
shift = np.linspace(0, L, ncolors, endpoint=False)
print(shift)
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
seconds = np.arange(60, 75.02, 0.04)
sns.lineplot(x=seconds, y=data.loc[1500:1875, "cardiac"], ax=ax[0], color="lightpink")
sns.lineplot(
    x=seconds, y=data.loc[1500:1875, "normcardiac"], ax=ax[1], color="lavender"
)
sns.lineplot(
    x=seconds, y=data.loc[1500:1875, "normcardiacfiltered"], ax=ax[2], color="skyblue"
)
# ax[2].set_xticklabels([label.text for label in ax[2].get_xticklabels()])
plt.savefig("figures/cardiac_functions.png", dpi=300, bbox_inches="tight")

# %%
import nibabel as nb

print(
    nb.load(
        r"data\sub-4001\ses-S1\func\sub-4001_ses-S1_task-study_run-1_bold.nii.gz"
    ).shape
)

# %%
from nilearn.plotting import plot_epi, plot_anat
import nibabel as nb

vessel_map = nb.load(r"data\happy_out\_vesselmask.nii.gz")
functional = nb.load(
    r"data\sub-4001\ses-S1\func\sub-4001_ses-S1_task-study_run-1_bold.nii.gz"
)
g = plot_anat(functional.slicer[:, :, :, 0])
g.add_overlay(vessel_map)
# plot_epi(vessel_map)
# %%
import scipy.fft as fft
import matplotlib.pyplot as plt
import numpy as np

yf = fft.fft(data)
xf = fft.fftfreq(8050, 1 / 25)
# sns.lineplot(x=xf, y=yf)
plt.plot(xf, 2.0 / 8050 * np.abs(yf))
# %%
# %%
events = pd.read_csv(
    r"D:\Documents\Florida International University\EMU Study - Documents\EMU Data\behavior\pre_covid\task_files\raw\4001_emotionps_study_2018_Aug_11_0930_run1.txt",
    na_values="None", sep='\t'
)
from scipy import signal

data["ncdl_delta"] = data["normcardiacfiltered"] - data["normcardiacfiltered"].shift(1)
extremes = np.array(
    signal.argrelextrema(data["normcardiacfiltered"].values, np.greater)
)
data["extremes"] = data["normcardiacfiltered"].values
data["sqdif"] = np.nan
extremes = [x for x in data.index if x in extremes[0, :]]
non_extremes = [x for x in data.index if x not in extremes]
data.loc[non_extremes, "extremes"] = 0
sqdif = ((np.array(extremes) * 80) - (np.roll(np.array(extremes) * 80, -1)))
sqdif = sqdif - np.roll(sqdif, -1)
sqdif[0] = 0
# print(sqdif)
# print(np.array(extremes) * 80)
# print(np.roll(np.array(extremes) * 80, -1))
for idx, row in data.iterrows():
    if idx in extremes:
        data.loc[idx, "sqdif"] = sqdif[extremes.index(idx)] ** 2
data.ffill(inplace=True)
# data.loc[extremes, 'sqdif'] = (extremes * 80) - (np.roll(extremes, 1) * 80)
# rr_intervals[:, 0] = 0
# interval_bpm = (rr_intervals.shape[-1] * 80) / (322/60)
# print(np.sqrt(np.mean((rr_intervals - np.roll(rr_intervals, 1)) ** 2)))
# sns.lineplot(x=seconds, y=data.loc[1500:1875, 'normcardiacfiltered'])
seconds = np.arange(60, 75.02, 0.04)
valence_dict = {1: [], 2: [], 3: []}
sns.lineplot(x=seconds, y=data.loc[1500:1875, "sqdif"])
for valence, val_frame in events.groupby("Stimtype"):
    # print(val_frame.shape)
    for idx, row in val_frame.iterrows():
        interval = data.loc[round((row['StimOnset'] - events.loc[0, 'StimOnset']) * 25) : round((row['StimOnset'] - events.loc[0, 'StimOnset']) * 25) + 75, "sqdif"]
        print(interval)
        # print(interval)
        valence_dict[valence].append(
            np.sqrt(
                np.mean(
                    np.unique(interval)
                )
            )
        )
print(events)
