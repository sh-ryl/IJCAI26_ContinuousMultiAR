# Re-import necessary libraries for data handling and plotting
import pandas as pd
import matplotlib.pyplot as plt
from constants import COLORS

# TODO create a function for this instead!

# File paths of CSVs
# cloth_stick
# root = "mod/ar_log/cloth_stick"
# ag_wc = range(1, 10, 2)
# ag_weight_combo = [
#     f'/cloth_{round(x/10,1)}_stick_{round(1-x/10,1)}' for x in ag_wc]
# ag_titles = [
#     f"cloth: {round(x/10,1)}, stick: {round(1-x/10,1)}" for x in ag_wc]
# ar_wc = range(1, 10, 2)
# ar_weight_combo = [
#     f"cloth: {round(x/10,1)}, stick: {round(1-x/10,1)}" for x in ar_wc]
# dkl_y = [0.9, 2.5]
# bi_y = [-0.05, 1.05]
# legend_loc = (0.84, 0.37)

# UVFA cloth_stick
# root = "mod/ar_log/cloth_stick"
# ag_wc = range(1, 10, 2)
# ag_weight_combo = [
#     f'/cloth_{round(x/10,1)}_stick_{round(1-x/10,1)}_uvfa' for x in ag_wc]
# ag_titles = [
#     f"cloth: {round(x/10,1)}, stick: {round(1-x/10,1)}" for x in ag_wc]
# ar_wc = range(1, 10, 1)
# ar_weight_combo = [
#     f"cloth: {round(x/10,1)}, stick: {round(1-x/10,1)}" for x in ar_wc]
# dkl_y = [0.9, 2.5]
# bi_y = [-0.05, 1.05]
# legend_loc = (0.84, 0.37)

# axe_bridge
# root = "mod/ar_log/axe_bridge"
# ag_wc = [100, 500]
# ag_weight_combo = [f'/axe_1_bridge_0.7_ability_{lv}_incentive' for lv in ag_wc]
# ag_titles = [f"level: {lv}" for lv in ag_wc]
# ar_wc = ag_wc
# ar_weight_combo = [f"level: {lv}" for lv in ag_wc]
# dkl_y = [0.9, 2.5]
# bi_y = [-0.05, 1.05]
# legend_loc = (0.37, 0.45)

# iron_wood_grass
# root = "mod/ar_log/iron_wood_grass"
# ag_wc = [(1, -1, ""), (-1, 1, ""), (1, -1, "_belief"), (-1, 1, "_belief")]
# ag_weight_combo = [f'/iron_0.7_wood_{w[0]}_grass_{w[1]}{w[2]}' for w in ag_wc]
# ag_titles = ["full obs with grass -1", "full obs with wood -1",
#              "part obs with grass -1", "part obs with wood -1"]
# ar_weight_combo = ["full obs with wood -1", "full obs with grass -1",
#                    "part obs with grass -1", "part obs with wood -1"]
# dkl_y = [0.5, 3.8]
# bi_y = [-0.05, 1.05]
# legend_loc = (0.68, 0.42)

# "axe_1_bridge_0.7"  # "cloth_1_stick_1_plank_1_rope_1"
root = "cloth_1_stick_1_plank_1_rope_1"
label = "dcprobsnoincent"  # "s4r1_redo"
# [x for x in range(100, 1200, int(((1200-100)/20)))]
# [x for x in range(100, 1200, 200)]
# [300, 500, 700]
# [100, 500]
wc = [x for x in range(100, 1200, 200)]
obs_wc = [x for x in range(100, 1200, int(((1200-100)/20)))]
fn = [
    f'mod/ar_log/{root}_{label}_{x}_20gt_1/{root}_ability_{x}_uvfa_avg_bi_prob' for x in wc]
bi_y = [-0.05, 1]
legend_loc = (0.63, 0.15)

# target
# target_file_1 = "_avg_dkl_zbc.csv"
target_file_2 = ".csv"

# Plotting 5 horizontal subplots
total_plot = len(wc)
fig, axs = plt.subplots(1, total_plot, figsize=(10, 5))

# Load the datasets
# ag_df_1 = [pd.read_csv(root+combo+target_file_1, header=None)
#            for combo in ag_weight_combo]
ag_df_2 = [pd.read_csv(f"{file}{target_file_2}", header=None)
           for file in fn]

# print(ag_df_2)
# input()

colors = ['blue', 'orange', 'green', 'red', 'purple']

# Plot each dataset on a separate subplot
# for row in range(2):
#     data = ag_df_1
#     if row == 1:
#         data = ag_df_2
#     for ag in range(len(ag_wc)):
#         for ar in range(len(ar_weight_combo)):
#             y = data[ag].iloc[:, ar].values
#             x = range(len(y))
#             axs[row, ag].plot(
#                 x, y, label=ar_weight_combo[ar])

#             if row == 0:
#                 axs[row, ag].set_title(ag_titles[ag], fontsize=12)
#                 axs[row, ag].set_ylim(dkl_y)
#                 if ag == 0:
#                     axs[row, ag].set_ylabel("Running Average DKL")
#             elif row == 1:
#                 axs[row, ag].set_ylim(bi_y)
#                 if ag == 0:
#                     axs[row, ag].set_ylabel("Bayesian Inference Prob.")

#             axs[row, ag].set_xlabel("timestep")

data = ag_df_2

for ag in range(len(wc)):  # each file = executing agent
    for ar in range(len(obs_wc)):  # each column in each file = observing agent
        y = data[ag].iloc[:, ar].values
        x = range(len(y))  # timesteps
        axs[ag].plot(x, y, label=obs_wc[ar], color=COLORS[ar])

        axs[ag].set_ylim(bi_y)
        axs[ag].set_ylabel("Bayesian Inference Prob.")

        axs[ag].set_xlabel("timestep")

# Clear the last subplot
for x in range(total_plot-len(wc)):
    axs[0, x+len(wc)].axis("off")
    axs[1, x+len(wc)].axis("off")

# Add a common legend outside all subplots
handles, labels = axs[ag].get_legend_handles_labels()
fig.legend(handles, labels, loc=legend_loc, title="AR Observer")

# Adjust layout
plt.tight_layout()
plt.show()
