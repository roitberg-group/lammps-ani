# import cudf
# df = cudf.read_csv('analyze/2023-10-13-163952.474802.0.3.csv')
# df['atom_count'] = df['formula'].str.len()

import re
import pandas as pd
df = pd.read_csv('analyze/2023-08-02-035928.488223.csv')

def total_atom_count_efficient(formula):
    return sum(int(count) if count else 1 for _, count in re.findall(r'([A-Z][a-z]*)(\d*)', formula))


df['atom_count'] = df['formula'].map(total_atom_count_efficient)

df.sort_values("atom_count").iloc[-50:]
atom_count_np_array = df.atom_count.to_numpy()


# Set the font to Times New Roman
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.hist(atom_count_np_array, bins=20, edgecolor='black')
ax.set_title('Histogram of Fragment Size')
ax.set_xlabel('Number of Atoms')
ax.set_ylabel('Frequency')
fig.savefig('atom_count_histogram_small.png')
ax.set_yscale('log')
fig.savefig('atom_count_histogram_small_log.png')

# grouped_df = df.groupby('time').agg({'atom_count': 'mean'}).reset_index()
# grouped_df.rename(columns={'atom_count': 'average_atom_count'}, inplace=True)
# grouped_df
#time_np_arraytime_np_array = grouped_df['time'].to_numpy()
#average_atom_count_np_array = grouped_df['average_atom_count'].to_numpy()
#fig, ax = plt.subplots()
#ax.plot(time_np_array, average_atom_count_np_array, marker='o')
#ax.plot(time_np_arraytime_np_array, average_atom_count_np_array, marker='o')
#ax.set_title('Average Atom Count Over Time')
#ax.set_xlabel('Time')
#ax.set_ylabel('Average Atom Count')
#fig.savefig('average_atom_count_over_time.png')
