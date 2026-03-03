#   "The original CSV had 33,138 rows. After reindexing to strict 30-min steps
#    we got 37,730 rows — meaning 4,592 timestamps were missing and gap-filled."

original_rows  = 33138   # rows in DS_DF_WQI_Train.csv
reindexed_rows = len(df) # rows after asfreq('30min')
gap_filled     = reindexed_rows - original_rows

print(f'Original CSV rows  : {original_rows:,}')
print(f'After reindexing   : {reindexed_rows:,}')
print(f'Gap-filled rows    : {gap_filled:,}  ({100*gap_filled/reindexed_rows:.1f}% of total)')

# Show the gap visually — compare original data count per month vs expected
df['month'] = df.index.month
actual_per_month   = df.groupby('month').size()
# Expected = 30-min steps per month (approx), actual readings from original file
original_df        = pd.read_csv('DS_DF_WQI_Train.csv', index_col=0, parse_dates=True)
original_df['month'] = original_df.index.month
original_per_month = original_df.groupby('month').size()
gap_per_month      = actual_per_month - original_per_month

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('Data Quality — Gap-filled Rows', fontsize=13, fontweight='bold')

# Left: bar showing gap count per month
ax = axes[0]
ax.bar(gap_per_month.index, gap_per_month.values, color='#e74c3c', alpha=0.8)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylabel('Gap-filled rows')
ax.set_title('Gap-filled Rows per Month')

# Right: stacked bar — original vs gap-filled per month
ax2 = axes[1]
ax2.bar(original_per_month.index, original_per_month.values,
        color='#3498db', alpha=0.85, label='Original rows')
ax2.bar(gap_per_month.index, gap_per_month.values,
        bottom=original_per_month.values,
        color='#e74c3c', alpha=0.75, label=f'Gap-filled ({gap_filled:,})')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(MONTH_LABELS)
ax2.set_ylabel('Row count')
ax2.set_title('Original vs Gap-filled Rows per Month')
ax2.legend()

plt.tight_layout()
plt.show()
