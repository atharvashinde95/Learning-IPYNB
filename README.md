# Learning-IPYNB
#Simple bar chart of WQI by month (replaces the text bar)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(monthly.index, monthly['WQI'], color='#3498db', alpha=0.85, edgecolor='white')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylabel('Median WQI')
ax.set_title('WQI Median by Month', fontweight='bold')
for i, v in enumerate(monthly['WQI']):
    ax.text(i + 1, v + 0.3, f'{v:.1f}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()
