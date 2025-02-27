import numpy as np
import matplotlib.pyplot as plt

# Define the categories and their values
categories = ["A", "B", "C", "D", "E"]
values = [4, 3, 2, 5, 4]

# Different ranges for each axis
ranges = {"A": [0, 50], "B": [0, 5], "C": [0, 3], "D": [0, 7], "E": [0, 60]}

# Number of categories
num_vars = len(categories)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The radar chart is a circular plot, so we need to "close the loop"
values += values[:1]
angles += angles[:1]

# Create the radar plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Plot the data
ax.plot(angles, values, linewidth=2, linestyle="solid", label="Data")

# Fill the area
ax.fill(angles, values, color="blue", alpha=0.25)

# Add category labels
ax.set_yticklabels([])  # Hide radial ticks
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Set different ranges for each axis
for i, category in enumerate(categories):
    ax.set_ylim(ranges[category][0], ranges[category][1])

# Set title
ax.set_title("Radar Chart with Different Ranges")

# Display the plot
plt.show()
