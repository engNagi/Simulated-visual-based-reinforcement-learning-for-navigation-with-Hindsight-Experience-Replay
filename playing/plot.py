
import matplotlib.pyplot as plt

bar_names = []
bar_heights = []


textfile = open("/Users/mohamednagi/Desktop/Master Thesis final /Master_Playing/step/steps.txt", "r")
for line in textfile:
	bar_name, bar_height = line.split(",")

	bar_names.append(bar_name)
	bar_heights.append(bar_height)

plt.bar(bar_names, bar_heights)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Agent trained only on images number of steps per Episode")

plt.show()