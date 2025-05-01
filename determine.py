import matplotlib.pyplot as plt 

# Global variables
seconds = 0
x_data = []
y_data = []

# Setup the plot once
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')  # Use line style for time series
ax.set_xlabel("Time (Seconds)")
ax.set_ylabel("e[0]")
ax.set_title("Real-time Time Series of e[0]")
ax.set_xlim(0, 10)
ax.set_ylim(-5, 5) 

def then_c(e, y, d, p, c):

    global seconds, x_data, y_data, line, ax
    e_c=[0 if i<2.5 else 1 for i in e]

    print(c)
    print("***********")  

    if c == 30:
        print("#############################")
        x_data.append(seconds)
        y_data.append(e[0])
        seconds += 1

        line.set_data(x_data, y_data)

        # Scroll the x-axis like a time series
        if seconds > ax.get_xlim()[1]:
            ax.set_xlim(seconds - 10, seconds + 1)  

        plt.draw()
        plt.pause(0.01)  # Refresh the plot
