import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.dates as mdates

# Read in bikes.csv into a pandas dataframe
### Your code here
bikes = pd.read_csv("bikes.csv", index_col="bike_id")
bikes_status = (
    bikes["status"]
    .value_counts()
    .rename_axis("unique_values")
    .reset_index(name="counts")
)
bikes_status

# Read in DOX.csv into a pandas dataframe
# Be sure to parse the 'Date' column as a datetime
### Your code here
DOX = pd.read_csv("DOX.csv")
# Let's make sure 'date' is actually a date in pandas
DOX["Date"] = pd.to_datetime(DOX["Date"])

# Divide the figure into six subplots
# Divide the figure into subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Make a pie chart
### Your code here

patches, texts, pcts = axs[0][0].pie(
    bikes_status["counts"],
    labels=bikes_status["unique_values"],
    textprops={"color": "w"},
    autopct="%1.1f%%",
)
for i, patch in enumerate(patches):
    texts[i].set_color(patch.get_facecolor())
axs[0][0].set_title("Current Status")


# Make a histogram with quartile lines
# There should be 20 bins
### Your code here

axs[0][1].hist(bikes["purchase_price"], bins=20, histtype="step")
axs[0][1].set(
    xlabel="US Dollars", ylabel="Number of Bikes", title="Price Histogram (1000bikes)"
)
axs[0][1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("$%d"))
quants = [
    bikes["purchase_price"].quantile(0.0),
    bikes["purchase_price"].quantile(0.25),
    bikes["purchase_price"].quantile(0.50),
    bikes["purchase_price"].quantile(0.75),
    bikes["purchase_price"].quantile(1.0),
]
for k in quants:
    axs[0][1].axvline(k, linestyle="--", color="k")

axs[0][1].text(quants[0] + 8, 10, f"min: ${quants[0]: .0f}", size=12, rotation=90)
axs[0][1].text(quants[1] + 8, 10, f"25%: ${quants[1]: .0f}", size=12, rotation=90)
axs[0][1].text(quants[2] + 8, 10, f"50%: ${quants[2]: .0f}", size=12, rotation=90)
axs[0][1].text(quants[3] + 8, 10, f"75%: ${quants[3]: .0f}", size=12, rotation=90)
axs[0][1].text(quants[4] + 8, 10, f"max: ${quants[4]: ,.0f}", size=12, rotation=90)

# Make a scatter plot with a trend line
### Your code here
axs[1][0].scatter(bikes["purchase_price"], bikes["weight"], s=0.5)
# calculate equation for trendline
z = np.polyfit(bikes["purchase_price"], bikes["weight"], 1)
p = np.poly1d(z)
axs[1][0].plot(
    bikes["purchase_price"], p(bikes["purchase_price"]), color="red", linewidth=1
)
axs[1][0].set(xlabel="Price", ylabel="Weight", title="Price vs. Weight")
axs[1][0].xaxis.set_major_formatter("${x:1.0f}")
axs[1][0].yaxis.set_major_formatter("{x:1.0f} kg")


# Make a time-series plot
axs[1][1].plot(DOX["Date"], DOX["Close"])
axs[1][1].yaxis.set_major_formatter("${x:1.2f}")
axs[1][1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
axs[1][1].grid()
axs[1][1].set(title="DOX")

# Make a boxplot sorted so mean values are increasing
# Hide outliers
### Your code here


def boxplot_sorted(df, by, column, rot=0):
    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    grouped_bikes = pd.DataFrame({col: vals[column] for col, vals in df.groupby(by)})
    # find and sort the median values in this new dataframe
    sorted_median = grouped_bikes.median().sort_values()
    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    return grouped_bikes[sorted_median.index].boxplot(
        rot=rot, return_type="axes", ax=axs[2][0], showfliers=False
    )


axs[2][0] = boxplot_sorted(bikes, by=["brand"], column="purchase_price")
axs[2][0].set_title("Brand vs. Price")
axs[2][0].yaxis.set_major_formatter("${x:1.0f}")


# Make a violin plot
### Your code here
grouped = bikes.groupby(["brand"])
df2 = pd.DataFrame({col: vals["purchase_price"] for col, vals in grouped})
sorted_index_mean = df2.mean().sort_values().index.to_list()
axs[2][1].violinplot(
    dataset=[
        bikes.groupby(["brand"])["purchase_price"].apply(list)[idx]
        for idx in sorted_index_mean
    ],
    showmeans=True,
)
axs[2][1].set_title("Brand vs. Price")
axs[2][1].yaxis.set_major_formatter("${x:1.0f}")
axs[2][1].set_xticklabels([""] + sorted_index_mean)

# Create some space between subplots
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Write out the plots as an image
plt.tight_layout()
plt.savefig("plots.png")


from sklearn.linear_model import LinearRegression

df = bikes
# Get data as numpy arrays
X = df["purchase_price"].values.reshape(-1, 1)
y = df["weight"].values.reshape(-1, 1)
# Do linear regression
reg = LinearRegression()
reg.fit(X, y)
# Get the parameters
slope = reg.coef_[0]
intercept = reg.intercept_
print(f"Slope: {slope}, Intercept: {intercept}")
