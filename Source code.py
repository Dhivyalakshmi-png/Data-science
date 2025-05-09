import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 📥 Load Dataset
df = pd.read_csv("stock_price_time_series.csv")
# 🗓️ Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"])
# 📊 Sort values by Company and Date
df = df.sort_values(by=["Company", "Date"]).reset_index(drop=True)
# 🧠 Data Information
print("🔍 Dataset Info:")
print(df.info())
# ❓ Check for Missing Values
print("\n🧼 Missing Values:")
print(df.isnull().sum())
# ❗ Check for Duplicate Rows
print("\n📄 Duplicate Rows:", df.duplicated().sum())
# 👀 Preview First 5 Rows
print("\n📝 First 5 Records:")
print(df.head())
# 📚 Required Libraries
import matplotlib.pyplot as plt
import seaborn as sns
# 🧪 Filter for one company (example: Company_1)
company_df = df[df["Company"] == "Company_1"]
# 🎨 Set plot style
sns.set(style="whitegrid")
# 📈 1. Line Plot: Closing Price over Time
plt.figure(figsize=(14, 5))
plt.plot(company_df["Date"], company_df["Close"], label="Closing Price", color='blue')
plt.title("📈 Closing Stock Price Over Time - Company_1")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()
# 📉 2. Line Plot: Volume over Time
plt.figure(figsize=(14, 5))
plt.plot(company_df["Date"], company_df["Volume"], label="Volume", color='orange')
plt.title("📊 Trading Volume Over Time - Company_1")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.tight_layout()
plt.show()
# 🧾 3. Histogram: Distribution of Closing Prices
plt.figure(figsize=(10, 4))
sns.histplot(company_df["Close"], bins=50, kde=True, color="skyblue")
plt.title("📊 Distribution of Closing Prices - Company_1")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
# 🔥 4. Heatmap: Correlation Matrix
plt.figure(figsize=(6, 4))
# Check available columns in company_df
print(company_df.columns)
# Select existing columns for correlation
corr_matrix = company_df[["Close", "Volume"]].corr()  # Only 'Close' and 'Volume' are available
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🔥 Feature Correlation Matrix - Company_1")
plt.tight_layout()
plt.show()
👣 Step 1: Ensure 'Date' column is in datetime format
if df["Date"].dtype != 'datetime64[ns]':
    df["Date"] = pd.to_datetime(df["Date"])

# 👣 Step 2: Set 'Date' as index (do only once)
if df.index.name != "Date":
    df.set_index("Date", inplace=True)

# 👣 Step 3: Filter for one company (if not already done)
if "company_df" not in locals():
    company_df = df[df["Company"] == "Company_1"].copy()

# 👣 Step 4: Check index is datetime (for safety)
if not pd.api.types.is_datetime64_any_dtype(company_df.index):
    company_df.index = pd.to_datetime(company_df.index)

# ✅ Step 5: Create time-based features
company_df["Year"] = company_df.index.year
company_df["Month"] = company_df.index.month
company_df["Day"] = company_df.index.day
company_df["Weekday"] = company_df.index.weekday

# ✅ Step 6: Create lag features
company_df["Lag_1"] = company_df["Close"].shift(1)
company_df["Lag_7"] = company_df["Close"].shift(7)

# ✅ Step 7: Create rolling averages
company_df["Rolling_7"] = company_df["Close"].rolling(window=7).mean()
company_df["Rolling_30"] = company_df["Close"].rolling(window=30).mean()

# ✅ Step 8: Drop rows with missing values (caused by lag/rolling)
company_df.dropna(inplace=True)

# 🖥️ Preview result
print(company_df.head())
# 📦 Install Prophet if not installed (uncomment below in Colab or local)
# !pip install prophet

# 📚 Import Prophet
from prophet import Prophet

# 🧹 Prepare DataFrame for Prophet: requires 'ds' and 'y' columns
prophet_df = company_df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

# 📈 Initialize and fit the model
model = Prophet(daily_seasonality=True)
model.fit(prophet_df)
# ⏩ Create future dates (next 60 days)
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)
# 📊 Plot forecast
fig1 = model.plot(forecast)
plt.title("📈 Prophet Forecast for Stock Price - Company_1")
plt.show()
# 📉 Plot forecast components (trend, weekly, yearly)
fig2 = model.plot_components(forecast)
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 📌 Keep only the last 60 days of real values
actual = prophet_df.tail(60).reset_index(drop=True)
predicted = forecast[forecast["ds"].isin(actual["ds"])][["ds", "yhat"]].reset_index(drop=True)

# 🎯 Error Metrics
mae = mean_absolute_error(actual["y"], predicted["yhat"])
rmse = np.sqrt(mean_squared_error(actual["y"], predicted["yhat"]))

print(f"📏 MAE: {mae:.2f}")
print(f"📏 RMSE: {rmse:.2f}")

# 📊 Plot
plt.figure(figsize=(10, 5))
plt.plot(actual["ds"], actual["y"], label="Actual", marker='o')
plt.plot(predicted["ds"], predicted["yhat"], label="Predicted", marker='x')
plt.title("📊 Actual vs Predicted Close Price (Last 60 Days)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
