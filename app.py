import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# Set the browser tab title and make the dashboard use the full page width
st.set_page_config(page_title="Clinic Demand Dashboard", layout="wide")

# Main dashboard title
st.title("Physio Clinic Demand Dashboard")

# Allows user to upload the Antibex appointment CSV file
uploaded_file = st.file_uploader("Upload Patient Scheduled Appts CSV", type=["csv"])

# Only run the dashboard after a file has been uploaded
if uploaded_file:
    # Read the uploaded CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Rename unclear Antibex column names into readable names
    df = df.rename(columns={
        "TextBoxDTL2": "appointment_date",
        "Text2": "start_time",
        "Text3": "end_time",
        "Text1": "status",
        "Text4": "appointment_type",
        "Text5": "patient_name"
    })

    # Convert appointment date into real datetime format
    df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")

    # Combine appointment date and start time into one full datetime column
    df["start_datetime"] = pd.to_datetime(
        df["appointment_date"].dt.strftime("%Y-%m-%d") + " " + df["start_time"],
        errors="coerce"
    )

    # Remove rows where date/time conversion failed
    df = df.dropna(subset=["appointment_date", "start_datetime"])

    # Create useful time-based columns for analysis
    df["year"] = df["appointment_date"].dt.year
    df["month"] = df["appointment_date"].dt.month_name()
    df["month_number"] = df["appointment_date"].dt.month
    df["day_of_week"] = df["appointment_date"].dt.day_name()
    df["hour"] = df["start_datetime"].dt.hour
    df["hour_label"] = df["start_datetime"].dt.strftime("%I %p")
    # Sidebar filters
    st.sidebar.header("Filters")

    selected_years = st.sidebar.multiselect(
        "Select Year(s)",
        options=sorted(df["year"].unique()),
        default=sorted(df["year"].unique())
    )

    selected_appointment_types = st.sidebar.multiselect(
        "Select Appointment Type(s)",
        options=sorted(df["appointment_type"].dropna().unique()),
        default=sorted(df["appointment_type"].dropna().unique())
    )

    # Apply filters to the dataframe
    df = df[
        (df["year"].isin(selected_years)) &
        (df["appointment_type"].isin(selected_appointment_types))
        ]
    # Stop app if filters return no rows
    if df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    # Overview metrics section
    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)

    # Basic KPI cards
    col1.metric("Total Appointments", len(df))
    col2.metric("Years of Data", df["year"].nunique())
    col3.metric("Appointment Types", df["appointment_type"].nunique())

    # Monthly appointment demand section
    st.subheader("Appointments by Month")

    # Group appointments by year/month and count appointments
    monthly = df.groupby(["year", "month_number", "month"]).size().reset_index(name="appointments")

    # Create a proper date column for plotting the monthly trend
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month_number"].astype(str) + "-01"
    )

    # Line chart showing monthly demand over time
    fig_month = px.line(
        monthly,
        x="date",
        y="appointments",
        markers=True,
        title="Monthly Appointment Demand",
        labels={
            "date":"Month",
            "appointments":"Appointments"
        }
    )
    st.plotly_chart(fig_month, use_container_width=True)

    # Yearly growth section
    st.subheader("Appointments by Year")

    yearly = (
        df.groupby("year")
        .size()
        .reset_index(name="appointments")
    )

    fig_year = px.bar(
        yearly,
        x="year",
        y="appointments",
        title="Total Appointments by Year",
    labels = {
        "year": "Year",
        "appointments": "Appointments"
    }
    )

    st.plotly_chart(fig_year, use_container_width=True)
    yearly["growth_percent"] = yearly["appointments"].pct_change() * 100
    st.info("Note: The latest year may be incomplete, so year-over-year growth should be interpreted carefully.")

    # Year over Year Growth Summary
    st.subheader("Year-over-Year Growth")
    yearly_display = yearly.rename(columns={
        "year":"Year",
        "appointments":"Appointments",
        "growth_percent":"Growth Percent"
    })
    yearly_display["Growth Percent"] = yearly_display["Growth Percent"].round(2)
    st.dataframe(yearly_display, hide_index=True)


    # Day-of-week appointment demand section
    st.subheader("Appointments by Day of Week")

    # Keeps days in normal calendar order instead of random/count order
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Count appointments for each weekday
    day_counts = df["day_of_week"].value_counts().reindex(day_order).reset_index()
    day_counts.columns = ["day_of_week", "appointments"]

    # Bar chart showing which weekdays are busiest
    fig_day = px.bar(
        day_counts,
        x="day_of_week",
        y="appointments",
        title="Appointments by Day of Week",
        labels={
            "day_of_week":"Day of Week",
            "appointments":"Appointments"
        }
    )
    st.plotly_chart(fig_day, use_container_width=True)

    # Hourly demand section
    st.subheader("Appointments by Hour")

    # Count appointments by start hour
    hour_counts = (
        df.groupby(["hour", "hour_label"])
        .size()
        .reset_index(name="appointments")
        .sort_values("hour")
    )


    # Bar chart showing busiest appointment hours
    fig_hour = px.bar(
        hour_counts,
        x="hour_label",
        y="appointments",
        title="Appointments by Start Hour",
    labels = {
        "hour_label": "Time",
        "appointments": "Appointments"
    }
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # Heatmap section
    st.subheader("Day × Hour Demand Heatmap")

    # Count appointments for every day/hour combination
    heatmap_data = (
        df.groupby(["day_of_week", "hour", "hour_label"])
        .size()
        .reset_index(name="appointments")
        .sort_values("hour")
    )

    # Create an ordered list of hour labels so the heatmap stays in correct time order
    hour_order = (
        df[["hour", "hour_label"]]
        .drop_duplicates()
        .sort_values("hour")["hour_label"]
        .tolist()
    )

    # Convert grouped data into a matrix for the heatmap
    heatmap_pivot = heatmap_data.pivot(
        index="day_of_week",
        columns="hour_label",
        values="appointments"
    ).reindex(index=day_order, columns=hour_order)

    # Heatmap showing demand intensity by weekday and hour
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour", y="Day", color="Appointments"),
        title="Demand Heatmap by Day and Hour",
        aspect="auto"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Appointment type breakdown section
    st.subheader("Appointment Type Breakdown")

    # Count each appointment type
    type_counts = df["appointment_type"].value_counts().reset_index()
    type_counts.columns = ["appointment_type", "appointments"]

    # Pie chart showing appointment type distribution
    fig_type = px.pie(
        type_counts,
        names="appointment_type",
        values="appointments",
        title="Appointment Type Distribution"
    )
    st.plotly_chart(fig_type, use_container_width=True)

    st.subheader("Busiest and Slowest Time Slots")

    slot_summary = (
        df.groupby(["day_of_week", "hour", "hour_label"])
        .size()
        .reset_index(name="appointments")
        .sort_values("hour")
    )

    busiest_slots = slot_summary.sort_values("appointments", ascending=False).head(10)
    slowest_slots = slot_summary.sort_values("appointments", ascending=True).head(10)

    busiest_slots = busiest_slots[["day_of_week", "hour_label", "appointments"]].rename(columns={
        "day_of_week":"Day of Week", "hour_label":"Time", "appointments":"Appointments"
    })
    slowest_slots = slowest_slots[["day_of_week", "hour_label", "appointments"]].rename(columns={
        "day_of_week":"Day of Week", "hour_label":"Time", "appointments":"Appointments"
    })

    col1, col2 = st.columns(2)

    with col1:
        st.write("Highest Historical Demand Slots")
        st.dataframe(busiest_slots, hide_index=True)

    with col2:
        st.write("Lowest Historical Demand Slots")
        st.dataframe(slowest_slots, hide_index=True)

    # Average historical demand by day section
    st.subheader("Average Historical Demand by Day")

    # Count total appointments per actual date
    daily_counts = (
        df.groupby(["appointment_date", "day_of_week"])
        .size()
        .reset_index(name="appointments")
    )

    # Average appointments by day of week
    average_day_demand = (
        daily_counts.groupby("day_of_week")["appointments"]
        .mean()
        .reset_index(name="average_appointments")
    )

    # Keep days in calendar order
    average_day_demand["day_of_week"] = pd.Categorical(
        average_day_demand["day_of_week"],
        categories=day_order,
        ordered=True
    )

    average_day_demand = average_day_demand.sort_values("day_of_week")

    # Clean column names
    average_day_demand_display = average_day_demand.rename(columns={
        "day_of_week": "Day of Week",
        "average_appointments": "Average Appointments per Day"
    })

    average_day_demand_display["Average Appointments per Day"] = (
        average_day_demand_display["Average Appointments per Day"].round(2)
    )

    st.dataframe(average_day_demand_display, hide_index=True)

    # Daily forecasting model section
    st.subheader("Daily Appointment Forecasting Model")

    # Create daily demand dataset
    daily_forecast_data = (
        df.groupby("appointment_date")
        .size()
        .reset_index(name="appointments")
    )

    # Add date-based features
    daily_forecast_data["year"] = daily_forecast_data["appointment_date"].dt.year
    daily_forecast_data["month_number"] = daily_forecast_data["appointment_date"].dt.month
    daily_forecast_data["day"] = daily_forecast_data["appointment_date"].dt.day
    daily_forecast_data["day_of_week"] = daily_forecast_data["appointment_date"].dt.day_name()
    daily_forecast_data["week_of_year"] = daily_forecast_data["appointment_date"].dt.isocalendar().week.astype(int)

    # Sort chronologically
    daily_forecast_data = daily_forecast_data.sort_values("appointment_date").reset_index(drop=True)

    # Features for the model
    daily_features = [
        "year",
        "month_number",
        "day",
        "week_of_year",
        "day_of_week"
    ]

    # Convert day_of_week into numerical dummy columns
    daily_model_data = pd.get_dummies(
        daily_forecast_data[daily_features + ["appointments"]],
        columns=["day_of_week"]
    )

    X_daily = daily_model_data.drop(columns=["appointments"])
    y_daily = daily_model_data["appointments"]

    if len(daily_model_data) > 30:
        # Chronological train/test split
        split_index = int(len(daily_model_data) * 0.8)

        X_train_daily = X_daily.iloc[:split_index]
        X_test_daily = X_daily.iloc[split_index:]

        y_train_daily = y_daily.iloc[:split_index]
        y_test_daily = y_daily.iloc[split_index:]

        train_daily_data = daily_forecast_data.iloc[:split_index].copy()
        test_daily_data = daily_forecast_data.iloc[split_index:].copy()

        # Train Random Forest model
        daily_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=3
        )

        daily_model.fit(X_train_daily, y_train_daily)

        daily_predictions = daily_model.predict(X_test_daily)

        # Baseline: average appointments by day of week
        baseline_by_day = train_daily_data.groupby("day_of_week")["appointments"].mean()
        global_baseline = train_daily_data["appointments"].mean()

        baseline_predictions = (
            test_daily_data["day_of_week"]
            .map(baseline_by_day)
            .fillna(global_baseline)
        )

        # Metrics
        model_mae = mean_absolute_error(
            y_test_daily,
            daily_predictions
        )  # Average model error in appointments

        baseline_mae = mean_absolute_error(
            y_test_daily,
            baseline_predictions
        )  # Average error from simple weekday averages

        model_bias = (
                daily_predictions - y_test_daily
        ).mean()  # Positive = overpredicting, negative = underpredicting

        col1, col2, col3 = st.columns(3)

        col1.metric("Model MAE", round(model_mae, 2))
        col2.metric("Baseline MAE", round(baseline_mae, 2))
        col3.metric("Model Bias", round(model_bias, 2))

        if model_mae < baseline_mae:
            st.success("The model is performing better than the historical average baseline.")
        else:
            st.warning("The historical average baseline is performing better than the model.")

        st.write(
            "This model predicts the expected total number of appointments for a selected date."
        )

        # Future prediction input
        st.subheader("Predict Appointments for a Future Day")

        default_future_date = (df["appointment_date"].max() + pd.Timedelta(days=7)).date()

        selected_date = st.date_input(
            "Select Date",
            value=default_future_date
        )

        selected_date = pd.to_datetime(selected_date)

        future_day_data = pd.DataFrame({
            "year": [selected_date.year],
            "month_number": [selected_date.month],
            "day": [selected_date.day],
            "week_of_year": [selected_date.isocalendar().week],
            "day_of_week": [selected_date.day_name()]
        })

        future_day_data = pd.get_dummies(
            future_day_data,
            columns=["day_of_week"]
        )

        # Match future input columns to training columns
        future_day_data = future_day_data.reindex(columns=X_daily.columns, fill_value=0)

        predicted_daily_appointments = daily_model.predict(future_day_data)[0]

        st.metric(
            "Predicted Total Appointments",
            round(predicted_daily_appointments, 1)
        )

    else:
        st.warning("Not enough daily data to train the forecasting model.")

# Message shown before any file is uploaded
else:
    st.info("Upload the Patient Scheduled Appts CSV to begin.")