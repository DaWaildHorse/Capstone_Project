import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from model import GDPPredictor
st.set_page_config(layout="wide")


df = pd.read_csv("./data/GDP_df.csv")

# Calculate total across years to sort by overall value
df['Total'] = df[['2012', '2013', '2014', '2015']].sum(axis=1)


def prepare_long_data(df_subset):
    """Convert wide dataframe subset to long format for Altair plotting."""
    return df_subset.melt(id_vars=['Country Name'], value_vars=['2012', '2013', '2014', '2015'],
                         var_name='Year', value_name='Value')


def create_stacked_bar_chart(data_long, title):
    """Create Altair stacked horizontal bar chart with value labels."""
    base = (
        alt.Chart(data_long)
        .mark_bar()
        .encode(
            y=alt.Y('Country Name:N', sort=alt.EncodingSortField(field='Value', op='sum', order='descending')),
            x=alt.X('Value:Q'),
            color=alt.Color('Year:N', scale=alt.Scale(scheme='category10')),
            tooltip=['Country Name', 'Year', 'Value']
        )
    )

    text = (
        alt.Chart(data_long)
        .mark_text(dx=3, dy=0, color='black', fontSize=10)
        .encode(
            y=alt.Y('Country Name:N', sort=alt.EncodingSortField(field='Value', op='sum', order='descending')),
            x=alt.X('Value:Q', stack='center'),
            detail='Year:N',
            text=alt.Text('Value:Q', format='.3f')
        )
    )

    chart = (base + text).properties(width=700, height=300, title=title)
    return chart

def plot_top_bottom(df, year):
    # Sort values for the year
    sorted_df = df[['Country Name', year]].sort_values(by=year)
    
    # Select bottom 5 and top 5
    bottom_5 = sorted_df.head(5)
    top_5 = sorted_df.tail(5)
    
    # Combine
    combined = pd.concat([bottom_5, top_5])
    
    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(combined['Country Name'], combined[year], color=['red']*5 + ['green']*5)
    
    ax.set_xlabel('Value')
    ax.set_title(f'Top 5 and Bottom 5 Countries in {year}')
    plt.tight_layout()
    st.pyplot(fig)



def prepare_long_data(df_subset):
    """Convert wide dataframe subset to long format for Altair plotting."""
    return df_subset.melt(id_vars=['Country Name'], value_vars=['2012', '2013', '2014', '2015'],
                         var_name='Year', value_name='Value')


def create_stacked_bar_chart(data_long, title):
    """Create Altair stacked horizontal bar chart with value labels."""
    base = (
        alt.Chart(data_long)
        .mark_bar()
        .encode(
            y=alt.Y('Country Name:N', sort=alt.EncodingSortField(field='Value', op='sum', order='descending')),
            x=alt.X('Value:Q'),
            color=alt.Color('Year:N', scale=alt.Scale(scheme='category10')),
            tooltip=['Country Name', 'Year', 'Value']
        )
    )

   

    chart = (base).properties(width=700, height=300, title=title)
    return chart



# -------------------------------
# Function: Get top N countries for a given metric
# -------------------------------
def get_top_countries(df, metric, top_n=5, mode='mean'):
    if mode == 'mean':
        top_countries = df.groupby('Country')[metric].mean().nlargest(top_n).index
    elif mode == 'sum':
        top_countries = df.groupby('Country')[metric].sum().nlargest(top_n).index
    else:
        raise ValueError("mode must be 'mean' or 'sum'")
    
    return df[df['Country'].isin(top_countries)]

# -------------------------------
# Function: Create a line chart
# -------------------------------
def create_line_chart(df, metric, title="Line Chart"):
    df['Year'] = df['Year'].astype(str)
    
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X('Year:N', title='Year'),
            y=alt.Y(f'{metric}:Q', title=metric),
            color=alt.Color('Country:N'),
            tooltip=['Country', 'Year', metric]
        )
        .properties(
            width=800,
            height=400,
            title=title
        )
    )
    
    return chart





st.markdown("""
# Estimate life insurance premiums under changing mortality and economic conditions (Prediction problem)
""")
st.dataframe(df)


st.markdown("""
This dataset was extracted from [World Bank Group](https://databank.worldbank.org/source/global-financial-development/Series/GFDD.DI.09#) 
""")


st.title("Data Analysis")

# Select top 5 and bottom 5 by total
top5 = df.sort_values(by='Total', ascending=False).head(5)
bottom5 = df.sort_values(by='Total', ascending=True).head(5)

# Prepare data for plotting
top5_long = prepare_long_data(top5)
bottom5_long = prepare_long_data(bottom5)

# Create charts
chart_top5 = create_stacked_bar_chart(top5_long, "Top 5 Countries by Life Insurance Premium to GDP ")
chart_bottom5 = create_stacked_bar_chart(bottom5_long, "Bottom 5 Countries by Life Insurance Premium to GDP ")

st.altair_chart(chart_top5, use_container_width=True)
st.altair_chart(chart_bottom5, use_container_width=True)

# Load your dataset here (for demo, we'll assume df is already defined)
df = pd.read_csv("./data/final.csv")

# Example placeholder: Ensure to replace this with your actual data
# df = pd.read_csv("your_data.csv")
# Example columns: ['Country', 'Year', ..., 'Avg Adult Mortality', ...]

# Example: Show top 5 countries by Avg Adult Mortality over time
metric = "Avg Adult Mortality"
top_df = get_top_countries(df, metric, top_n=5, mode='mean')
line_chart = create_line_chart(top_df, metric, title=f"Top 5 Countries by {metric}")

st.altair_chart(line_chart, use_container_width=True)

st.title("RNN With LSTM")

# Load data
df = pd.read_csv('./data/final.csv')

# Initialize predictor
predictor = GDPPredictor(sequence_length=3)

# Prepare the same way
predictor.prepare_data(df)


import joblib

# Load scalers before loading model or before prediction
predictor.scaler_features = joblib.load('App/scaler_features.save')
predictor.scaler_target = joblib.load('App/scaler_target.save')

# Load the model
predictor.load('App/gdp_model.keras')

import streamlit as st
import pandas as pd

# Assuming your GDPPredictor class is already imported and available as predictor
# Also assume your original dataframe 'df' is loaded and ready

# Load or initialize your predictor and data outside the Streamlit UI flow
# For example:
# predictor = GDPPredictor(sequence_length=3)
# predictor.load('your_model.keras')  # Or however you load it

st.title("GDP Percent Multi-Country Forecast")

# Get the list of countries from your dataframe
countries_list = df['Country'].unique().tolist()

# Multiselect widget for countries
selected_countries = st.multiselect("Select countries to predict", countries_list, default=['China'])

# Slider for how many years ahead to predict
future_years = st.slider("Years to predict ahead", min_value=1, max_value=10, value=3)
print(df.columns)

import plotly.graph_objects as go

if st.button("Predict GDP Percent"): 
    if not selected_countries:
        st.warning("Please select at least one country")
    else:
        with st.spinner("Predicting..."):
            predictions_df = predictor.predict_multiple_countries(
                selected_countries, df, future_years=future_years
            )

            if predictions_df.empty:
                st.write("No predictions were made.")
            else:
                st.write("### Predictions for selected countries:")
                st.dataframe(predictions_df)

                # Clean column names
                predictions_df.columns = predictions_df.columns.str.strip()
                df.columns = df.columns.str.strip()

                for country in selected_countries:
                    # Get historical and predicted
                    hist_data = df[df['Country'] == country][['Year', 'GDP_Percent']].dropna().sort_values('Year')
                    pred_data = predictions_df[predictions_df['Country'] == country][['Year', 'Predicted_GDP_Percent']].dropna().sort_values('Year')

                    if hist_data.empty or pred_data.empty:
                        continue

                    # Rename predicted column to match historical
                    pred_data = pred_data.rename(columns={'Predicted_GDP_Percent': 'GDP_Percent'})

                    # Add flag column to mark predicted rows
                    hist_data['Type'] = 'Historical'
                    pred_data['Type'] = 'Predicted'

                    import numpy as np

                    # Merge both datasets
                    
                    import plotly.graph_objects as go

                    # Merge historical and predicted
                    hist_data['type'] = 'historical'
                    pred_data['type'] = 'predicted'
                    combined = pd.concat([hist_data, pred_data], ignore_index=True).sort_values('Year')

                    fig = go.Figure()

                    # Full line (unified view: solid + dashed)
                    fig.add_trace(go.Scatter(
                        x=combined['Year'],
                        y=combined['GDP_Percent'],
                        mode='lines+markers',
                        name='GDP Percent',
                        line=dict(color='green'),
                    ))

                    # Historical area fill
                    fig.add_trace(go.Scatter(
                        x=combined[combined['type'] == 'historical']['Year'],
                        y=combined[combined['type'] == 'historical']['GDP_Percent'],
                        mode='lines',
                        line=dict(color='green'),
                        name='Historical Area',
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 0, 0.2)',
                        hoverinfo='skip',
                        showlegend=True
                    ))

                    # Predicted area fill
                    fig.add_trace(go.Scatter(
                        x=combined[combined['type'] == 'predicted']['Year'],
                        y=combined[combined['type'] == 'predicted']['GDP_Percent'],
                        mode='lines',
                        line=dict(color='orange', dash='dash'),
                        name='Predicted Area',
                        fill='tozeroy',
                        fillcolor='rgba(255, 165, 0, 0.2)',
                        hoverinfo='skip',
                        showlegend=True
                    ))

                    # Highlight only between 2015–2016 following the line shape
                    highlight = combined[(combined['Year'] >= 2015) & (combined['Year'] <= 2016)]
                    if not highlight.empty:
                        fig.add_trace(go.Scatter(
                            x=highlight['Year'],
                            y=highlight['GDP_Percent'],
                            mode='lines',
                            line=dict(color='orange'),
                            name='2015–2016 Highlight',
                            fill='tozeroy',
                            fillcolor='rgba(255, 165, 0, 0.2)',
                            hoverinfo='skip',
                            showlegend=False
                        ))

                    fig.update_layout(
                        title=f"GDP Percent for {country}",
                        xaxis_title="Year",
                        yaxis_title="GDP Percent",
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)
