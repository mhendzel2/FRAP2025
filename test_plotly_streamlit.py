"""
Minimal test to verify Plotly + Streamlit compatibility
"""
import streamlit as st
import plotly.graph_objects as go

st.title("Plotly + Streamlit Test")

st.write(f"Streamlit version: {st.__version__}")

try:
    import plotly
    st.write(f"Plotly version: {plotly.__version__}")
except Exception as e:
    st.error(f"Failed to import plotly: {e}")

st.header("Test 1: Basic Plotly Chart")
try:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[1, 2, 3, 4, 5],
        y=[1, 4, 9, 16, 25],
        mode='lines+markers',
        name='Test Line'
    ))
    fig.update_layout(
        title='Simple Test Chart',
        xaxis_title='X',
        yaxis_title='Y²'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ Chart displayed successfully!")
    
except Exception as e:
    st.error(f"❌ Failed to display chart: {e}")
    st.exception(e)

st.header("Test 2: Plotly Express")
try:
    import plotly.express as px
    import pandas as pd
    
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })
    
    fig2 = px.scatter(df, x='x', y='y', title='Express Test')
    st.plotly_chart(fig2, use_container_width=True)
    st.success("✅ Plotly Express chart displayed!")
    
except Exception as e:
    st.error(f"❌ Failed to display Express chart: {e}")
    st.exception(e)
