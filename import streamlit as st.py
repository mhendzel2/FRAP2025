import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="FRAP - Financial Ratios Analysis Platform", page_icon=":bar_chart:", layout="wide")

# --- Helper Functions ---

@st.cache_data
def load_data(file):
    """Load data from an Excel file."""
    xls = pd.ExcelFile(file)
    sheet_names = xls.sheet_names
    data = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in sheet_names}
    return data

@st.cache_data
def calculate_ratios(df):
    """Calculate financial ratios."""
    # Example calculations - these will vary based on actual requirements
    df['Current Ratio'] = df['Current Assets'] / df['Current Liabilities']
    df['Quick Ratio'] = (df['Current Assets'] - df['Inventories']) / df['Current Liabilities']
    df['Debt to Equity'] = df['Total Liabilities'] / df['Shareholders Equity']
    df['Return on Equity'] = df['Net Income'] / df['Shareholders Equity']
    return df

# --- File Upload ---
st.sidebar.header("Upload your data")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xls", "xlsx"])

if uploaded_file is not None:
    # Load and cache the data
    data = load_data(uploaded_file)
    
    # --- Data Overview ---
    st.title("Financial Data Overview")
    for sheet_name, df in data.items():
        st.subheader(f"Sheet: {sheet_name}")
        st.write(df)
        
        # --- Ratio Calculation ---
        st.subheader("Calculated Ratios")
        df_ratios = calculate_ratios(df)
        st.write(df_ratios)
        
        # --- Download Link for Processed Data ---
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(df_ratios)
        st.download_button(
            label="Download processed data as CSV",
            data=csv,
            file_name=f"{sheet_name}_processed_data.csv",
            mime="text/csv",
            key=f"download-csv-{sheet_name}"
        )
        
        # --- Visualization ---
        st.subheader("Data Visualization")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            x_axis = st.selectbox("Select X-axis column", numeric_cols, key=f"xaxis-{sheet_name}")
            y_axis = st.selectbox("Select Y-axis column", numeric_cols, key=f"yaxis-{sheet_name}")
            
            fig, ax = plt.subplots()
            ax.scatter(df[x_axis], df[y_axis])
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"{y_axis} vs {x_axis}")
            st.pyplot(fig)
            
            # Correlation heatmap
            if st.checkbox("Show correlation heatmap", key=f"heatmap-{sheet_name}"):
                fig, ax = plt.subplots()
                sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", ax=ax)
                st.pyplot(fig)
        
        # --- Session State Example ---
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = uploaded_file.name
        st.write(f"Uploaded file: {st.session_state.uploaded_file_name}")
        
        # Save session data
        if st.button("Save session data"):
            session_filename = f"FRAP_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(session_filename, 'wb') as f:
                pickle.dump(st.session_state, f)
            st.success(f"Session data saved as {session_filename}")
        
        # Load session data
        if st.button("Load session data"):
            session_files = [f for f in os.listdir() if f.startswith("FRAP_Session_") and f.endswith(".pkl")]
            if session_files:
                selected_file = st.selectbox("Select a session file", session_files)
                with open(selected_file, 'rb') as f:
                    st.session_state.update(pickle.load(f))
                st.success(f"Session data loaded from {selected_file}")
            else:
                st.warning("No session files found. Save a session first.")

# --- Footer ---
st.sidebar.info("FRAP - Financial Ratios Analysis Platform")
st.sidebar.text("Created by: Your Name")
st.sidebar.text("Date: 2023")