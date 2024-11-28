import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

def execute_plt_code(code: str, df: pd.DataFrame):
    try:
        local_vars = {"df": df, "plt": plt, "sns": sns, "pd": pd}
        compile_code = compile(code, "<string>", "exec")
        exec(compile_code, globals(), local_vars)
        return plt.gcf()
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None